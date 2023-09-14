import os
import random
import torch
import torch.utils.data
from tqdm import tqdm
from loguru import logger
import commons
from mel_processing import spectrogram_torch, mel_spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import cleaned_text_to_sequence

"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 300)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        logger.info("Init dataset...")

        for data in tqdm(self.audiopaths_sid_text):
            audiopath, phones = data["path"], data["phones"]
            if self.min_text_len <= len(phones) and len(phones) <= self.max_text_len and os.path.exists(audiopath):
                audiopaths_sid_text_new.append(data)
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            else:
                skipped += 1

        logger.info(
            f"Skipped: {skipped} because of text length, total: {len(self.audiopaths_sid_text)}"
        )

        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, data):
        sid = torch.LongTensor([int(self.spk_map[data["spk"]])])
        spec, wav = self.get_audio(data["path"])

        phones, tones, languages = cleaned_text_to_sequence(
            data["phones"], data["tones"], data["languages"]
        )
        token_ids, offsets = data["token_ids"], data["offsets"]

        # Convert offsets to mapping
        phones2tokens = [0] * len(phones)  # All use CLS by default
        for i in range(len(offsets)):
            if offsets[i] is None:
                continue

            start, end = offsets[i]
            for j in range(start, end):
                phones2tokens[j] = i

        if self.add_blank:
            phones = commons.intersperse(phones, 0)
            tones = commons.intersperse(tones, 0)
            languages = commons.intersperse(languages, 0)
            phones2tokens = commons.intersperse(phones2tokens, 0)

            # Don't intersperse tokens since they will be handled by Bert

        assert len(phones) == len(tones) == len(languages) == len(phones2tokens)

        phones = torch.LongTensor(phones)
        tones = torch.LongTensor(tones)
        languages = torch.LongTensor(languages)
        token_ids = torch.LongTensor(token_ids)
        phones2tokens = torch.LongTensor(phones2tokens)

        return dict(
            phones=phones,
            spec=spec,
            wav=wav,
            sid=sid,
            tones=tones,
            languages=languages,
            token_ids=token_ids,
            phones2tokens=phones2tokens,
        )

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    filename, sampling_rate, self.sampling_rate
                )
            )
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")
        try:
            spec = torch.load(spec_filename)
        except:
            if self.use_mel_spec_posterior:
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [{phones, spec, wav, sid, tones, languages, token_ids, phones2tokens}]
        """

        # Right zero-pad all one-hot text sequences to max input length
        batch = sorted(batch, key=lambda x: x["spec"].size(1), reverse=True)

        max_text_len = max([len(x["phones"]) for x in batch])
        max_spec_len = max([x["spec"].size(1) for x in batch])
        max_wav_len = max([x["wav"].size(1) for x in batch])
        max_token_ids_len = max([len(x["token_ids"]) for x in batch])

        text_lengths = torch.zeros(len(batch), dtype=torch.long)
        spec_lengths = torch.zeros(len(batch), dtype=torch.long)
        wav_lengths = torch.zeros(len(batch), dtype=torch.long)
        sid = torch.zeros(len(batch), dtype=torch.long)

        text_padded = torch.zeros((len(batch), max_text_len), dtype=torch.long)
        tone_padded = torch.zeros((len(batch), max_text_len), dtype=torch.long)
        language_padded = torch.zeros((len(batch), max_text_len), dtype=torch.long)
        phones2tokens_padded = torch.zeros((len(batch), max_text_len), dtype=torch.long)

        spec_padded = torch.zeros(
            (len(batch), batch[0]["spec"].size(0), max_spec_len), dtype=torch.float
        )
        wav_padded = torch.zeros((len(batch), 1, max_wav_len), dtype=torch.float)

        token_ids_padded = torch.zeros(
            (len(batch), max_token_ids_len), dtype=torch.long
        )
        tokens_attention_mask = torch.zeros(
            (len(batch), max_token_ids_len), dtype=torch.long
        )

        for i, row in enumerate(batch):
            text = row["phones"]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row["spec"]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row["wav"]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row["sid"]

            tone = row["tones"]
            tone_padded[i, : tone.size(0)] = tone

            language = row["languages"]
            language_padded[i, : language.size(0)] = language

            phones2tokens = row["phones2tokens"]
            phones2tokens_padded[i, : phones2tokens.size(0)] = phones2tokens

            token_ids = row["token_ids"]
            token_ids_padded[i, : token_ids.size(0)] = token_ids
            tokens_attention_mask[i, : token_ids.size(0)] = 1

        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
            tone_padded,
            language_padded,
            token_ids_padded,
            tokens_attention_mask,
            phones2tokens_padded,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            print("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
