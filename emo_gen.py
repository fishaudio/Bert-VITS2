import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import librosa
import numpy as np
import argparse
from config import config
import utils
import os
from tqdm import tqdm


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
        self,
        input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


class AudioDataset(Dataset):
    def __init__(self, list_of_wav_files, sr, processor):
        self.list_of_wav_files = list_of_wav_files
        self.processor = processor
        self.sr = sr

    def __len__(self):
        return len(self.list_of_wav_files)

    def __getitem__(self, idx):
        wav_file = self.list_of_wav_files[idx]
        audio_data, _ = librosa.load(wav_file, sr=self.sr)
        processed_data = self.processor(audio_data, sampling_rate=self.sr)[
            "input_values"
        ][0]
        return torch.from_numpy(processed_data)


model_name = "./emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    model: EmotionModel,
    processor: Wav2Vec2Processor,
    device: str,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""
    model = model.to(device)
    y = processor(x, sampling_rate=sampling_rate)
    y = y["input_values"][0]
    y = torch.from_numpy(y).unsqueeze(0).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


def get_emo(path):
    wav, sr = librosa.load(path, 16000)
    device = config.bert_gen_config.device
    return process_func(
        np.expand_dims(wav, 0).astype(np.float),
        sr,
        model,
        processor,
        device,
        embeddings=True,
    ).squeeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.bert_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes", type=int, default=config.bert_gen_config.num_processes
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    hps = utils.get_hparams_from_file(config_path)

    device = config.bert_gen_config.device

    model_name = "./emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    processor = (
        Wav2Vec2Processor.from_pretrained(model_name)
        if processor is None
        else processor
    )
    model = (
        EmotionModel.from_pretrained(model_name).to(device)
        if model is None
        else model.to(device)
    )

    lines = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    wavnames = [line.split("|")[0] for line in lines]
    dataset = AudioDataset(wavnames, 16000, processor)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            wavname = wavnames[i]
            emo_path = wavname.replace(".wav", ".emo.npy")
            if os.path.exists(emo_path):
                continue
            emb = model(data.to(device))[0].detach().cpu().numpy()
            np.save(emo_path, emb)

    print("Emo vec 生成完毕!")
