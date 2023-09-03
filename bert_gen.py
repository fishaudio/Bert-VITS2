import torch
from torch.utils.data import DataLoader
from multiprocessing import Pool
import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from tqdm import tqdm
import warnings

from text import cleaned_text_to_sequence, get_bert

config_path = 'configs/config.json'
hps = utils.get_hparams_from_file(config_path)

def process_line(line):
    _id, spk, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    w2pho = [i for i in word2ph]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    wav_path = f'{_id}'

    bert_path = wav_path.replace(".wav", ".bert.pt")
    try:
        bert = torch.load(bert_path)
        assert bert.shape[-1] == len(phone)
    except:
        bert = get_bert(text, word2ph, language_str)
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_path)


if __name__ == '__main__':
    lines = []
    with open(hps.data.training_files, encoding='utf-8' ) as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding='utf-8' ) as f:
        lines.extend(f.readlines())

    with Pool(processes=12) as pool: #A100 40GB suitable config,if coom,please decrease the processess number.
        for _ in tqdm(pool.imap_unordered(process_line, lines)):
            pass
