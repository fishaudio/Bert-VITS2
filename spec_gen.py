import torch
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from tqdm import tqdm

from text import cleaned_text_to_sequence, get_bert

config_path = 'configs/config.json'
hps = utils.get_hparams_from_file(config_path)

train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)

collate_fn = TextAudioSpeakerCollate()
train_loader = DataLoader(train_dataset, num_workers=12, shuffle=False,
                         batch_size=32, pin_memory=True,
                         drop_last=False, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, num_workers=12, shuffle=False,
                         batch_size=32, pin_memory=True,
                         drop_last=False, collate_fn=collate_fn)
for _ in tqdm(train_loader):
    pass
for _ in tqdm(eval_loader):
    pass

# for line in tqdm( open(hps.data.training_files).readlines()):
#     _id, spk, language_str, text, phones, tone, word2ph = line.strip().split("|")
#     phone = phones.split(" ")
#     tone = [int(i) for i in tone.split(" ")]
#     word2ph = [int(i) for i in word2ph.split(" ")]
#     # print(text, word2ph,phone, tone, language_str)
#     w2pho = [i for i in word2ph]
#     word2ph = [i for i in word2ph]
#     phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
#     pold2 = phone

#     if hps.data.add_blank:
#         phone = commons.intersperse(phone, 0)
#         tone = commons.intersperse(tone, 0)
#         language = commons.intersperse(language, 0)
#         for i in range(len(word2ph)):
#             word2ph[i] = word2ph[i] * 2
#         word2ph[0] += 1
#     wav_path = f'dataset/{spk}/{_id}.wav'

#     bert_path = wav_path.replace(".wav", ".bert.pt")
#     try:
#         bert = torch.load(bert_path)
#         assert bert.shape[-1] == len(phone)
#     except:
#         bert = get_bert(text, word2ph, language_str)
#         assert bert.shape[-1] == len(phone)
#         torch.save(bert, bert_path)

