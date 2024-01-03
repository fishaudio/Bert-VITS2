import sys

import torch
from transformers import ClapModel, ClapProcessor

from config import config

device = "cuda"
processor = ClapProcessor.from_pretrained("./emotional/clap-htsat-fused")
models = ClapModel.from_pretrained("./emotional/clap-htsat-fused").to(device)

def get_clap_audio_feature(audio_data, device=config.bert_gen_config.device):
    with torch.no_grad():
        inputs = processor(
            audios=audio_data, return_tensors="pt", sampling_rate=48000
        ).to(device)
        emb = models[device].get_audio_features(**inputs)
    return emb.T


def get_clap_text_feature(text, device=config.bert_gen_config.device):
    with torch.no_grad():
        inputs = processor(text=text, return_tensors="pt").to(device)
        emb = models[device].get_text_features(**inputs)
    return emb.T
