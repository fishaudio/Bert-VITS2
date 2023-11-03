import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import DebertaV2Model, DebertaV2Tokenizer

from config import config

REPO_ID = "microsoft/deberta-v3-large"
LOCAL_PATH = "./bert/deberta-v3-large"
FILES = ["spm.model", "pytorch_model.bin"]
for file in FILES:
    if not Path(LOCAL_PATH).joinpath(file).exists():
        hf_hub_download(
            REPO_ID, file, local_dir=LOCAL_PATH, local_dir_use_symlinks=False
        )

tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)

models = dict()


def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = DebertaV2Model.from_pretrained(LOCAL_PATH).to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # assert len(word2ph) == len(text)+2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
