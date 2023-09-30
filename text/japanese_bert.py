import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys

BERT = "./bert/bert-large-japanese-v2"

tokenizer = AutoTokenizer.from_pretrained(BERT)
# bert-large model has 25 hidden layers.You can decide which layer to use by setting this variable to a specific value
# default value is 3(untested)
BERT_LAYER = 20
models = dict()


def get_bert_feature(text, word2ph, device=None):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(BERT).to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)
        res = res["hidden_states"][BERT_LAYER]
    # assert inputs["input_ids"].shape[-1] == len(word2ph)
    # word2phone = word2ph
    phone_level_feature = []
    # for i in range(len(word2phone)):
    #    repeat_feature = res[0][i].repeat(word2phone[i], 1)
    #    phone_level_feature.append(repeat_feature)

    # phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return res[0]
