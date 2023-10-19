import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-historic-english-cased")
model = AutoModelForMaskedLM.from_pretrained("./bert/bert-base-historic-english-cased")


def get_bert_feature(text, word2ph, device=None):
    with torch.no_grad():
        if device:
            model_new = model.to(device)
        else:
            model_new = model
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            if device:
                inputs[i] = inputs[i].to(device)

        res = model_new(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    # assert len(word2ph) == len(text)+2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
