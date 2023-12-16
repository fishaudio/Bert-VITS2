import sys

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DebertaV2Model,
    DebertaV2Tokenizer,
    ClapModel,
    ClapProcessor,
)

from config import config
from text.japanese import text2sep_kata


class BertFeature:
    def __init__(self, model_path, language="ZH"):
        self.model_path = model_path
        self.language = language
        self.tokenizer = None
        self.model = None
        self.device = None

        self._prepare()

    def _get_device(self, device=config.bert_gen_config.device):
        if (
            sys.platform == "darwin"
            and torch.backends.mps.is_available()
            and device == "cpu"
        ):
            device = "mps"
        if not device:
            device = "cuda"
        return device

    def _prepare(self):
        self.device = self._get_device()

        if self.language == "EN":
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)
            self.model = DebertaV2Model.from_pretrained(self.model_path).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path).to(
                self.device
            )
        self.model.eval()

    def get_bert_feature(self, text, word2ph):
        if self.language == "JP":
            text = "".join(text2sep_kata(text)[0])
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

        word2phone = word2ph
        phone_level_feature = []
        for i in range(len(word2phone)):
            repeat_feature = res[i].repeat(word2phone[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        return phone_level_feature.T


class ClapFeature:
    def __init__(self, model_path):
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.device = None

        self._prepare()

    def _get_device(self, device=config.bert_gen_config.device):
        if (
            sys.platform == "darwin"
            and torch.backends.mps.is_available()
            and device == "cpu"
        ):
            device = "mps"
        if not device:
            device = "cuda"
        return device

    def _prepare(self):
        self.device = self._get_device()

        self.processor = ClapProcessor.from_pretrained(self.model_path)
        self.model = ClapModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()

    def get_clap_audio_feature(self, audio_data):
        with torch.no_grad():
            inputs = self.processor(
                audios=audio_data, return_tensors="pt", sampling_rate=48000
            ).to(self.device)
            emb = self.model.get_audio_features(**inputs)
        return emb.T

    def get_clap_text_feature(self, text):
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            emb = self.model.get_text_features(**inputs)
        return emb.T
