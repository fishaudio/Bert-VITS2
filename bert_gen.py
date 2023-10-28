import torch
import torch.nn as nn
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
from pqdm.processes import pqdm


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


model_name = "./emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim"
global processor, model
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""
    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    global model
    m = model.to(device)
    y = processor(x, sampling_rate=sampling_rate)
    y = y["input_values"][0]
    y = torch.from_numpy(y).unsqueeze(0).to(device)

    # run through model
    with torch.no_grad():
        y = m(y)[0 if embeddings else 1]
    del m

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


def get_emo(path):
    wav, sr = librosa.load(path, 16000)
    device = config.bert_gen_config.device
    return process_func(
        np.expand_dims(wav, 0).astype(np.float),
        sr,
        embeddings=True,
    ).squeeze(0)


def extract_dir(data):
    try:
        wavname = data.split("|")[0]  # 获取每一行的第一部分
        emo_path = wavname.replace(".wav", ".emo.npy")
        if os.path.exists(emo_path):
            return f"{emo_path} 已存在！"
        wav, sr = librosa.load(wavname, 16000)
        emb = process_func(np.expand_dims(wav, 0), sr, embeddings=True)
        np.save(emo_path, emb.squeeze(0))
        del data, wav, sr, emb
    except Exception as e:
        return e


lines = []
with open(hps.data.training_files, encoding="utf-8") as f:
    lines.extend(f.readlines())

with open(hps.data.validation_files, encoding="utf-8") as f:
    lines.extend(f.readlines())

errors = pqdm(
    lines,
    extract_dir,
    n_jobs=4,
)

print(errors)

print("Emo vec 生成完毕!")
