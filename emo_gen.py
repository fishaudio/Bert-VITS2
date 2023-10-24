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
import torch.multiprocessing as mp


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


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    model: EmotionModel,
    processor: Wav2Vec2Processor,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""
    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
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


wavnames = []


def extract_dir(data_queue, model, processor):
    while True:
        data = data_queue.get()
        if data is None:
            break
        wavname = data.split("|")[0]  # 获取每一行的第一部分
        emo_path = wavname.replace(".wav", ".emo.npy")
        if os.path.exists(emo_path):
            continue
        wav, sr = librosa.load(wavname, 16000)
        emb = process_func(
            np.expand_dims(wav, 0), sr, model, processor, embeddings=True
        )
        wavnames.append(wavname)
        np.save(emo_path, emb.squeeze(0))
        print(f"{emo_path} 生成完毕！")
        del data, wav, sr, emb


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
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name)

    lines = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    processes = []
    data_queue = mp.Queue()

    # 将数据放入队列
    for data in lines:
        data_queue.put(data)

    for _ in range(args.num_processes):  # 创建工作进程
        p = mp.Process(target=extract_dir, args=(data_queue, model, processor))
        p.start()
        processes.append(p)

    # 等待所有工作进程完成
    for p in processes:
        p.join()

    print(f"Emo vec 生成完毕!, 共有 {len(wavnames)} 个 emo.npy 生成!")
