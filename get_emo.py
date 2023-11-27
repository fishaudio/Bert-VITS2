from emo_gen import EmotionModel, process_func

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor

from config import config

model_name = "./emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)


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
