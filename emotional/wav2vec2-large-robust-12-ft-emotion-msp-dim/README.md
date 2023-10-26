---
language: en
datasets:
- msp-podcast
inference: true
tags:
- speech
- audio
- wav2vec2
- audio-classification
- emotion-recognition
license: cc-by-nc-sa-4.0
pipeline_tag: audio-classification
---

# Model for Dimensional Speech Emotion Recognition based on Wav2vec 2.0

The model expects a raw audio signal as input and outputs predictions for arousal, dominance and valence in a range of approximately 0...1. In addition, it also provides the pooled states of the last transformer layer. The model was created by fine-tuning [
Wav2Vec2-Large-Robust](https://huggingface.co/facebook/wav2vec2-large-robust) on [MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) (v1.7). The model was pruned from 24 to 12 transformer layers before fine-tuning. An [ONNX](https://onnx.ai/") export of the model is available from [doi:10.5281/zenodo.6221127](https://zenodo.org/record/6221127). Further details are given in the associated [paper](https://arxiv.org/abs/2203.07378) and [tutorial](https://github.com/audeering/w2v2-how-to).

# Usage

```python
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


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



# load model from hub
device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)

# dummy signal
sampling_rate = 16000
signal = np.zeros((1, sampling_rate), dtype=np.float32)


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


print(process_func(signal, sampling_rate))
#  Arousal    dominance valence
# [[0.5460754  0.6062266  0.40431657]]

print(process_func(signal, sampling_rate, embeddings=True))
# Pooled hidden states of last transformer layer
# [[-0.00752167  0.0065819  -0.00746342 ...  0.00663632  0.00848748
#    0.00599211]]
```
