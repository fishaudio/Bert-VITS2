---
language: ja
license: cc-by-sa-4.0
library_name: transformers
tags:
  - deberta
  - deberta-v2
  - fill-mask
datasets:
  - wikipedia
  - cc100
  - oscar
metrics:
  - accuracy
mask_token: "[MASK]"
widget:
  - text: "京都 大学 で 自然 言語 処理 を [MASK] する 。"
---

# Model Card for Japanese DeBERTa V2 large

## Model description

This is a Japanese DeBERTa V2 large model pre-trained on Japanese Wikipedia, the Japanese portion of CC-100, and the
Japanese portion of OSCAR.

## How to use

You can use this model for masked language modeling as follows:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('ku-nlp/deberta-v2-large-japanese')
model = AutoModelForMaskedLM.from_pretrained('ku-nlp/deberta-v2-large-japanese')

sentence = '京都 大学 で 自然 言語 処理 を [MASK] する 。'  # input should be segmented into words by Juman++ in advance
encoding = tokenizer(sentence, return_tensors='pt')
...
```

You can also fine-tune this model on downstream tasks.

## Tokenization

The input text should be segmented into words by [Juman++](https://github.com/ku-nlp/jumanpp) in
advance. [Juman++ 2.0.0-rc3](https://github.com/ku-nlp/jumanpp/releases/tag/v2.0.0-rc3) was used for pre-training. Each
word is tokenized into subwords by [sentencepiece](https://github.com/google/sentencepiece).

## Training data

We used the following corpora for pre-training:

- Japanese Wikipedia (as of 20221020, 3.2GB, 27M sentences, 1.3M documents)
- Japanese portion of CC-100 (85GB, 619M sentences, 66M documents)
- Japanese portion of OSCAR (54GB, 326M sentences, 25M documents)

Note that we filtered out documents annotated with "header", "footer", or "noisy" tags in OSCAR.
Also note that Japanese Wikipedia was duplicated 10 times to make the total size of the corpus comparable to that of
CC-100 and OSCAR. As a result, the total size of the training data is 171GB.

## Training procedure

We first segmented texts in the corpora into words using [Juman++](https://github.com/ku-nlp/jumanpp).
Then, we built a sentencepiece model with 32000 tokens including words ([JumanDIC](https://github.com/ku-nlp/JumanDIC))
and subwords induced by the unigram language model of [sentencepiece](https://github.com/google/sentencepiece).

We tokenized the segmented corpora into subwords using the sentencepiece model and trained the Japanese DeBERTa model
using [transformers](https://github.com/huggingface/transformers) library.
The training took 36 days using 8 NVIDIA A100-SXM4-40GB GPUs.

The following hyperparameters were used during pre-training:

- learning_rate: 1e-4
- per_device_train_batch_size: 18
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 16
- total_train_batch_size: 2,304
- max_seq_length: 512
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-06
- lr_scheduler_type: linear schedule with warmup
- training_steps: 300,000
- warmup_steps: 10,000

The accuracy of the trained model on the masked language modeling task was 0.799.
The evaluation set consists of 5,000 randomly sampled documents from each of the training corpora.

## Fine-tuning on NLU tasks

We fine-tuned the following models and evaluated them on the dev set of JGLUE.
We tuned learning rate and training epochs for each model and task
following [the JGLUE paper](https://www.jstage.jst.go.jp/article/jnlp/30/1/30_63/_pdf/-char/ja).

| Model                         | MARC-ja/acc | JSTS/pearson | JSTS/spearman | JNLI/acc | JSQuAD/EM | JSQuAD/F1 | JComQA/acc |
|-------------------------------|-------------|--------------|---------------|----------|-----------|-----------|------------|
| Waseda RoBERTa base           | 0.965       | 0.913        | 0.876         | 0.905    | 0.853     | 0.916     | 0.853      |
| Waseda RoBERTa large (seq512) | 0.969       | 0.925        | 0.890         | 0.928    | 0.910     | 0.955     | 0.900      |
| LUKE Japanese base*           | 0.965       | 0.916        | 0.877         | 0.912    | -         | -         | 0.842      |
| LUKE Japanese large*          | 0.965       | 0.932        | 0.902         | 0.927    | -         | -         | 0.893      |
| DeBERTaV2 base                | 0.970       | 0.922        | 0.886         | 0.922    | 0.899     | 0.951     | 0.873      |
| DeBERTaV2 large               | 0.968       | 0.925        | 0.892         | 0.924    | 0.912     | 0.959     | 0.890      |

*The scores of LUKE are from [the official repository](https://github.com/studio-ousia/luke).

## Acknowledgments

This work was supported by Joint Usage/Research Center for Interdisciplinary Large-scale Information Infrastructures (
JHPCN) through General Collaboration Project no. jh221004, "Developing a Platform for Constructing and Sharing of
Large-Scale Japanese Language Models".
For training models, we used the mdx: a platform for the data-driven future.
