---
license: apache-2.0
datasets:
- cc100
- wikipedia
language:
- ja
widget:
- text: 東北大学で[MASK]の研究をしています。
---

# BERT base Japanese (unidic-lite with whole word masking, CC-100 and jawiki-20230102)

This is a [BERT](https://github.com/google-research/bert) model pretrained on texts in the Japanese language.

This version of the model processes input texts with word-level tokenization based on the Unidic 2.1.2 dictionary (available in [unidic-lite](https://pypi.org/project/unidic-lite/) package), followed by the WordPiece subword tokenization.
Additionally, the model is trained with the whole word masking enabled for the masked language modeling (MLM) objective.

The codes for the pretraining are available at [cl-tohoku/bert-japanese](https://github.com/cl-tohoku/bert-japanese/).

## Model architecture

The model architecture is the same as the original BERT base model; 12 layers, 768 dimensions of hidden states, and 12 attention heads.

## Training Data

The model is trained on the Japanese portion of [CC-100 dataset](https://data.statmt.org/cc-100/) and the Japanese version of Wikipedia.
For Wikipedia, we generated a text corpus from the [Wikipedia Cirrussearch dump file](https://dumps.wikimedia.org/other/cirrussearch/) as of January 2, 2023.
The corpus files generated from CC-100 and Wikipedia are 74.3GB and 4.9GB in size and consist of approximately 392M and 34M sentences, respectively.

For the purpose of splitting texts into sentences, we used [fugashi](https://github.com/polm/fugashi) with [mecab-ipadic-NEologd](https://github.com/neologd/mecab-ipadic-neologd) dictionary (v0.0.7).

## Tokenization

The texts are first tokenized by MeCab with the Unidic 2.1.2 dictionary and then split into subwords by the WordPiece algorithm.
The vocabulary size is 32768.

We used [fugashi](https://github.com/polm/fugashi) and [unidic-lite](https://github.com/polm/unidic-lite) packages for the tokenization.

## Training

We trained the model first on the CC-100 corpus for 1M steps and then on the Wikipedia corpus for another 1M steps.
For training of the MLM (masked language modeling) objective, we introduced whole word masking in which all of the subword tokens corresponding to a single word (tokenized by MeCab) are masked at once.

For training of each model, we used a v3-8 instance of Cloud TPUs provided by [TPU Research Cloud](https://sites.research.google/trc/about/).

## Licenses

The pretrained models are distributed under the Apache License 2.0.

## Acknowledgments

This model is trained with Cloud TPUs provided by [TPU Research Cloud](https://sites.research.google/trc/about/) program.
