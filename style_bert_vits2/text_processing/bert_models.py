"""
Style-Bert-VITS2 の学習・推論に必要な各言語ごとの BERT モデルをロード/取得するためのモジュール。

オリジナルの Bert-VITS2 では各言語ごとの BERT モデルが初回インポート時にハードコードされたパスから「暗黙的に」ロードされているが、
場合によっては多重にロードされて非効率なほか、BERT モデルのロード元のパスがハードコードされているためライブラリ化ができない。

そこで、ライブラリの利用前に、音声合成に利用する言語の BERT モデルだけを「明示的に」ロードできるようにした。
一度 load_model/tokenizer() で当該言語の BERT モデルがロードされていれば、ライブラリ内部のどこからでもロード済みのモデル/トークナイザーを取得できる。
"""

import gc
from typing import cast

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DebertaV2Model,
    DebertaV2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from style_bert_vits2.constants import DEFAULT_BERT_TOKENIZER_PATHS, Languages
from style_bert_vits2.logging import logger


# 各言語ごとのロード済みの BERT モデルを格納する辞書
__loaded_models: dict[Languages, PreTrainedModel | DebertaV2Model] = {}

# 各言語ごとのロード済みの BERT トークナイザーを格納する辞書
__loaded_tokenizers: dict[Languages, PreTrainedTokenizer | PreTrainedTokenizerFast | DebertaV2Tokenizer] = {}


def load_model(
    language: Languages,
    pretrained_model_name_or_path: str | None = None,
) -> PreTrainedModel | DebertaV2Model:
    """
    指定された言語の BERT モデルをロードし、ロード済みの BERT モデルを返す
    一度ロードされていれば、ロード済みの BERT モデルを即座に返す
    ライブラリ利用時は常に pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある
    ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき

    Style-Bert-VITS2 では、BERT モデルに下記の 3 つが利用されている
    これ以外の BERT モデルを指定した場合は正常に動作しない可能性が高い
    - 日本語: ku-nlp/deberta-v2-large-japanese-char-wwm
    - 英語: microsoft/deberta-v3-large
    - 中国語: hfl/chinese-roberta-wwm-ext-large

    Args:
        language (Languages): ロードする学習済みモデルの対象言語
        pretrained_model_name_or_path (str | None): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)

    Returns:
        PreTrainedModel | DebertaV2Model: ロード済みの BERT モデル
    """

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_models:
        return __loaded_models[language]

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        assert DEFAULT_BERT_TOKENIZER_PATHS[language].exists(), \
            f"The default {language} BERT model does not exist on the file system. Please specify the path to the pre-trained model."
        pretrained_model_name_or_path = str(DEFAULT_BERT_TOKENIZER_PATHS[language])

    # BERT モデルをロードし、辞書に格納して返す
    ## 英語のみ DebertaV2Model でロードする必要がある
    if language == Languages.EN:
        model = cast(DebertaV2Model, DebertaV2Model.from_pretrained(pretrained_model_name_or_path))
    else:
        model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
    __loaded_models[language] = model
    logger.info(f"Loaded the {language} BERT model from {pretrained_model_name_or_path}")

    return model


def load_tokenizer(
    language: Languages,
    pretrained_model_name_or_path: str | None = None,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast | DebertaV2Tokenizer:
    """
    指定された言語の BERT モデルをロードし、ロード済みの BERT トークナイザーを返す
    一度ロードされていれば、ロード済みの BERT トークナイザーを即座に返す
    ライブラリ利用時は常に pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある
    ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき

    Style-Bert-VITS2 では、BERT モデルに下記の 3 つが利用されている
    これ以外の BERT モデルを指定した場合は正常に動作しない可能性が高い
    - 日本語: ku-nlp/deberta-v2-large-japanese-char-wwm
    - 英語: microsoft/deberta-v3-large
    - 中国語: hfl/chinese-roberta-wwm-ext-large

    Args:
        language (Languages): ロードする学習済みモデルの対象言語
        pretrained_model_name_or_path (str | None): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)

    Returns:
        PreTrainedTokenizer | PreTrainedTokenizerFast | DebertaV2Tokenizer: ロード済みの BERT トークナイザー
    """

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_tokenizers:
        return __loaded_tokenizers[language]

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        assert DEFAULT_BERT_TOKENIZER_PATHS[language].exists(), \
            f"The default {language} BERT tokenizer does not exist on the file system. Please specify the path to the pre-trained model."
        pretrained_model_name_or_path = str(DEFAULT_BERT_TOKENIZER_PATHS[language])

    # BERT トークナイザーをロードし、辞書に格納して返す
    ## 英語のみ DebertaV2Tokenizer でロードする必要がある
    if language == Languages.EN:
        tokenizer = DebertaV2Tokenizer.from_pretrained(pretrained_model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    __loaded_tokenizers[language] = tokenizer
    logger.info(f"Loaded the {language} BERT tokenizer from {pretrained_model_name_or_path}")

    return tokenizer


def unload_model(language: Languages) -> None:
    """
    指定された言語の BERT モデルをアンロードする

    Args:
        language (Languages): アンロードする BERT モデルの言語
    """

    if language in __loaded_models:
        del __loaded_models[language]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Unloaded the {language} BERT model")


def unload_tokenizer(language: Languages) -> None:
    """
    指定された言語の BERT トークナイザーをアンロードする

    Args:
        language (Languages): アンロードする BERT トークナイザーの言語
    """

    if language in __loaded_tokenizers:
        del __loaded_tokenizers[language]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Unloaded the {language} BERT tokenizer")


def unload_all_models() -> None:
    """
    すべての BERT モデルをアンロードする
    """

    for language in list(__loaded_models.keys()):
        unload_model(language)
    logger.info("Unloaded all BERT models")


def unload_all_tokenizers() -> None:
    """
    すべての BERT トークナイザーをアンロードする
    """

    for language in list(__loaded_tokenizers.keys()):
        unload_tokenizer(language)
    logger.info("Unloaded all BERT tokenizers")
