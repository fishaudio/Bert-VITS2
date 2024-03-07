import torch

from style_bert_vits2.constants import Languages
from style_bert_vits2.text_processing.symbols import (
    LANGUAGE_ID_MAP,
    LANGUAGE_TONE_START_MAP,
    SYMBOLS,
)


_symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}


def cleaned_text_to_sequence(cleaned_text: str, tones: list[int], language: Languages) -> tuple[list[int], list[int], list[int]]:
    """
    Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    Args:
        cleaned_text (str): string to convert to a sequence
        tones (list[int]): List of tones
        language (Languages): Language of the text

    Returns:
        tuple[list[int], list[int], list[int]]: List of integers corresponding to the symbols in the text
    """

    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = LANGUAGE_TONE_START_MAP[language]
    tones = [i + tone_start for i in tones]
    lang_id = LANGUAGE_ID_MAP[language]
    lang_ids = [lang_id for i in phones]

    return phones, tones, lang_ids


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    language: Languages,
    device: torch.device | str,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    テキストから BERT の特徴量を抽出する

    Args:
        text (str): テキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        language (Languages): テキストの言語
        device (torch.device | str): 推論に利用するデバイス
        assist_text (str | None, optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        torch.Tensor: BERT の特徴量
    """

    if language == Languages.JP:
        from style_bert_vits2.text_processing.japanese.bert_feature import extract_bert_feature
    elif language == Languages.EN:
        from style_bert_vits2.text_processing.english.bert_feature import extract_bert_feature
    elif language == Languages.ZH:
        from style_bert_vits2.text_processing.chinese.bert_feature import extract_bert_feature
    else:
        raise ValueError(f"Language {language} not supported")

    return extract_bert_feature(text, word2ph, device, assist_text, assist_text_weight)
