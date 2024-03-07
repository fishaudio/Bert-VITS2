from style_bert_vits2.constants import Languages
from style_bert_vits2.text_processing.symbols import *


_symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}


def cleaned_text_to_sequence(cleaned_text: str, tones: list[int], language: Languages):
    """
    Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    Args:
      text: string to convert to a sequence

    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = LANGUAGE_TONE_START_MAP[language]
    tones = [i + tone_start for i in tones]
    lang_id = LANGUAGE_ID_MAP[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert(
    text: str,
    word2ph,
    language: Languages,
    device: str,
    assist_text: str | None = None,
    assist_text_weight: float = 0.7,
):
    if language == Languages.ZH:
        from .chinese_bert import get_bert_feature
    elif language == Languages.EN:
        from .english_bert_mock import get_bert_feature
    elif language == Languages.JP:
        from .japanese_bert import get_bert_feature
    else:
        raise ValueError(f"Language {language} not supported")

    return get_bert_feature(text, word2ph, device, assist_text, assist_text_weight)
