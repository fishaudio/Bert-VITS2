from text.symbols import *


_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(phones, tones, languages):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """

    assert len(phones) == len(tones) == len(languages)

    phones = [_symbol_to_id[symbol] for symbol in phones]
    tones = [i + language_tone_start_map[lang] for i, lang in zip(tones, languages)]
    lang_ids = [language_id_map[i] for i in languages]

    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language, device):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert_mock import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert

    lang_bert_func_map = {"ZH": zh_bert, "EN": en_bert, "JP": jp_bert}
    bert = lang_bert_func_map[language](norm_text, word2ph, device)
    return bert
