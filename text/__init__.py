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
