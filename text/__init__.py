from text.symbols import language_tone_start_map, language_id_map, symbols


def cleaned_text_to_sequence(phones, tones, languages):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """  # noqa: E501

    assert len(phones) == len(tones) == len(languages)

    new_phones = []
    for i, lang in zip(phones, languages):
        if f"{lang}_{i}" in symbols:
            new_phones.append(symbols.index(f"{lang}_{i}"))
        else:
            # Maybe it's a punctuation mark
            new_phones.append(symbols.index(i))

    tones = [i + language_tone_start_map[lang] for i, lang in zip(tones, languages)]
    lang_ids = [language_id_map[i] for i in languages]

    return new_phones, tones, lang_ids
