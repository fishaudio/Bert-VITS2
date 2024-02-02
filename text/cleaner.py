from text import chinese, japanese, english, cleaned_text_to_sequence


language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}


def clean_text(text, language, use_jp_extra=True):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    if language == "JP":
        phones, tones, word2ph = language_module.g2p(norm_text, use_jp_extra)
    else:
        phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    bert = language_module.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
