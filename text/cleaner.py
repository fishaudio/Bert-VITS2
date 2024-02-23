def clean_text(text, language, use_jp_extra=True, ignore_unknown=False):
    # Changed to import inside if condition to avoid unnecessary import
    if language == "ZH":
        from . import chinese as language_module

        norm_text = language_module.text_normalize(text)
        phones, tones, word2ph = language_module.g2p(norm_text)
    elif language == "EN":
        from . import english as language_module

        norm_text = language_module.text_normalize(text)
        phones, tones, word2ph = language_module.g2p(norm_text)
    elif language == "JP":
        from . import japanese as language_module

        norm_text = language_module.text_normalize(text)
        phones, tones, word2ph = language_module.g2p(
            norm_text, use_jp_extra, ignore_unknown=ignore_unknown
        )
    else:
        raise ValueError(f"Language {language} not supported")
    return norm_text, phones, tones, word2ph


if __name__ == "__main__":
    pass
