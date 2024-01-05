from text import chinese, cleaned_text_to_sequence


def clean_text(text, language):
    norm_text = chinese.text_normalize(text)
    phones, tones, word2ph = chinese.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language):
    norm_text = chinese.text_normalize(text)
    phones, tones, word2ph = chinese.g2p(norm_text)
    bert = chinese.get_bert_feature(norm_text, word2ph)
    return phones, tones, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
