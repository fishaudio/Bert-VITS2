from text import chinese, japanese, english, cleaned_text_to_sequence
from tools.sentence import split_by_language
from typing import Tuple, List
from tools.filelist_utils import LangType


language_module_map = {"ZH": chinese, "JP": japanese, "EN": english}


def clean_text_auto(
    text: str
) -> Tuple[
    List[str], List[List[str]], List[List[int]], List[List[int]], List[LangType]
]:
    # we use the same utils in the infer engine to auto split the input into multiple language
    sentences_list = split_by_language(text, target_languages=["zh", "en"])
    norm_texts, phones, tones, word2phs, langs = [], [], [], [], []
    for i, (sentence, lang) in enumerate(sentences_list):
        if sentence == "":
            continue
        norm_text, phone, tone, word2ph = clean_text(sentence, lang.upper())
        norm_texts.append(norm_text)
        phones.append(phone)
        tones.append(tone)
        word2phs.append(word2ph)
        langs.append(lang.upper())
    return norm_texts, phones, tones, word2phs, langs


def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
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
