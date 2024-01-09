"""
文本转拼音
"""
import commons
from text import cleaned_text_to_sequence
from text.cleaner import clean_text


def gen_phones(text, language_str, add_blank, style_text=None, style_weight=0.7):
    style_text = None if style_text == "" else style_text
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
    phone = commons.intersperse(phone, 0)
    tone = commons.intersperse(tone, 0)
    language = commons.intersperse(language, 0)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    result = "{}|{}|{}|{}".format(norm_text, phone, tone, word2ph)
    return result
