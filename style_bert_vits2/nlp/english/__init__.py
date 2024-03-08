import pickle
import os
import re
from pathlib import Path

import inflect
from g2p_en import G2p

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.symbols import PUNCTUATIONS, SYMBOLS


CMU_DICT_PATH = Path(__file__).parent / "cmudict.rep"
CACHE_PATH = Path(__file__).parent / "cmudict_cache.pickle"


def g2p(text: str) -> tuple[list[str], list[int], list[int]]:

    ARPA = {
        "AH0",
        "S",
        "AH1",
        "EY2",
        "AE2",
        "EH0",
        "OW2",
        "UH0",
        "NG",
        "B",
        "G",
        "AY0",
        "M",
        "AA0",
        "F",
        "AO0",
        "ER2",
        "UH1",
        "IY1",
        "AH2",
        "DH",
        "IY0",
        "EY1",
        "IH0",
        "K",
        "N",
        "W",
        "IY2",
        "T",
        "AA1",
        "ER1",
        "EH2",
        "OY0",
        "UH2",
        "UW1",
        "Z",
        "AW2",
        "AW1",
        "V",
        "UW2",
        "AA2",
        "ER",
        "AW0",
        "UW0",
        "R",
        "OW1",
        "EH1",
        "ZH",
        "AE0",
        "IH2",
        "IH",
        "Y",
        "JH",
        "P",
        "AY1",
        "EY0",
        "OY2",
        "TH",
        "HH",
        "D",
        "ER0",
        "CH",
        "AO1",
        "AE1",
        "AO2",
        "OY1",
        "AY2",
        "IH1",
        "OW0",
        "L",
        "SH",
    }

    _g2p = G2p()

    phones = []
    tones = []
    phone_len = []
    # tokens = [tokenizer.tokenize(i) for i in words]
    words = __text_to_words(text)
    eng_dict = __get_dict()

    for word in words:
        temp_phones, temp_tones = [], []
        if len(word) > 1:
            if "'" in word:
                word = ["".join(word)]
        for w in word:
            if w in PUNCTUATIONS:
                temp_phones.append(w)
                temp_tones.append(0)
                continue
            if w.upper() in eng_dict:
                phns, tns = __refine_syllables(eng_dict[w.upper()])
                temp_phones += [__post_replace_ph(i) for i in phns]
                temp_tones += tns
                # w2ph.append(len(phns))
            else:
                phone_list = list(filter(lambda p: p != " ", _g2p(w)))  # type: ignore
                phns = []
                tns = []
                for ph in phone_list:
                    if ph in ARPA:
                        ph, tn = __refine_ph(ph)
                        phns.append(ph)
                        tns.append(tn)
                    else:
                        phns.append(ph)
                        tns.append(0)
                temp_phones += [__post_replace_ph(i) for i in phns]
                temp_tones += tns
        phones += temp_phones
        tones += temp_tones
        phone_len.append(len(temp_phones))
        # phones = [post_replace_ph(i) for i in phones]

    word2ph = []
    for token, pl in zip(words, phone_len):
        word_len = len(token)

        aaa = __distribute_phone(pl, word_len)
        word2ph += aaa

    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    assert len(phones) == len(tones), text
    assert len(phones) == sum(word2ph), text

    return phones, tones, word2ph


def normalize_text(text: str) -> str:
    text = __normalize_numbers(text)
    text = __replace_punctuation(text)
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
    return text


def __normalize_numbers(text: str) -> str:
    text = re.sub(__comma_number_re, __remove_commas, text)
    text = re.sub(__pounds_re, r"\1 pounds", text)
    text = re.sub(__dollars_re, __expand_dollars, text)
    text = re.sub(__decimal_number_re, __expand_decimal_point, text)
    text = re.sub(__ordinal_re, __expand_ordinal, text)
    text = re.sub(__number_re, __expand_number, text)
    return text


def __replace_punctuation(text: str) -> str:
    REPLACE_MAP = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "．": ".",
        "…": "...",
        "···": "...",
        "・・・": "...",
        "·": ",",
        "・": ",",
        "、": ",",
        "$": ".",
        "“": "'",
        "”": "'",
        '"': "'",
        "‘": "'",
        "’": "'",
        "（": "'",
        "）": "'",
        "(": "'",
        ")": "'",
        "《": "'",
        "》": "'",
        "【": "'",
        "】": "'",
        "[": "'",
        "]": "'",
        "—": "-",
        "−": "-",
        "～": "-",
        "~": "-",
        "「": "'",
        "」": "'",
    }
    pattern = re.compile("|".join(re.escape(p) for p in REPLACE_MAP.keys()))
    replaced_text = pattern.sub(lambda x: REPLACE_MAP[x.group()], text)
    # replaced_text = re.sub(
    #     r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    #     + "".join(punctuation)
    #     + r"]+",
    #     "",
    #     replaced_text,
    # )
    return replaced_text


def __post_replace_ph(ph: str) -> str:
    REPLACE_MAP = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "…": "...",
        "···": "...",
        "・・・": "...",
        "v": "V",
    }
    if ph in REPLACE_MAP.keys():
        ph = REPLACE_MAP[ph]
    if ph in SYMBOLS:
        return ph
    if ph not in SYMBOLS:
        ph = "UNK"
    return ph


def __read_dict() -> dict[str, list[list[str]]]:
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def __cache_dict(g2p_dict: dict[str, list[list[str]]], file_path: Path) -> None:
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def __get_dict() -> dict[str, list[list[str]]]:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = __read_dict()
        __cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict


def __refine_ph(phn: str) -> tuple[str, int]:
    tone = 0
    if re.search(r"\d$", phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    else:
        tone = 3
    return phn.lower(), tone


def __refine_syllables(syllables: list[list[str]]) -> tuple[list[str], list[int]]:
    tones = []
    phonemes = []
    for phn_list in syllables:
        for i in range(len(phn_list)):
            phn = phn_list[i]
            phn, tone = __refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


__inflect = inflect.engine()
__comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
__decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
__pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
__dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
__ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
__number_re = re.compile(r"[0-9]+")


def __expand_dollars(m: re.Match[str]) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def __remove_commas(m: re.Match[str]) -> str:
    return m.group(1).replace(",", "")


def __expand_ordinal(m: re.Match[str]) -> str:
    return __inflect.number_to_words(m.group(0))  # type: ignore


def __expand_number(m: re.Match[str]) -> str:
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + __inflect.number_to_words(num % 100)  # type: ignore
        elif num % 100 == 0:
            return __inflect.number_to_words(num // 100) + " hundred"  # type: ignore
        else:
            return __inflect.number_to_words(
                num, andword="", zero="oh", group=2  # type: ignore
            ).replace(", ", " ")  # type: ignore
    else:
        return __inflect.number_to_words(num, andword="")  # type: ignore


def __expand_decimal_point(m: re.Match[str]) -> str:
    return m.group(1).replace(".", " point ")


def __distribute_phone(n_phone: int, n_word: int) -> list[int]:
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def __text_to_words(text: str) -> list[list[str]]:
    tokenizer = bert_models.load_tokenizer(Languages.EN)
    tokens = tokenizer.tokenize(text)
    words = []
    for idx, t in enumerate(tokens):
        if t.startswith("▁"):
            words.append([t[1:]])
        else:
            if t in PUNCTUATIONS:
                if idx == len(tokens) - 1:
                    words.append([f"{t}"])
                else:
                    if (
                        not tokens[idx + 1].startswith("▁")
                        and tokens[idx + 1] not in PUNCTUATIONS
                    ):
                        if idx == 0:
                            words.append([])
                        words[-1].append(f"{t}")
                    else:
                        words.append([f"{t}"])
            else:
                if idx == 0:
                    words.append([])
                words[-1].append(f"{t}")
    return words


if __name__ == "__main__":
    # print(get_dict())
    # print(eng_word_to_phoneme("hello"))
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))
    # all_phones = set()
    # eng_dict = get_dict()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    #         for ph in group:
    #             all_phones.add(ph)
    # print(all_phones)
