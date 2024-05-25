import re

from g2p_en import G2p

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.english.cmudict import get_dict
from style_bert_vits2.nlp.symbols import PUNCTUATIONS, SYMBOLS


# Initialize global variables once
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
eng_dict = get_dict()


def g2p(text: str) -> tuple[list[str], list[int], list[int]]:
    phones = []
    tones = []
    phone_len = []
    words = __text_to_words(text)

    for word in words:
        temp_phones, temp_tones = [], []
        if len(word) > 1 and "'" in word:
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
            else:
                phone_list = list(filter(lambda p: p != " ", _g2p(w)))
                phns, tns = [], []
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

    word2ph = []
    for token, pl in zip(words, phone_len):
        word_len = len(token)
        word2ph += __distribute_phone(pl, word_len)

    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    assert len(phones) == len(tones), text
    assert len(phones) == sum(word2ph), text

    return phones, tones, word2ph


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
    if ph in REPLACE_MAP:
        ph = REPLACE_MAP[ph]
    if ph in SYMBOLS:
        return ph
    return "UNK"


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
        for phn in phn_list:
            phn, tone = __refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


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
        elif t in PUNCTUATIONS:
            if idx == len(tokens) - 1:
                words.append([f"{t}"])
            elif (
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
