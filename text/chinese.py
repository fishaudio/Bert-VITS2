import os
import re

import jieba
import cn2an
from pypinyin import lazy_pinyin, Style, BOPOMOFO

from text.symbols import punctuation, cjke_symbols
from text.tone_sandhi import ToneSandhi

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba.posseg as psg


_bopomofo_to_ipa = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("ㄅㄛ", "p⁼wo"),
        ("ㄆㄛ", "pʰwo"),
        ("ㄇㄛ", "mwo"),
        ("ㄈㄛ", "fwo"),
        ("ㄅ", "p⁼"),
        ("ㄆ", "pʰ"),
        ("ㄇ", "m"),
        ("ㄈ", "f"),
        ("ㄉ", "t⁼"),
        ("ㄊ", "tʰ"),
        ("ㄋ", "n"),
        ("ㄌ", "l"),
        ("ㄍ", "k⁼"),
        ("ㄎ", "kʰ"),
        ("ㄏ", "x"),
        ("ㄐ", "tʃ⁼"),
        ("ㄑ", "tʃʰ"),
        ("ㄒ", "ʃ"),
        ("ㄓ", "ts`⁼"),
        ("ㄔ", "ts`ʰ"),
        ("ㄕ", "s`"),
        ("ㄖ", "ɹ`"),
        ("ㄗ", "ts⁼"),
        ("ㄘ", "tsʰ"),
        ("ㄙ", "s"),
        ("ㄚ", "a"),
        ("ㄛ", "o"),
        ("ㄜ", "ə"),
        ("ㄝ", "ɛ"),
        ("ㄞ", "aɪ"),
        ("ㄟ", "eɪ"),
        ("ㄠ", "ɑʊ"),
        ("ㄡ", "oʊ"),
        ("ㄧㄢ", "jɛn"),
        ("ㄩㄢ", "ɥæn"),
        ("ㄢ", "an"),
        ("ㄧㄣ", "in"),
        ("ㄩㄣ", "ɥn"),
        ("ㄣ", "ən"),
        ("ㄤ", "ɑŋ"),
        ("ㄧㄥ", "iŋ"),
        ("ㄨㄥ", "ʊŋ"),
        ("ㄩㄥ", "jʊŋ"),
        ("ㄥ", "əŋ"),
        ("ㄦ", "əɻ"),
        ("ㄧ", "i"),
        ("ㄨ", "u"),
        ("ㄩ", "ɥ"),
        ("，", ","),
        ("。", "."),
        ("！", "!"),
        ("？", "?"),
        ("—", "-"),
    ]
]


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    ":": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "...": "…",
    "···": "…",
    "・・・": "…",
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


tone_modifier = ToneSandhi()


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )

    return replaced_text


def g2p(text):
    text = text_normalize(text)
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    # phones, tones, word2ph = _g2p(sentences)
    phones, tones, word2ph = chinese_to_ipa(text)
    assert sum(word2ph) == len(phones)
    assert (
        len(word2ph) == len(text) + 2
    )  # Sometimes it will crash,you can add a try-catch.
    return phones, tones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _g2p(segments):
    phones_list = []
    tones_list = []
    word2ph = []
    for seg in segments:
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = psg.lcut(seg)
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        for word, pos in seg_cut:
            if pos == "eng":
                continue
            sub_initials, sub_finals = _get_initials_finals(word)
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            initials.append(sub_initials)
            finals.append(sub_finals)

            # assert len(sub_initials) == len(sub_finals) == len(word)
        initials = sum(initials, [])
        finals = sum(finals, [])
        #
        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:
                assert c in punctuation
                phone = [c]
                tone = "0"
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                phone = pinyin_to_symbol_map[pinyin].split(" ")
                word2ph.append(len(phone))

            phones_list += phone
            tones_list += [int(tone)] * len(phone)
    return phones_list, tones_list, word2ph


def text_normalize(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    text = replace_punctuation(text)
    return text


def get_bert_feature(text, word2ph):
    from text import chinese_bert

    return chinese_bert.get_bert_feature(text, word2ph)


def chinese_to_bopomofo(text):
    words = psg.lcut(text)  # jieba.lcut(text, cut_all=False)
    words = tone_modifier.pre_merge_for_modify(words)
    text = []
    initials = []
    finals = []
    for word, pos in words:
        bopomofos = lazy_pinyin(word, BOPOMOFO)
        text += bopomofos
        sub_initials, sub_finals = _get_initials_finals(word)
        sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
        initials.append(sub_initials)
        finals.append(sub_finals)
    initials = sum(initials, [])
    finals = sum(finals, [])
    tones = []
    for c, v in zip(initials, finals):
        if c == v:
            tone = "0"
        else:
            tone = v[-1]
        tones.append(int(tone))
    return text, tones


def bopomofo_to_ipa(text, tones):
    for i in range(len(text)):
        text[i] = re.sub(r"[ˉˊˇˋ˙]", "", text[i])
        for regex, replacement in _bopomofo_to_ipa:
            text[i] = re.sub(regex, replacement, text[i])
    return text, tones


def chinese_to_ipa(text):
    text = text_normalize(text)
    text, tones = chinese_to_bopomofo(text)
    # text = latin_to_bopomofo(text)
    text, ts = bopomofo_to_ipa(text, tones)
    for i in range(len(text)):
        text[i] = re.sub("i([aoe])", r"j\1", text[i])
        text[i] = re.sub("u([aoəe])", r"w\1", text[i])
        text[i] = re.sub("([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)", r"\1ɹ`\2", text[i]).replace(
            "ɻ", "ɹ`"
        )
        text[i] = re.sub("([s][⁼ʰ]?)([→↓↑ ]+|$)", r"\1ɹ\2", text[i])
    word2ph = []
    tones = []
    for i in text:
        if any(j not in cjke_symbols for j in i):
            word2ph += [1] * len(i)
        else:
            word2ph += [len(i)]
        tones += [ts.pop(0)] * len(i)
    word2ph = [1] + word2ph + [1]
    tones = [0] + tones + [0]
    phones = ["_"] + [j for i in text for j in i] + ["_"]
    return phones, tones, word2ph


if __name__ == "__main__":
    from text.chinese_bert import get_bert_feature

    text = "啊！但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)


# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
