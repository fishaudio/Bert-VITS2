import re
from pathlib import Path

import jieba.posseg as psg
from pypinyin import Style, lazy_pinyin

from style_bert_vits2.nlp.chinese.tone_sandhi import ToneSandhi
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


with open(Path(__file__).parent / "opencpop-strict.txt", encoding="utf-8") as f:
    __PINYIN_TO_SYMBOL_MAP = {
        line.split("\t")[0]: line.strip().split("\t")[1] for line in f.readlines()
    }


def g2p(text: str) -> tuple[list[str], list[int], list[int]]:
    pattern = r"(?<=[{0}])\s*".format("".join(PUNCTUATIONS))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, tones, word2ph = __g2p(sentences)
    assert sum(word2ph) == len(phones)
    assert len(word2ph) == len(text)  # Sometimes it will crash,you can add a try-catch.
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def __g2p(segments: list[str]) -> tuple[list[str], list[int], list[int]]:
    phones_list = []
    tones_list = []
    word2ph = []
    tone_modifier = ToneSandhi()
    for seg in segments:
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = psg.lcut(seg)
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)  # type: ignore
        for word, pos in seg_cut:
            if pos == "eng":
                continue
            sub_initials, sub_finals = __get_initials_finals(word)
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
                assert c in PUNCTUATIONS
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
                    if v_without_tone in v_rep_map:
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map:
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map:
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in __PINYIN_TO_SYMBOL_MAP, (
                    pinyin,
                    seg,
                    raw_pinyin,
                )
                phone = __PINYIN_TO_SYMBOL_MAP[pinyin].split(" ")
                word2ph.append(len(phone))

            phones_list += phone
            tones_list += [int(tone)] * len(phone)
    return phones_list, tones_list, word2ph


def __get_initials_finals(word: str) -> tuple[list[str], list[str]]:
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


if __name__ == "__main__":
    from style_bert_vits2.nlp.chinese.bert_feature import extract_bert_feature
    from style_bert_vits2.nlp.chinese.normalizer import normalize_text

    text = "啊！但是《原神》是由,米哈游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = normalize_text(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = extract_bert_feature(text, word2ph, "cuda")

    print(phones, tones, word2ph, bert.shape)


# 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
