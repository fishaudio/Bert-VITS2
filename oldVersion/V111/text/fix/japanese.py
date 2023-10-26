# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re
import unicodedata

from transformers import AutoTokenizer

from .. import punctuation, symbols

from num2words import num2words

import pyopenjtalk
import jaconv


def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    text = text.strip()
    if text == "ー":
        return ["ー"]
    elif text.startswith("ー"):
        return ["ー"] + kata2phoneme(text[1:])
    res = []
    prev = None
    while text:
        if re.match(_MARKS, text):
            res.append(text)
            text = text[1:]
            continue
        if text.startswith("ー"):
            if prev:
                res.append(prev[-1])
            text = text[1:]
            continue
        res += pyopenjtalk.g2p(text).lower().replace("cl", "q").split(" ")
        break
    # res = _COLON_RX.sub(":", res)
    return res


def hira2kata(text: str) -> str:
    return jaconv.hira2kata(text)


_SYMBOL_TOKENS = set(list("・、。？！"))
_NO_YOMI_TOKENS = set(list("「」『』―（）［］[]"))
_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)


def text2kata(text: str) -> str:
    parsed = pyopenjtalk.run_frontend(text)

    res = []
    for parts in parsed:
        word, yomi = replace_punctuation(parts["orig"]), parts["pron"].replace("’", "")
        if yomi:
            if re.match(_MARKS, yomi):
                if len(word) > 1:
                    word = [replace_punctuation(i) for i in list(word)]
                    yomi = word
                    res += yomi
                    sep += word
                    continue
                elif word not in rep_map.keys() and word not in rep_map.values():
                    word = ","
                yomi = word
            res.append(yomi)
        else:
            if word in _SYMBOL_TOKENS:
                res.append(word)
            elif word in ("っ", "ッ"):
                res.append("ッ")
            elif word in _NO_YOMI_TOKENS:
                pass
            else:
                res.append(word)
    return hira2kata("".join(res))


def text2sep_kata(text: str) -> (list, list):
    parsed = pyopenjtalk.run_frontend(text)

    res = []
    sep = []
    for parts in parsed:
        word, yomi = replace_punctuation(parts["orig"]), parts["pron"].replace("’", "")
        if yomi:
            if re.match(_MARKS, yomi):
                if len(word) > 1:
                    word = [replace_punctuation(i) for i in list(word)]
                    yomi = word
                    res += yomi
                    sep += word
                    continue
                elif word not in rep_map.keys() and word not in rep_map.values():
                    word = ","
                yomi = word
            res.append(yomi)
        else:
            if word in _SYMBOL_TOKENS:
                res.append(word)
            elif word in ("っ", "ッ"):
                res.append("ッ")
            elif word in _NO_YOMI_TOKENS:
                pass
            else:
                res.append(word)
        sep.append(word)
    return sep, [hira2kata(i) for i in res]


_ALPHASYMBOL_YOMI = {
    "#": "シャープ",
    "%": "パーセント",
    "&": "アンド",
    "+": "プラス",
    "-": "マイナス",
    ":": "コロン",
    ";": "セミコロン",
    "<": "小なり",
    "=": "イコール",
    ">": "大なり",
    "@": "アット",
    "a": "エー",
    "b": "ビー",
    "c": "シー",
    "d": "ディー",
    "e": "イー",
    "f": "エフ",
    "g": "ジー",
    "h": "エイチ",
    "i": "アイ",
    "j": "ジェー",
    "k": "ケー",
    "l": "エル",
    "m": "エム",
    "n": "エヌ",
    "o": "オー",
    "p": "ピー",
    "q": "キュー",
    "r": "アール",
    "s": "エス",
    "t": "ティー",
    "u": "ユー",
    "v": "ブイ",
    "w": "ダブリュー",
    "x": "エックス",
    "y": "ワイ",
    "z": "ゼット",
    "α": "アルファ",
    "β": "ベータ",
    "γ": "ガンマ",
    "δ": "デルタ",
    "ε": "イプシロン",
    "ζ": "ゼータ",
    "η": "イータ",
    "θ": "シータ",
    "ι": "イオタ",
    "κ": "カッパ",
    "λ": "ラムダ",
    "μ": "ミュー",
    "ν": "ニュー",
    "ξ": "クサイ",
    "ο": "オミクロン",
    "π": "パイ",
    "ρ": "ロー",
    "σ": "シグマ",
    "τ": "タウ",
    "υ": "ウプシロン",
    "φ": "ファイ",
    "χ": "カイ",
    "ψ": "プサイ",
    "ω": "オメガ",
}


_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")


def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res


def japanese_convert_alpha_symbols_to_words(text: str) -> str:
    return "".join([_ALPHASYMBOL_YOMI.get(ch, ch) for ch in text.lower()])


def japanese_text_to_phonemes(text: str) -> str:
    """Convert Japanese text to phonemes."""
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = japanese_convert_alpha_symbols_to_words(res)
    res = text2kata(res)
    res = kata2phoneme(res)
    return res


def is_japanese_character(char):
    # 定义日语文字系统的 Unicode 范围
    japanese_ranges = [
        (0x3040, 0x309F),  # 平假名
        (0x30A0, 0x30FF),  # 片假名
        (0x4E00, 0x9FFF),  # 汉字 (CJK Unified Ideographs)
        (0x3400, 0x4DBF),  # 汉字扩展 A
        (0x20000, 0x2A6DF),  # 汉字扩展 B
        # 可以根据需要添加其他汉字扩展范围
    ]

    # 将字符的 Unicode 编码转换为整数
    char_code = ord(char)

    # 检查字符是否在任何一个日语范围内
    for start, end in japanese_ranges:
        if start <= char_code <= end:
            return True

    return False


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
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


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF"
        + "".join(punctuation)
        + r"]+",
        "",
        replaced_text,
    )

    return replaced_text


def text_normalize(text):
    res = unicodedata.normalize("NFKC", text)
    res = japanese_convert_numbers_to_words(res)
    # res = "".join([i for i in res if is_japanese_character(i)])
    res = replace_punctuation(res)
    return res


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def handle_long(sep_phonemes):
    for i in range(len(sep_phonemes)):
        if sep_phonemes[i][0] == "ー":
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]
    return sep_phonemes


tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")


def g2p(norm_text):
    sep_text, sep_kata = text2sep_kata(norm_text)
    sep_tokenized = [tokenizer.tokenize(i) for i in sep_text]
    sep_phonemes = handle_long([kata2phoneme(i) for i in sep_kata])
    # 异常处理，MeCab不认识的词的话会一路传到这里来，然后炸掉。目前来看只有那些超级稀有的生僻词会出现这种情况
    for i in sep_phonemes:
        for j in i:
            assert j in symbols, (sep_text, sep_kata, sep_phonemes)

    word2ph = []
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        phone_len = len(phoneme)
        word_len = len(token)

        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
    phones = ["_"] + [j for i in sep_phonemes for j in i] + ["_"]
    tones = [0 for i in phones]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")
    text = "hello,こんにちは、世界ー！……"
    from text.japanese_bert import get_bert_feature

    text = text_normalize(text)
    print(text)

    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)
