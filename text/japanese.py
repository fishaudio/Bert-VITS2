# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re
import unicodedata

from transformers import AutoTokenizer

from text import punctuation, symbols

from num2words import num2words

import pyopenjtalk
import jaconv


# Mapping of hiragana to phonetic representation
hiragana_map = {
    "う゛ぁ": " v a",
    "う゛ぃ": " v i",
    "う゛ぇ": " v e",
    "う゛ぉ": " v o",
    "う゛ゅ": " by u",
    "ぅ゛": " v u",
    # ゔ等の処理を追加
    "ゔぁ": " v a",
    "ゔぃ": " v i",
    "ゔぇ": " v e",
    "ゔぉ": " v o",
    "ゔゅ": " by u",
    # 2文字からなる変換規則
    "あぁ": " a a",
    "いぃ": " i i",
    "いぇ": " i e",
    "いゃ": " y a",
    "うぅ": " u:",
    "えぇ": " e e",
    "おぉ": " o:",
    "かぁ": " k a:",
    "きぃ": " k i:",
    "くぅ": " k u:",
    "くゃ": " ky a",
    "くゅ": " ky u",
    "くょ": " ky o",
    "けぇ": " k e:",
    "こぉ": " k o:",
    "がぁ": " g a:",
    "ぎぃ": " g i:",
    "ぐぅ": " g u:",
    "ぐゃ": " gy a",
    "ぐゅ": " gy u",
    "ぐょ": " gy o",
    "げぇ": " g e:",
    "ごぉ": " g o:",
    "さぁ": " s a:",
    "しぃ": " sh i",
    "すぅ": " s u:",
    "すゃ": " sh a",
    "すゅ": " sh u",
    "すょ": " sh o",
    "せぇ": " s e:",
    "そぉ": " s o:",
    "ざぁ": " z a:",
    "じぃ": " j i:",
    "ずぅ": " z u:",
    "ずゃ": " zy a",
    "ずゅ": " zy u",
    "ずょ": " zy o",
    "ぜぇ": " z e:",
    "ぞぉ": " z o:",
    "たぁ": " t a:",
    "ちぃ": " ch i",
    "つぁ": " ts a",
    "つぃ": " ts i",
    "つぅ": " ts u",
    "つゃ": " ch a",
    "つゅ": " ch u",
    "つょ": " ch o",
    "つぇ": " ts e",
    "つぉ": " ts o",
    "てぇ": " t e:",
    "とぉ": " t o:",
    "だぁ": " d a:",
    "ぢぃ": " j i:",
    "づぅ": " d u:",
    "づゃ": " zy a",
    "づゅ": " zy u",
    "づょ": " zy o",
    "でぇ": " d e:",
    "なぁ": " n a:",
    "にぃ": " n i:",
    "ぬぅ": " n u:",
    "ぬゃ": " ny a",
    "ぬゅ": " ny u",
    "ぬょ": " ny o",
    "ねぇ": " n e:",
    "のぉ": " n o:",
    "はぁ": " h a:",
    "ひぃ": " h i:",
    "ふぅ": " f u:",
    "ふゃ": " hy a",
    "へぇ": " h e:",
    "ほぉ": " h o:",
    "ばぁ": " b a:",
    "びぃ": " b i:",
    "ぶぅ": " b u:",
    "ぶゅ": " by u",
    "べぇ": " b e:",
    "ぼぉ": " b o:",
    "ぱぁ": " p a:",
    "ぴぃ": " p i:",
    "ぷぅ": " p u:",
    "ぷゃ": " py a",
    "ぷゅ": " py u",
    "ぷょ": " py o",
    "ぺぇ": " p e:",
    "ぽぉ": " p o:",
    "まぁ": " m a:",
    "みぃ": " m i:",
    "むぅ": " m u:",
    "むゃ": " my a",
    "むゅ": " my u",
    "むょ": " my o",
    "めぇ": " m e:",
    "もぉ": " m o:",
    "やぁ": " y a:",
    "ゆぅ": " y u:",
    "ゆゃ": " y a:",
    "ゆゅ": " y u:",
    "ゆょ": " y o:",
    "よぉ": " y o:",
    "らぁ": " r a:",
    "りぃ": " r i:",
    "るぅ": " r u:",
    "るゃ": " ry a",
    "るゅ": " ry u",
    "るょ": " ry o",
    "れぇ": " r e:",
    "ろぉ": " r o:",
    "わぁ": " w a:",
    "をぉ": " o:",
    "う゛": " b u",
    "でぃ": " d i",
    "でゃ": " dy a",
    "でゅ": " dy u",
    "でょ": " dy o",
    "てぃ": " t i",
    "てゃ": " ty a",
    "てゅ": " ty u",
    "てょ": " ty o",
    "すぃ": " s i",
    "ずぁ": " z u",
    "ずぃ": " z i",
    "ずぇ": " z e",
    "ずぉ": " z o",
    "きゃ": " ky a",
    "きゅ": " ky u",
    "きょ": " ky o",
    "しゃ": " sh a",
    "しゅ": " sh u",
    "しぇ": " sh e",
    "しょ": " sh o",
    "ちゃ": " ch a",
    "ちゅ": " ch u",
    "ちぇ": " ch e",
    "ちょ": " ch o",
    "とぅ": " t u",
    "とゃ": " ty a",
    "とゅ": " ty u",
    "とょ": " ty o",
    "どぁ": " d o ",
    "どぅ": " d u",
    "どゃ": " dy a",
    "どゅ": " dy u",
    "どょ": " dy o",
    "どぉ": " d o:",
    "にゃ": " ny a",
    "にゅ": " ny u",
    "にょ": " ny o",
    "ひゃ": " hy a",
    "ひゅ": " hy u",
    "ひょ": " hy o",
    "みゃ": " my a",
    "みゅ": " my u",
    "みょ": " my o",
    "りゃ": " ry a",
    "りゅ": " ry u",
    "りょ": " ry o",
    "ぎゃ": " gy a",
    "ぎゅ": " gy u",
    "ぎょ": " gy o",
    "ぢぇ": " j e",
    "ぢゃ": " j a",
    "ぢゅ": " j u",
    "ぢょ": " j o",
    "じぇ": " j e",
    "じゃ": " j a",
    "じゅ": " j u",
    "じょ": " j o",
    "びゃ": " by a",
    "びゅ": " by u",
    "びょ": " by o",
    "ぴゃ": " py a",
    "ぴゅ": " py u",
    "ぴょ": " py o",
    "うぁ": " u a",
    "うぃ": " w i",
    "うぇ": " w e",
    "うぉ": " w o",
    "ふぁ": " f a",
    "ふぃ": " f i",
    "ふゅ": " hy u",
    "ふょ": " hy o",
    "ふぇ": " f e",
    "ふぉ": " f o",
    # 1音からなる変換規則
    "あ": " a",
    "い": " i",
    "う": " u",
    "ゔ": " v u",  # ゔの処理を追加
    "え": " e",
    "お": " o",
    "か": " k a",
    "き": " k i",
    "く": " k u",
    "け": " k e",
    "こ": " k o",
    "さ": " s a",
    "し": " sh i",
    "す": " s u",
    "せ": " s e",
    "そ": " s o",
    "た": " t a",
    "ち": " ch i",
    "つ": " ts u",
    "て": " t e",
    "と": " t o",
    "な": " n a",
    "に": " n i",
    "ぬ": " n u",
    "ね": " n e",
    "の": " n o",
    "は": " h a",
    "ひ": " h i",
    "ふ": " f u",
    "へ": " h e",
    "ほ": " h o",
    "ま": " m a",
    "み": " m i",
    "む": " m u",
    "め": " m e",
    "も": " m o",
    "ら": " r a",
    "り": " r i",
    "る": " r u",
    "れ": " r e",
    "ろ": " r o",
    "が": " g a",
    "ぎ": " g i",
    "ぐ": " g u",
    "げ": " g e",
    "ご": " g o",
    "ざ": " z a",
    "じ": " j i",
    "ず": " z u",
    "ぜ": " z e",
    "ぞ": " z o",
    "だ": " d a",
    "ぢ": " j i",
    "づ": " z u",
    "で": " d e",
    "ど": " d o",
    "ば": " b a",
    "び": " b i",
    "ぶ": " b u",
    "べ": " b e",
    "ぼ": " b o",
    "ぱ": " p a",
    "ぴ": " p i",
    "ぷ": " p u",
    "ぺ": " p e",
    "ぽ": " p o",
    "や": " y a",
    "ゆ": " y u",
    "よ": " y o",
    "わ": " w a",
    "ゐ": " i",
    "ゑ": " e",
    "ん": " N",
    "っ": " q",
    # ここまでに処理されてない ぁぃぅぇぉ はそのまま大文字扱い
    "ぁ": " a",
    "ぃ": " i",
    "ぅ": " u",
    "ぇ": " e",
    "ぉ": " o",
    "ゎ": " w a",
    # 長音の処理
    # for (pattern, replace_str) in JULIUS_LONG_VOWEL:
    #     text = pattern.sub(replace_str, text)
    # text = text.replace("o u", "o:")  # おう -> おーの音便
    "ー": ":",
    "〜": ":",
    "−": ":",
    "-": ":",
    # その他特別な処理
    "を": " o",
    # ここまでに処理されていないゅ等もそのまま大文字扱い（追加）
    "ゃ": " y a",
    "ゅ": " y u",
    "ょ": " y o",
}


def hiragana2p(txt: str) -> str:
    """
    Modification of `jaconv.hiragana2julius`.
    - avoid using `:`, instead, `あーーー` -> `a a a a`.
    - avoid converting `o u` to `o o` (because the input is already actual `yomi`).
    - avoid using `N` for `ん` (for compatibility)
    - use `v` for `ゔ` related text.
    - add bare `ゃ` `ゅ` `ょ` to `y a` `y u` `y o` (for compatibility).
    """

    result = []
    skip = 0
    for i in range(len(txt)):
        if skip:
            skip -= 1
            continue

        for length in range(3, 0, -1):
            if txt[i : i + length] in hiragana_map:
                result.append(hiragana_map[txt[i : i + length]])
                skip = length - 1
                break

    txt = "".join(result)
    txt = txt.strip()
    txt = txt.replace(":+", ":")

    # ここまで`jaconv.hiragana2julius`と音便処理と長音処理をのぞいて同じ
    # ここから`k a:: k i:`→`k a a a k i i`のように`:`の数だけ繰り返す処理
    pattern = r"(\w)(:*)"
    replacement = lambda m: m.group(1) + (" " + m.group(1)) * len(m.group(2))

    txt = re.sub(pattern, replacement, txt)
    txt = txt.replace("N", "n")  # 促音のNをnに変換
    return txt


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
        res += hiragana2p(jaconv.kata2hira(text)).split(" ")
        break
    # res = _COLON_RX.sub(":", res)
    return res


_SYMBOL_TOKENS = set(list("・、。？！"))
_NO_YOMI_TOKENS = set(list("「」『』―（）［］[]"))
_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)


def text2sep_kata(text: str):
    parsed = pyopenjtalk.run_frontend(text)
    res = []
    sep = []
    for parts in parsed:
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )
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
    return sep, res, get_accent(parsed)


def get_accent(parsed):
    labels = pyopenjtalk.make_label(parsed)

    phonemes = []
    accents = []
    for n, label in enumerate(labels):
        phoneme = re.search(r"\-([^\+]*)\+", label).group(1)
        if phoneme not in ["sil", "pau"]:
            phonemes.append(phoneme.replace("cl", "q").lower())
        else:
            continue
        a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))
        a2 = int(re.search(r"\+(\d+)\+", label).group(1))
        if re.search(r"\-([^\+]*)\+", labels[n + 1]).group(1) in ["sil", "pau"]:
            a2_next = -1
        else:
            a2_next = int(re.search(r"\+(\d+)\+", labels[n + 1]).group(1))
        # Falling
        if a1 == 0 and a2_next == a2 + 1:
            accents.append(-1)
        # Rising
        elif a2 == 1 and a2_next == 2:
            accents.append(1)
        else:
            accents.append(0)
    return list(zip(phonemes, accents))


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


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
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
    res = res.replace("゙", "")
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


tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese-char-wwm")


def align_tones(phones, tones):
    res = []
    for pho in phones:
        temp = [0] * len(pho)
        for idx, p in enumerate(pho):
            if len(tones) == 0:
                break
            if p == tones[0][0]:
                temp[idx] = tones[0][1]
                if idx > 0:
                    temp[idx] += temp[idx - 1]
                tones.pop(0)
        temp = [0] + temp
        temp = temp[:-1]
        if -1 in temp:
            temp = [i + 1 for i in temp]
        res.append(temp)
    res = [i for j in res for i in j]
    assert not any([i < 0 for i in res]) and not any([i > 1 for i in res])
    return res


def rearrange_tones(tones, phones):
    res = [0] * len(tones)
    for i in range(len(tones)):
        if i == 0:
            if tones[i] not in punctuation:
                res[i] = 1
        elif tones[i] == prev:
            if phones[i] in punctuation:
                res[i] = 0
            else:
                res[i] = 1
        elif tones[i] > prev:
            res[i] = 2
        elif tones[i] < prev:
            res[i - 1] = 3
            res[i] = 1
        prev = tones[i]
    return res


def g2p(norm_text):
    sep_text, sep_kata, acc = text2sep_kata(norm_text)
    sep_tokenized = []
    for i in sep_text:
        if i not in punctuation:
            sep_tokenized.append(tokenizer.tokenize(i))
        else:
            sep_tokenized.append([i])

    sep_phonemes = handle_long([kata2phoneme(i) for i in sep_kata])
    # 异常处理，MeCab不认识的词的话会一路传到这里来，然后炸掉。目前来看只有那些超级稀有的生僻词会出现这种情况
    for i in sep_phonemes:
        for j in i:
            assert j in symbols, (sep_text, sep_kata, sep_phonemes)
    tones = align_tones(sep_phonemes, acc)

    word2ph = []
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        phone_len = len(phoneme)
        word_len = len(token)

        aaa = distribute_phone(phone_len, word_len)
        word2ph += aaa
    phones = ["_"] + [j for i in sep_phonemes for j in i] + ["_"]
    # tones = [0] + rearrange_tones(tones, phones[1:-1]) + [0]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    assert len(phones) == len(tones)
    return phones, tones, word2ph


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")
    text = "hello,こんにちは、世界ー！……"
    from text.japanese_bert import get_bert_feature

    text = text_normalize(text)
    print(text)

    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)
