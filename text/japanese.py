# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re
import unicodedata

from transformers import AutoTokenizer

from text import punctuation, symbols

from num2words import num2words
from common.log import logger

import pyopenjtalk
import jaconv


def hiragana2p(text: str) -> str:
    """
    Modification of `jaconv.hiragana2julius`.
    - avoid using `:`, instead, `あーーー` -> `a a a a`.
    - avoid converting `o u` to `o o` (because the input is already actual `yomi`).
    - avoid using `N` for `ん` (for compatibility)
    - use `v` for `ゔ` related text.
    - add bare `ゃ` `ゅ` `ょ` to `y a` `y u` `y o` (for compatibility).
    """
    # 3文字以上からなる変換規則
    text = text.replace("う゛ぁ", " v a")
    text = text.replace("う゛ぃ", " v i")
    text = text.replace("う゛ぇ", " v e")
    text = text.replace("う゛ぉ", " v o")
    text = text.replace("う゛ゅ", " by u")

    # ゔ等の処理を追加
    text = text.replace("ゔぁ", " v a")
    text = text.replace("ゔぃ", " v i")
    text = text.replace("ゔぇ", " v e")
    text = text.replace("ゔぉ", " v o")
    text = text.replace("ゔゅ", " by u")

    # 2文字からなる変換規則
    text = text.replace("ぅ゛", " v u")

    text = text.replace("あぁ", " a a")
    text = text.replace("いぃ", " i i")
    text = text.replace("いぇ", " i e")
    text = text.replace("いゃ", " y a")
    text = text.replace("うぅ", " u:")
    text = text.replace("えぇ", " e e")
    text = text.replace("おぉ", " o:")
    text = text.replace("かぁ", " k a:")
    text = text.replace("きぃ", " k i:")
    text = text.replace("くぅ", " k u:")
    text = text.replace("くゃ", " ky a")
    text = text.replace("くゅ", " ky u")
    text = text.replace("くょ", " ky o")
    text = text.replace("けぇ", " k e:")
    text = text.replace("こぉ", " k o:")
    text = text.replace("がぁ", " g a:")
    text = text.replace("ぎぃ", " g i:")
    text = text.replace("ぐぅ", " g u:")
    text = text.replace("ぐゃ", " gy a")
    text = text.replace("ぐゅ", " gy u")
    text = text.replace("ぐょ", " gy o")
    text = text.replace("げぇ", " g e:")
    text = text.replace("ごぉ", " g o:")
    text = text.replace("さぁ", " s a:")
    text = text.replace("しぃ", " sh i:")
    text = text.replace("すぅ", " s u:")
    text = text.replace("すゃ", " sh a")
    text = text.replace("すゅ", " sh u")
    text = text.replace("すょ", " sh o")
    text = text.replace("せぇ", " s e:")
    text = text.replace("そぉ", " s o:")
    text = text.replace("ざぁ", " z a:")
    text = text.replace("じぃ", " j i:")
    text = text.replace("ずぅ", " z u:")
    text = text.replace("ずゃ", " zy a")
    text = text.replace("ずゅ", " zy u")
    text = text.replace("ずょ", " zy o")
    text = text.replace("ぜぇ", " z e:")
    text = text.replace("ぞぉ", " z o:")
    text = text.replace("たぁ", " t a:")
    text = text.replace("ちぃ", " ch i:")
    text = text.replace("つぁ", " ts a")
    text = text.replace("つぃ", " ts i")
    text = text.replace("つぅ", " ts u:")
    text = text.replace("つゃ", " ch a")
    text = text.replace("つゅ", " ch u")
    text = text.replace("つょ", " ch o")
    text = text.replace("つぇ", " ts e")
    text = text.replace("つぉ", " ts o")
    text = text.replace("てぇ", " t e:")
    text = text.replace("とぉ", " t o:")
    text = text.replace("だぁ", " d a:")
    text = text.replace("ぢぃ", " j i:")
    text = text.replace("づぅ", " d u:")
    text = text.replace("づゃ", " zy a")
    text = text.replace("づゅ", " zy u")
    text = text.replace("づょ", " zy o")
    text = text.replace("でぇ", " d e:")
    text = text.replace("どぉ", " d o:")
    text = text.replace("なぁ", " n a:")
    text = text.replace("にぃ", " n i:")
    text = text.replace("ぬぅ", " n u:")
    text = text.replace("ぬゃ", " ny a")
    text = text.replace("ぬゅ", " ny u")
    text = text.replace("ぬょ", " ny o")
    text = text.replace("ねぇ", " n e:")
    text = text.replace("のぉ", " n o:")
    text = text.replace("はぁ", " h a:")
    text = text.replace("ひぃ", " h i:")
    text = text.replace("ふぅ", " f u:")
    text = text.replace("ふゃ", " hy a")
    text = text.replace("ふゅ", " hy u")
    text = text.replace("ふょ", " hy o")
    text = text.replace("へぇ", " h e:")
    text = text.replace("ほぉ", " h o:")
    text = text.replace("ばぁ", " b a:")
    text = text.replace("びぃ", " b i:")
    text = text.replace("ぶぅ", " b u:")
    text = text.replace("ふゃ", " hy a")
    text = text.replace("ぶゅ", " by u")
    text = text.replace("ふょ", " hy o")
    text = text.replace("べぇ", " b e:")
    text = text.replace("ぼぉ", " b o:")
    text = text.replace("ぱぁ", " p a:")
    text = text.replace("ぴぃ", " p i:")
    text = text.replace("ぷぅ", " p u:")
    text = text.replace("ぷゃ", " py a")
    text = text.replace("ぷゅ", " py u")
    text = text.replace("ぷょ", " py o")
    text = text.replace("ぺぇ", " p e:")
    text = text.replace("ぽぉ", " p o:")
    text = text.replace("まぁ", " m a:")
    text = text.replace("みぃ", " m i:")
    text = text.replace("むぅ", " m u:")
    text = text.replace("むゃ", " my a")
    text = text.replace("むゅ", " my u")
    text = text.replace("むょ", " my o")
    text = text.replace("めぇ", " m e:")
    text = text.replace("もぉ", " m o:")
    text = text.replace("やぁ", " y a:")
    text = text.replace("ゆぅ", " y u:")
    text = text.replace("ゆゃ", " y a:")
    text = text.replace("ゆゅ", " y u:")
    text = text.replace("ゆょ", " y o:")
    text = text.replace("よぉ", " y o:")
    text = text.replace("らぁ", " r a:")
    text = text.replace("りぃ", " r i:")
    text = text.replace("るぅ", " r u:")
    text = text.replace("るゃ", " ry a")
    text = text.replace("るゅ", " ry u")
    text = text.replace("るょ", " ry o")
    text = text.replace("れぇ", " r e:")
    text = text.replace("ろぉ", " r o:")
    text = text.replace("わぁ", " w a:")
    text = text.replace("をぉ", " o:")

    text = text.replace("う゛", " b u")
    text = text.replace("でぃ", " d i")
    text = text.replace("でぇ", " d e:")
    text = text.replace("でゃ", " dy a")
    text = text.replace("でゅ", " dy u")
    text = text.replace("でょ", " dy o")
    text = text.replace("てぃ", " t i")
    text = text.replace("てぇ", " t e:")
    text = text.replace("てゃ", " ty a")
    text = text.replace("てゅ", " ty u")
    text = text.replace("てょ", " ty o")
    text = text.replace("すぃ", " s i")
    text = text.replace("ずぁ", " z u a")
    text = text.replace("ずぃ", " z i")
    text = text.replace("ずぅ", " z u")
    text = text.replace("ずゃ", " zy a")
    text = text.replace("ずゅ", " zy u")
    text = text.replace("ずょ", " zy o")
    text = text.replace("ずぇ", " z e")
    text = text.replace("ずぉ", " z o")
    text = text.replace("きゃ", " ky a")
    text = text.replace("きゅ", " ky u")
    text = text.replace("きょ", " ky o")
    text = text.replace("しゃ", " sh a")
    text = text.replace("しゅ", " sh u")
    text = text.replace("しぇ", " sh e")
    text = text.replace("しょ", " sh o")
    text = text.replace("ちゃ", " ch a")
    text = text.replace("ちゅ", " ch u")
    text = text.replace("ちぇ", " ch e")
    text = text.replace("ちょ", " ch o")
    text = text.replace("とぅ", " t u")
    text = text.replace("とゃ", " ty a")
    text = text.replace("とゅ", " ty u")
    text = text.replace("とょ", " ty o")
    text = text.replace("どぁ", " d o a")
    text = text.replace("どぅ", " d u")
    text = text.replace("どゃ", " dy a")
    text = text.replace("どゅ", " dy u")
    text = text.replace("どょ", " dy o")
    text = text.replace("どぉ", " d o:")
    text = text.replace("にゃ", " ny a")
    text = text.replace("にゅ", " ny u")
    text = text.replace("にょ", " ny o")
    text = text.replace("ひゃ", " hy a")
    text = text.replace("ひゅ", " hy u")
    text = text.replace("ひょ", " hy o")
    text = text.replace("みゃ", " my a")
    text = text.replace("みゅ", " my u")
    text = text.replace("みょ", " my o")
    text = text.replace("りゃ", " ry a")
    text = text.replace("りゅ", " ry u")
    text = text.replace("りょ", " ry o")
    text = text.replace("ぎゃ", " gy a")
    text = text.replace("ぎゅ", " gy u")
    text = text.replace("ぎょ", " gy o")
    text = text.replace("ぢぇ", " j e")
    text = text.replace("ぢゃ", " j a")
    text = text.replace("ぢゅ", " j u")
    text = text.replace("ぢょ", " j o")
    text = text.replace("じぇ", " j e")
    text = text.replace("じゃ", " j a")
    text = text.replace("じゅ", " j u")
    text = text.replace("じょ", " j o")
    text = text.replace("びゃ", " by a")
    text = text.replace("びゅ", " by u")
    text = text.replace("びょ", " by o")
    text = text.replace("ぴゃ", " py a")
    text = text.replace("ぴゅ", " py u")
    text = text.replace("ぴょ", " py o")
    text = text.replace("うぁ", " u a")
    text = text.replace("うぃ", " w i")
    text = text.replace("うぇ", " w e")
    text = text.replace("うぉ", " w o")
    text = text.replace("ふぁ", " f a")
    text = text.replace("ふぃ", " f i")
    text = text.replace("ふぅ", " f u")
    text = text.replace("ふゃ", " hy a")
    text = text.replace("ふゅ", " hy u")
    text = text.replace("ふょ", " hy o")
    text = text.replace("ふぇ", " f e")
    text = text.replace("ふぉ", " f o")

    # 1音からなる変換規則
    text = text.replace("あ", " a")
    text = text.replace("い", " i")
    text = text.replace("う", " u")
    text = text.replace("ゔ", " v u")  # ゔの処理を追加
    text = text.replace("え", " e")
    text = text.replace("お", " o")
    text = text.replace("か", " k a")
    text = text.replace("き", " k i")
    text = text.replace("く", " k u")
    text = text.replace("け", " k e")
    text = text.replace("こ", " k o")
    text = text.replace("さ", " s a")
    text = text.replace("し", " sh i")
    text = text.replace("す", " s u")
    text = text.replace("せ", " s e")
    text = text.replace("そ", " s o")
    text = text.replace("た", " t a")
    text = text.replace("ち", " ch i")
    text = text.replace("つ", " ts u")
    text = text.replace("て", " t e")
    text = text.replace("と", " t o")
    text = text.replace("な", " n a")
    text = text.replace("に", " n i")
    text = text.replace("ぬ", " n u")
    text = text.replace("ね", " n e")
    text = text.replace("の", " n o")
    text = text.replace("は", " h a")
    text = text.replace("ひ", " h i")
    text = text.replace("ふ", " f u")
    text = text.replace("へ", " h e")
    text = text.replace("ほ", " h o")
    text = text.replace("ま", " m a")
    text = text.replace("み", " m i")
    text = text.replace("む", " m u")
    text = text.replace("め", " m e")
    text = text.replace("も", " m o")
    text = text.replace("ら", " r a")
    text = text.replace("り", " r i")
    text = text.replace("る", " r u")
    text = text.replace("れ", " r e")
    text = text.replace("ろ", " r o")
    text = text.replace("が", " g a")
    text = text.replace("ぎ", " g i")
    text = text.replace("ぐ", " g u")
    text = text.replace("げ", " g e")
    text = text.replace("ご", " g o")
    text = text.replace("ざ", " z a")
    text = text.replace("じ", " j i")
    text = text.replace("ず", " z u")
    text = text.replace("ぜ", " z e")
    text = text.replace("ぞ", " z o")
    text = text.replace("だ", " d a")
    text = text.replace("ぢ", " j i")
    text = text.replace("づ", " z u")
    text = text.replace("で", " d e")
    text = text.replace("ど", " d o")
    text = text.replace("ば", " b a")
    text = text.replace("び", " b i")
    text = text.replace("ぶ", " b u")
    text = text.replace("べ", " b e")
    text = text.replace("ぼ", " b o")
    text = text.replace("ぱ", " p a")
    text = text.replace("ぴ", " p i")
    text = text.replace("ぷ", " p u")
    text = text.replace("ぺ", " p e")
    text = text.replace("ぽ", " p o")
    text = text.replace("や", " y a")
    text = text.replace("ゆ", " y u")
    text = text.replace("よ", " y o")
    text = text.replace("わ", " w a")
    text = text.replace("ゐ", " i")
    text = text.replace("ゑ", " e")
    text = text.replace("ん", " N")
    text = text.replace("っ", " q")
    # ここまでに処理されてない ぁぃぅぇぉ はそのまま大文字扱い
    text = text.replace("ぁ", " a")
    text = text.replace("ぃ", " i")
    text = text.replace("ぅ", " u")
    text = text.replace("ぇ", " e")
    text = text.replace("ぉ", " o")
    text = text.replace("ゎ", " w a")
    text = text.replace("ぉ", " o")

    # ここまでに処理されていないゅ等もそのまま大文字扱い（追加）
    text = text.replace("ゃ", " y a")
    text = text.replace("ゅ", " y u")
    text = text.replace("ょ", " y o")

    # 長音の処理
    # for (pattern, replace_str) in JULIUS_LONG_VOWEL:
    #     text = pattern.sub(replace_str, text)
    # text = text.replace("o u", "o:")  # おう -> おーの音便
    text = text.replace("ー", ":")
    text = text.replace("〜", ":")
    text = text.replace("−", ":")
    text = text.replace("-", ":")

    # その他特別な処理
    text = text.replace("を", " o")

    text = text.strip()

    text = text.replace(":+", ":")

    # ここまで`jaconv.hiragana2julius`と音便処理と長音処理をのぞいて同じ
    # ここから`k a:: k i:`→`k a a a k i i`のように`:`の数だけ繰り返す処理
    pattern = r"(\w)(:*)"
    replacement = lambda m: m.group(1) + (" " + m.group(1)) * len(m.group(2))

    text = re.sub(pattern, replacement, text)
    text = text.replace("N", "n")  # 促音のNをnに変換
    return text


def kata2phoneme(text: str) -> str:
    """Convert katakana text to phonemes."""
    logger.debug(f"Kata2phoneme: {text}")
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
        logger.debug(f"text: {text}")
        logger.debug(jaconv.kata2hira(text))
        logger.debug(hiragana2p(jaconv.kata2hira(text)))
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
