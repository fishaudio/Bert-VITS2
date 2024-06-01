import re
import unicodedata

from num2words import num2words

from style_bert_vits2.nlp.symbols import PUNCTUATIONS


# 記号類の正規化マップ
__REPLACE_MAP = {
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
    # NFKC 正規化後のハイフン・ダッシュの変種を全て通常半角ハイフン - \u002d に変換
    "\u02d7": "\u002d",  # ˗, Modifier Letter Minus Sign
    "\u2010": "\u002d",  # ‐, Hyphen,
    # "\u2011": "\u002d",  # ‑, Non-Breaking Hyphen, NFKC により \u2010 に変換される
    "\u2012": "\u002d",  # ‒, Figure Dash
    "\u2013": "\u002d",  # –, En Dash
    "\u2014": "\u002d",  # —, Em Dash
    "\u2015": "\u002d",  # ―, Horizontal Bar
    "\u2043": "\u002d",  # ⁃, Hyphen Bullet
    "\u2212": "\u002d",  # −, Minus Sign
    "\u23af": "\u002d",  # ⎯, Horizontal Line Extension
    "\u23e4": "\u002d",  # ⏤, Straightness
    "\u2500": "\u002d",  # ─, Box Drawings Light Horizontal
    "\u2501": "\u002d",  # ━, Box Drawings Heavy Horizontal
    "\u2e3a": "\u002d",  # ⸺, Two-Em Dash
    "\u2e3b": "\u002d",  # ⸻, Three-Em Dash
    # "～": "-",  # これは長音記号「ー」として扱うよう変更
    # "~": "-",  # これも長音記号「ー」として扱うよう変更
    "「": "'",
    "」": "'",
}
# 記号類の正規化パターン
__REPLACE_PATTERN = re.compile("|".join(re.escape(p) for p in __REPLACE_MAP))
# 句読点等の正規化パターン
__PUNCTUATION_CLEANUP_PATTERN = re.compile(
    # ↓ ひらがな、カタカナ、漢字
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    # ↓ 半角アルファベット（大文字と小文字）
    + r"\u0041-\u005A\u0061-\u007A"
    # ↓ 全角アルファベット（大文字と小文字）
    + r"\uFF21-\uFF3A\uFF41-\uFF5A"
    # ↓ ギリシャ文字
    + r"\u0370-\u03FF\u1F00-\u1FFF"
    # ↓ "!", "?", "…", ",", ".", "'", "-", 但し`…`はすでに`...`に変換されている
    + "".join(PUNCTUATIONS) + r"]+",  # fmt: skip
)
# 数字・通貨記号の正規化パターン
__CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
__CURRENCY_PATTERN = re.compile(r"([$¥£€])([0-9.]*[0-9])")
__NUMBER_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)?")
__NUMBER_WITH_SEPARATOR_PATTERN = re.compile("[0-9]{1,3}(,[0-9]{3})+")


def normalize_text(text: str) -> str:
    """
    日本語のテキストを正規化する。
    結果は、ちょうど次の文字のみからなる：
    - ひらがな
    - カタカナ（全角長音記号「ー」が入る！）
    - 漢字
    - 半角アルファベット（大文字と小文字）
    - ギリシャ文字
    - `.` （句点`。`や`…`の一部や改行等）
    - `,` （読点`、`や`:`等）
    - `?` （疑問符`？`）
    - `!` （感嘆符`！`）
    - `'` （`「`や`」`等）
    - `-` （`―`（ダッシュ、長音記号ではない）や`-`等）

    注意点:
    - 三点リーダー`…`は`...`に変換される（`なるほど…。` → `なるほど....`）
    - 数字は漢字に変換される（`1,100円` → `千百円`、`52.34` → `五十二点三四`）
    - 読点や疑問符等の位置・個数等は保持される（`??あ、、！！！` → `??あ,,!!!`）

    Args:
        text (str): 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """

    res = unicodedata.normalize("NFKC", text)  # ここでアルファベットは半角になる
    res = __convert_numbers_to_words(res)  # 「100円」→「百円」等
    # 「～」と「〜」と「~」も長音記号として扱う
    res = res.replace("~", "ー")
    res = res.replace("～", "ー")
    res = res.replace("〜", "ー")

    res = replace_punctuation(res)  # 句読点等正規化、読めない文字を削除

    # 結合文字の濁点・半濁点を削除
    # 通常の「ば」等はそのままのこされる、「あ゛」は上で「あ゙」になりここで「あ」になる
    res = res.replace("\u3099", "")  # 結合文字の濁点を削除、る゙ → る
    res = res.replace("\u309A", "")  # 結合文字の半濁点を削除、な゚ → な
    return res


def replace_punctuation(text: str) -> str:
    """
    句読点等を「.」「,」「!」「?」「'」「-」に正規化し、OpenJTalk で読みが取得できるもののみ残す：
    漢字・平仮名・カタカナ、アルファベット、ギリシャ文字

    Args:
        text (str): 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """

    # 句読点を辞書で置換
    replaced_text = __REPLACE_PATTERN.sub(lambda x: __REPLACE_MAP[x.group()], text)

    # 上述以外の文字を削除
    replaced_text = __PUNCTUATION_CLEANUP_PATTERN.sub("", replaced_text)

    return replaced_text


def __convert_numbers_to_words(text: str) -> str:
    """
    記号や数字を日本語の文字表現に変換する。

    Args:
        text (str): 変換するテキスト

    Returns:
        str: 変換されたテキスト
    """

    res = __NUMBER_WITH_SEPARATOR_PATTERN.sub(lambda m: m[0].replace(",", ""), text)
    res = __CURRENCY_PATTERN.sub(lambda m: m[2] + __CURRENCY_MAP.get(m[1], m[1]), res)
    res = __NUMBER_PATTERN.sub(lambda m: num2words(m[0], lang="ja"), res)

    return res
