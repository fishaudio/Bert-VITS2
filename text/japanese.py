# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re
import unicodedata

import pyopenjtalk
from num2words import num2words
from transformers import AutoTokenizer

from common.log import logger
from text import punctuation
from text.japanese_mora_list import (
    mora_kata_to_mora_phonemes,
    mora_phonemes_to_mora_kata,
)

# 子音の集合
COSONANTS = set(
    [
        cosonant
        for cosonant, _ in mora_kata_to_mora_phonemes.values()
        if cosonant is not None
    ]
)

# 母音の集合、便宜上「ん」を含める
VOWELS = {"a", "i", "u", "e", "o", "N"}


# 正規化で記号を変換するための辞書
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
    # "～": "-",  # これは長音記号「ー」として扱うよう変更
    # "~": "-",  # これも長音記号「ー」として扱うよう変更
    "「": "'",
    "」": "'",
}


def text_normalize(text):
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
    """
    res = unicodedata.normalize("NFKC", text)  # ここでアルファベットは半角になる
    res = japanese_convert_numbers_to_words(res)  # 「100円」→「百円」等
    # 「～」と「~」も長音記号として扱う
    res = res.replace("~", "ー")
    res = res.replace("～", "ー")

    res = replace_punctuation(res)  # 句読点等正規化、読めない文字を削除

    # 結合文字の濁点・半濁点を削除
    # 通常の「ば」等はそのままのこされる、「あ゛」は上で「あ゙」になりここで「あ」になる
    res = res.replace("\u3099", "")  # 結合文字の濁点を削除、る゙ → る
    res = res.replace("\u309A", "")  # 結合文字の半濁点を削除、な゚ → な
    return res


def replace_punctuation(text: str) -> str:
    """句読点等を「.」「,」「!」「?」「'」「-」に正規化し、OpenJTalkで読みが取得できるもののみ残す：
    漢字・平仮名・カタカナ、アルファベット、ギリシャ文字
    """
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))

    # 句読点を辞書で置換
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    replaced_text = re.sub(
        # ↓ ひらがな、カタカナ、漢字
        r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
        # ↓ 半角アルファベット（大文字と小文字）
        + r"\u0041-\u005A\u0061-\u007A"
        # ↓ 全角アルファベット（大文字と小文字）
        + r"\uFF21-\uFF3A\uFF41-\uFF5A"
        # ↓ ギリシャ文字
        + r"\u0370-\u03FF\u1F00-\u1FFF"
        # ↓ "!", "?", "…", ",", ".", "'", "-", 但し`…`はすでに`...`に変換されている
        + "".join(punctuation) + r"]+",
        # 上述以外の文字を削除
        "",
        replaced_text,
    )

    return replaced_text


_NUMBER_WITH_SEPARATOR_RX = re.compile("[0-9]{1,3}(,[0-9]{3})+")
_CURRENCY_MAP = {"$": "ドル", "¥": "円", "£": "ポンド", "€": "ユーロ"}
_CURRENCY_RX = re.compile(r"([$¥£€])([0-9.]*[0-9])")
_NUMBER_RX = re.compile(r"[0-9]+(\.[0-9]+)?")


def japanese_convert_numbers_to_words(text: str) -> str:
    res = _NUMBER_WITH_SEPARATOR_RX.sub(lambda m: m[0].replace(",", ""), text)
    res = _CURRENCY_RX.sub(lambda m: m[2] + _CURRENCY_MAP.get(m[1], m[1]), res)
    res = _NUMBER_RX.sub(lambda m: num2words(m[0], lang="ja"), res)
    return res


def g2p(
    norm_text: str, use_jp_extra: bool = True
) -> tuple[list[str], list[int], list[int]]:
    """
    他で使われるメインの関数。`text_normalize()`で正規化された`norm_text`を受け取り、
    - phones: 音素のリスト（ただし`!`や`,`や`.`等punctuationが含まれうる）
    - tones: アクセントのリスト、0（低）と1（高）からなり、phonesと同じ長さ
    - word2ph: 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
    のタプルを返す。
    ただし`phones`と`tones`の最初と終わりに`_`が入り、応じて`word2ph`の最初と最後に1が追加される。
    use_jp_extra: Falseの場合、「ん」の音素を「N」ではなく「n」とする。
    """
    # pyopenjtalkのフルコンテキストラベルを使ってアクセントを取り出すと、punctuationの位置が消えてしまい情報が失われてしまう：
    # 「こんにちは、世界。」と「こんにちは！世界。」と「こんにちは！！！？？？世界……。」は全て同じになる。
    # よって、まずpunctuation無しの音素とアクセントのリストを作り、
    # それとは別にpyopenjtalk.run_frontend()で得られる音素リスト（こちらはpunctuationが保持される）を使い、
    # アクセント割当をしなおすことによってpunctuationを含めた音素とアクセントのリストを作る。

    # punctuationがすべて消えた、音素とアクセントのタプルのリスト（「ん」は「N」）
    phone_tone_list_wo_punct = g2phone_tone_wo_punct(norm_text)

    # sep_text: 単語単位の単語のリスト
    # sep_kata: 単語単位の単語のカタカナ読みのリスト
    sep_text, sep_kata = text2sep_kata(norm_text)

    # sep_phonemes: 各単語ごとの音素のリストのリスト
    sep_phonemes = handle_long([kata2phoneme_list(i) for i in sep_kata])

    # phone_w_punct: sep_phonemesを結合した、punctuationを元のまま保持した音素列
    phone_w_punct: list[str] = []
    for i in sep_phonemes:
        phone_w_punct += i

    # punctuation無しのアクセント情報を使って、punctuationを含めたアクセント情報を作る
    phone_tone_list = align_tones(phone_w_punct, phone_tone_list_wo_punct)
    # logger.debug(f"phone_tone_list:\n{phone_tone_list}")
    # word2phは厳密な解答は不可能なので（「今日」「眼鏡」等の熟字訓が存在）、
    # Bert-VITS2では、単語単位の分割を使って、単語の文字ごとにだいたい均等に音素を分配する

    # sep_textから、各単語を1文字1文字分割して、文字のリスト（のリスト）を作る
    sep_tokenized: list[list[str]] = []
    for i in sep_text:
        if i not in punctuation:
            sep_tokenized.append(
                tokenizer.tokenize(i)
            )  # ここでおそらく`i`が文字単位に分割される
        else:
            sep_tokenized.append([i])

    # 各単語について、音素の数と文字の数を比較して、均等っぽく分配する
    word2ph = []
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        phone_len = len(phoneme)
        word_len = len(token)
        word2ph += distribute_phone(phone_len, word_len)

    # 最初と最後に`_`記号を追加、アクセントは0（低）、word2phもそれに合わせて追加
    phone_tone_list = [("_", 0)] + phone_tone_list + [("_", 0)]
    word2ph = [1] + word2ph + [1]

    phones = [phone for phone, _ in phone_tone_list]
    tones = [tone for _, tone in phone_tone_list]

    assert len(phones) == sum(word2ph), f"{len(phones)} != {sum(word2ph)}"

    # use_jp_extraでない場合は「N」を「n」に変換
    if not use_jp_extra:
        phones = [phone if phone != "N" else "n" for phone in phones]

    return phones, tones, word2ph


def g2kata_tone(norm_text: str) -> list[tuple[str, int]]:
    phones, tones, _ = g2p(norm_text, use_jp_extra=True)
    return phone_tone2kata_tone(list(zip(phones, tones)))


def phone_tone2kata_tone(phone_tone: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """phone_toneをのphone部分をカタカナに変換する。ただし最初と最後の("_", 0)は無視"""
    phone_tone = phone_tone[1:]  # 最初の("_", 0)を無視
    phones = [phone for phone, _ in phone_tone]
    tones = [tone for _, tone in phone_tone]
    result: list[tuple[str, int]] = []
    current_mora = ""
    for phone, next_phone, tone, next_tone in zip(phones, phones[1:], tones, tones[1:]):
        # zipの関係で最後の("_", 0)は無視されている
        if phone in punctuation:
            result.append((phone, tone))
            continue
        if phone in COSONANTS:  # n以外の子音の場合
            assert current_mora == "", f"Unexpected {phone} after {current_mora}"
            assert tone == next_tone, f"Unexpected {phone} tone {tone} != {next_tone}"
            current_mora = phone
        else:
            # phoneが母音もしくは「N」
            current_mora += phone
            result.append((mora_phonemes_to_mora_kata[current_mora], tone))
            current_mora = ""
    return result


def kata_tone2phone_tone(kata_tone: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """`phone_tone2kata_tone()`の逆。"""
    result: list[tuple[str, int]] = [("_", 0)]
    for mora, tone in kata_tone:
        if mora in punctuation:
            result.append((mora, tone))
        else:
            cosonant, vowel = mora_kata_to_mora_phonemes[mora]
            if cosonant is None:
                result.append((vowel, tone))
            else:
                result.append((cosonant, tone))
                result.append((vowel, tone))
    result.append(("_", 0))
    return result


def g2phone_tone_wo_punct(text: str) -> list[tuple[str, int]]:
    """
    テキストに対して、音素とアクセント（0か1）のペアのリストを返す。
    ただし「!」「.」「?」等の非音素記号(punctuation)は全て消える（ポーズ記号も残さない）。
    非音素記号を含める処理は`align_tones()`で行われる。
    また「っ」は「q」に、「ん」は「N」に変換される。
    例: "こんにちは、世界ー。。元気？！" →
    [('k', 0), ('o', 0), ('N', 1), ('n', 1), ('i', 1), ('ch', 1), ('i', 1), ('w', 1), ('a', 1), ('s', 1), ('e', 1), ('k', 0), ('a', 0), ('i', 0), ('i', 0), ('g', 1), ('e', 1), ('N', 0), ('k', 0), ('i', 0)]
    """
    prosodies = pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True)
    # logger.debug(f"prosodies: {prosodies}")
    result: list[tuple[str, int]] = []
    current_phrase: list[tuple[str, int]] = []
    current_tone = 0
    for i, letter in enumerate(prosodies):
        # 特殊記号の処理

        # 文頭記号、無視する
        if letter == "^":
            assert i == 0, "Unexpected ^"
        # アクセント句の終わりに来る記号
        elif letter in ("$", "?", "_", "#"):
            # 保持しているフレーズを、アクセント数値を0-1に修正し結果に追加
            result.extend(fix_phone_tone(current_phrase))
            # 末尾に来る終了記号、無視（文中の疑問文は`_`になる）
            if letter in ("$", "?"):
                assert i == len(prosodies) - 1, f"Unexpected {letter}"
            # あとは"_"（ポーズ）と"#"（アクセント句の境界）のみ
            # これらは残さず、次のアクセント句に備える。
            current_phrase = []
            # 0を基準点にしてそこから上昇・下降する（負の場合は上の`fix_phone_tone`で直る）
            current_tone = 0
        # アクセント上昇記号
        elif letter == "[":
            current_tone = current_tone + 1
        # アクセント下降記号
        elif letter == "]":
            current_tone = current_tone - 1
        # それ以外は通常の音素
        else:
            if letter == "cl":  # 「っ」の処理
                letter = "q"
            # elif letter == "N":  # 「ん」の処理
            #     letter = "n"
            current_phrase.append((letter, current_tone))
    return result


def text2sep_kata(norm_text: str) -> tuple[list[str], list[str]]:
    """
    `text_normalize`で正規化済みの`norm_text`を受け取り、それを単語分割し、
    分割された単語リストとその読み（カタカナor記号1文字）のリストのタプルを返す。
    単語分割結果は、`g2p()`の`word2ph`で1文字あたりに割り振る音素記号の数を決めるために使う。
    例:
    `私はそう思う!って感じ?` →
    ["私", "は", "そう", "思う", "!", "って", "感じ", "?"], ["ワタシ", "ワ", "ソー", "オモウ", "!", "ッテ", "カンジ", "?"]
    """
    # parsed: OpenJTalkの解析結果
    parsed = pyopenjtalk.run_frontend(norm_text)
    sep_text: list[str] = []
    sep_kata: list[str] = []
    for parts in parsed:
        # word: 実際の単語の文字列
        # yomi: その読み、但し無声化サインの`’`は除去
        word, yomi = replace_punctuation(parts["string"]), parts["pron"].replace(
            "’", ""
        )
        """
        ここで`yomi`の取りうる値は以下の通りのはず。
        - `word`が通常単語 → 通常の読み（カタカナ）
            （カタカナからなり、長音記号も含みうる、`アー` 等）
        - `word`が`ー` から始まる → `ーラー` や `ーーー` など
        - `word`が句読点や空白等 → `、`
        - `word`が`?` → `？`（全角になる）
        他にも`word`が読めないキリル文字アラビア文字等が来ると`、`になるが、正規化でこの場合は起きないはず。
        また元のコードでは`yomi`が空白の場合の処理があったが、これは起きないはず。
        処理すべきは`yomi`が`、`の場合のみのはず。
        """
        assert yomi != "", f"Empty yomi: {word}"
        if yomi == "、":
            # wordは正規化されているので、`.`, `,`, `!`, `'`, `-`, `--` のいずれか
            if word not in (
                ".",
                ",",
                "!",
                "'",
                "-",
                "--",
            ):
                # ここはpyopenjtalkが読めない文字等のときに起こる
                raise ValueError(f"Cannot read: {word} in:\n{norm_text}")
            # yomiは元の記号のままに変更
            yomi = word
        elif yomi == "？":
            assert word == "?", f"yomi `？` comes from: {word}"
            yomi = "?"
        sep_text.append(word)
        sep_kata.append(yomi)
    return sep_text, sep_kata


# ESPnetの実装から引用、変更点無し。「ん」は「N」なことに注意。
# https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
def pyopenjtalk_g2p_prosody(text: str, drop_unvoiced_vowels: bool = True) -> list[str]:
    """Extract phoneme + prosoody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104

    """
    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def fix_phone_tone(phone_tone_list: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    `phone_tone_list`のtone（アクセントの値）を0か1の範囲に修正する。
    例: [(a, 0), (i, -1), (u, -1)] → [(a, 1), (i, 0), (u, 0)]
    """
    tone_values = set(tone for _, tone in phone_tone_list)
    if len(tone_values) == 1:
        assert tone_values == {0}, tone_values
        return phone_tone_list
    elif len(tone_values) == 2:
        if tone_values == {0, 1}:
            return phone_tone_list
        elif tone_values == {-1, 0}:
            return [
                (letter, 0 if tone == -1 else 1) for letter, tone in phone_tone_list
            ]
        else:
            raise ValueError(f"Unexpected tone values: {tone_values}")
    else:
        raise ValueError(f"Unexpected tone values: {tone_values}")


def distribute_phone(n_phone: int, n_word: int) -> list[int]:
    """
    左から右に1ずつ振り分け、次にまた左から右に1ずつ増やし、というふうに、
    音素の数`n_phone`を単語の数`n_word`に分配する。
    """
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def handle_long(sep_phonemes: list[list[str]]) -> list[list[str]]:
    for i in range(len(sep_phonemes)):
        if sep_phonemes[i][0] == "ー":
            sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]
    return sep_phonemes


tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese-char-wwm")


def align_tones(
    phones_with_punct: list[str], phone_tone_list: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    """
    例:
    …私は、、そう思う。
    phones_with_punct:
    [".", ".", ".", "w", "a", "t", "a", "sh", "i", "w", "a", ",", ",", "s", "o", "o", "o", "m", "o", "u", "."]
    phone_tone_list:
    [("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), ("_", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0))]
    Return:
    [(".", 0), (".", 0), (".", 0), ("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), (",", 0), (",", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0), (".", 0)]
    """
    result: list[tuple[str, int]] = []
    tone_index = 0
    for phone in phones_with_punct:
        if tone_index >= len(phone_tone_list):
            # 余ったpunctuationがある場合 → (punctuation, 0)を追加
            result.append((phone, 0))
        elif phone == phone_tone_list[tone_index][0]:
            # phone_tone_listの現在の音素と一致する場合 → toneをそこから取得、(phone, tone)を追加
            result.append((phone, phone_tone_list[tone_index][1]))
            # 探すindexを1つ進める
            tone_index += 1
        elif phone in punctuation:
            # phoneがpunctuationの場合 → (phone, 0)を追加
            result.append((phone, 0))
        else:
            logger.debug(f"phones: {phones_with_punct}")
            logger.debug(f"phone_tone_list: {phone_tone_list}")
            logger.debug(f"result: {result}")
            logger.debug(f"tone_index: {tone_index}")
            logger.debug(f"phone: {phone}")
            raise ValueError(f"Unexpected phone: {phone}")
    return result


def kata2phoneme_list(text: str) -> list[str]:
    """
    原則カタカナの`text`を受け取り、それをそのままいじらずに音素記号のリストに変換。
    注意点：
    - punctuationが来た場合（punctuationが1文字の場合がありうる）、処理せず1文字のリストを返す
    - 冒頭に続く「ー」はそのまま「ー」のままにする（`handle_long()`で処理される）
    - 文中の「ー」は前の音素記号の最後の音素記号に変換される。
    例：
    `ーーソーナノカーー` → ["ー", "ー", "s", "o", "o", "n", "a", "n", "o", "k", "a", "a", "a"]
    `?` → ["?"]
    """
    if text in punctuation:
        return [text]
    elif text == "--":
        return ["-", "-"]
    # `text`がカタカナ（`ー`含む）のみからなるかどうかをチェック
    if re.fullmatch(r"[\u30A0-\u30FF]+", text) is None:
        raise ValueError(f"Input must be katakana only: {text}")
    sorted_keys = sorted(mora_kata_to_mora_phonemes.keys(), key=len, reverse=True)
    pattern = "|".join(map(re.escape, sorted_keys))

    def mora2phonemes(mora: str) -> str:
        cosonant, vowel = mora_kata_to_mora_phonemes[mora]
        if cosonant is None:
            return f" {vowel}"
        return f" {cosonant} {vowel}"

    spaced_phonemes = re.sub(pattern, lambda m: mora2phonemes(m.group()), text)

    # 長音記号「ー」の処理
    long_pattern = r"(\w)(ー*)"
    long_replacement = lambda m: m.group(1) + (" " + m.group(1)) * len(m.group(2))
    spaced_phonemes = re.sub(long_pattern, long_replacement, spaced_phonemes)
    return spaced_phonemes.strip().split(" ")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")
    text = "hello,こんにちは、世界ー！……"
    from text.japanese_bert import get_bert_feature

    text = text_normalize(text)
    print(text)

    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)
