# Convert Japanese text to phonemes which is
# compatible with Julius https://github.com/julius-speech/segmentation-kit
import re
import unicodedata

import jaconv
import pyopenjtalk
from num2words import num2words
from transformers import AutoTokenizer

from common.log import logger
from text import punctuation

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
    "~": "-",
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


def g2p(norm_text: str) -> tuple[list[str], list[int], list[int]]:
    """
    他で使われるメインの関数。`text_normalize()`で正規化された`norm_text`を受け取り、
    - phones: 音素のリスト（ただし`!`や`,`や`.`等punctuationが含まれうる）
    - tones: アクセントのリスト、0（低）と1（高）からなり、phonesと同じ長さ
    - word2ph: 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
    のタプルを返す。
    ただし`phones`と`tones`の最初と終わりに`_`が入り、応じて`word2ph`の最初と最後に1が追加される。
    """
    # pyopenjtalkのフルコンテキストラベルを使ってアクセントを取り出すと、punctuationの位置が消えてしまい情報が失われてしまう：
    # 「こんにちは、世界。」と「こんにちは！世界。」と「こんにちは！！！？？？世界……。」は全て同じになる。
    # よって、まずpunctuation無しの音素とアクセントのリストを作り、
    # それとは別にpyopenjtalk.run_frontend()で得られる音素リスト（こちらはpunctuationが保持される）を使い、
    # アクセント割当をしなおすことによってpunctuationを含めた音素とアクセントのリストを作る。

    # punctuationがすべて消えた、音素とアクセントのタプルのリスト
    phone_tone_list_wo_punct = g2phone_tone_list(norm_text)

    # sep_text: 単語単位の単語のリスト
    # sep_kata: 単語単位の単語のカタカナ読みのリスト
    sep_text, sep_kata = text2sep_kata(norm_text)

    # sep_phonemes: 各単語ごとの音素のリストのリスト
    sep_phonemes = handle_long([kata2phoneme(i) for i in sep_kata])

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
            sep_tokenized.append(tokenizer.tokenize(i))  # ここでおそらく`i`が文字単位に分割される
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

    return phones, tones, word2ph


def g2phone_tone_list(text: str) -> list[tuple[str, int]]:
    """
    テキストに対して、音素とアクセント（0か1）のペアのリストを返す。
    ただし「!」「.」「?」等の非音素記号は全て消える（ポーズ記号も残さない）。
    非音素記号を含める処理は`align_tones()`で行われる。
    また「っ」は「q」に、「ん」は「n」に変換される。
    例: "こんにちは、世界ー。。元気？！" →
    [('k', 0), ('o', 0), ('n', 1), ('n', 1), ('i', 1), ('ch', 1), ('i', 1), ('w', 1), ('a', 1), ('s', 1), ('e', 1), ('k', 0), ('a', 0), ('i', 0), ('i', 0), ('g', 1), ('e', 1), ('n', 0), ('k', 0), ('i', 0)]
    """
    prosodies = pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True)
    logger.debug(f"prosodies: {prosodies}")
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
            elif letter == "N":  # 「ん」の処理
                letter = "n"
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
            assert len(word) == 1, f"yomi `、` comes from: {word}"
            # wordは1文字の記号。正規化されているので、`.`, `,`, `!`, `'`, `-`のいずれか
            if word not in (
                ".",
                ",",
                "!",
                "'",
                "-",
            ):
                raise ValueError(f"Cannot read: {word} in:\n{norm_text}")
            # yomiは元の記号のままに変更
            yomi = word
        elif yomi == "？":
            assert word == "?", f"yomi `？` comes from: {word}"
            yomi = "?"
        sep_text.append(word)
        sep_kata.append(yomi)
    return sep_text, sep_kata


def kata2phoneme(text: str) -> list[str]:
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
    # `text`がカタカナ（`ー`含む）のみからなるかどうかをチェック
    if re.fullmatch(r"[\u30A0-\u30FF]+", text) is None:
        raise ValueError(f"Non-punctuation input must be katakana only: {text}")
    # 以降`text`はカタカナのみからなる
    if text == "ー":
        return ["ー"]
    elif text.startswith("ー"):
        return ["ー"] + kata2phoneme(text[1:])
    res: list[str] = []
    while text:
        # カタカナをひらがなに変換してから`hiragana2p`をかける
        res += hiragana2p(jaconv.kata2hira(text)).split(" ")
        break
    return res


# ESPnetの実装から引用、変更点無し
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


# jaconvから借りて修正
# https://github.com/ikegami-yukino/jaconv/blob/master/jaconv/jaconv.py
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
    text = text.replace("N", "n")  # 「ん」のNをnに変換
    return text


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese")
    text = "hello,こんにちは、世界ー！……"
    from text.japanese_bert import get_bert_feature

    text = text_normalize(text)
    print(text)

    phones, tones, word2ph = g2p(text)
    bert = get_bert_feature(text, word2ph)

    print(phones, tones, word2ph, bert.shape)
