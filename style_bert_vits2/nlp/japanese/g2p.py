import re
from typing import TypedDict

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.mora_list import MORA_KATA_TO_MORA_PHONEMES, VOWELS
from style_bert_vits2.nlp.japanese.normalizer import replace_punctuation
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


def g2p(
    norm_text: str, use_jp_extra: bool = True, raise_yomi_error: bool = False
) -> tuple[list[str], list[int], list[int]]:
    """
    他で使われるメインの関数。`normalize_text()` で正規化された `norm_text` を受け取り、
    - phones: 音素のリスト（ただし `!` や `,` や `.` など punctuation が含まれうる）
    - tones: アクセントのリスト、0（低）と1（高）からなり、phones と同じ長さ
    - word2ph: 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
    のタプルを返す。
    ただし `phones` と `tones` の最初と終わりに `_` が入り、応じて `word2ph` の最初と最後に 1 が追加される。

    Args:
        norm_text (str): 正規化されたテキスト
        use_jp_extra (bool, optional): False の場合、「ん」の音素を「N」ではなく「n」とする。Defaults to True.
        raise_yomi_error (bool, optional): False の場合、読めない文字が「'」として発音される。Defaults to False.

    Returns:
        tuple[list[str], list[int], list[int]]: 音素のリスト、アクセントのリスト、word2ph のリスト
    """

    # pyopenjtalk のフルコンテキストラベルを使ってアクセントを取り出すと、punctuation の位置が消えてしまい情報が失われてしまう：
    # 「こんにちは、世界。」と「こんにちは！世界。」と「こんにちは！！！？？？世界……。」は全て同じになる。
    # よって、まず punctuation 無しの音素とアクセントのリストを作り、
    # それとは別に pyopenjtalk.run_frontend() で得られる音素リスト（こちらは punctuation が保持される）を使い、
    # アクセント割当をしなおすことによって punctuation を含めた音素とアクセントのリストを作る。

    # punctuation がすべて消えた、音素とアクセントのタプルのリスト（「ん」は「N」）
    phone_tone_list_wo_punct = __g2phone_tone_wo_punct(norm_text)

    # sep_text: 単語単位の単語のリスト
    # sep_kata: 単語単位の単語のカタカナ読みのリスト、読めない文字は raise_yomi_error=True なら例外、False なら読めない文字を「'」として返ってくる
    sep_text, sep_kata = text_to_sep_kata(norm_text, raise_yomi_error=raise_yomi_error)

    # sep_phonemes: 各単語ごとの音素のリストのリスト
    sep_phonemes = __handle_long([__kata_to_phoneme_list(i) for i in sep_kata])

    # phone_w_punct: sep_phonemes を結合した、punctuation を元のまま保持した音素列
    phone_w_punct: list[str] = []
    for i in sep_phonemes:
        phone_w_punct += i

    # punctuation 無しのアクセント情報を使って、punctuation を含めたアクセント情報を作る
    phone_tone_list = __align_tones(phone_w_punct, phone_tone_list_wo_punct)
    # logger.debug(f"phone_tone_list:\n{phone_tone_list}")

    # word2ph は厳密な解答は不可能なので（「今日」「眼鏡」等の熟字訓が存在）、
    # Bert-VITS2 では、単語単位の分割を使って、単語の文字ごとにだいたい均等に音素を分配する

    # sep_text から、各単語を1文字1文字分割して、文字のリスト（のリスト）を作る
    sep_tokenized: list[list[str]] = []
    for i in sep_text:
        if i not in PUNCTUATIONS:
            sep_tokenized.append(
                bert_models.load_tokenizer(Languages.JP).tokenize(i)
            )  # ここでおそらく`i`が文字単位に分割される
        else:
            sep_tokenized.append([i])

    # 各単語について、音素の数と文字の数を比較して、均等っぽく分配する
    word2ph = []
    for token, phoneme in zip(sep_tokenized, sep_phonemes):
        phone_len = len(phoneme)
        word_len = len(token)
        word2ph += __distribute_phone(phone_len, word_len)

    # 最初と最後に `_` 記号を追加、アクセントは 0（低）、word2ph もそれに合わせて追加
    phone_tone_list = [("_", 0)] + phone_tone_list + [("_", 0)]
    word2ph = [1] + word2ph + [1]

    phones = [phone for phone, _ in phone_tone_list]
    tones = [tone for _, tone in phone_tone_list]

    assert len(phones) == sum(word2ph), f"{len(phones)} != {sum(word2ph)}"

    # use_jp_extra でない場合は「N」を「n」に変換
    if not use_jp_extra:
        phones = [phone if phone != "N" else "n" for phone in phones]

    return phones, tones, word2ph


def text_to_sep_kata(
    norm_text: str, raise_yomi_error: bool = False
) -> tuple[list[str], list[str]]:
    """
    `normalize_text` で正規化済みの `norm_text` を受け取り、それを単語分割し、
    分割された単語リストとその読み（カタカナ or 記号1文字）のリストのタプルを返す。
    単語分割結果は、`g2p()` の `word2ph` で1文字あたりに割り振る音素記号の数を決めるために使う。
    例:
    `私はそう思う!って感じ?` →
    ["私", "は", "そう", "思う", "!", "って", "感じ", "?"], ["ワタシ", "ワ", "ソー", "オモウ", "!", "ッテ", "カンジ", "?"]

    Args:
        norm_text (str): 正規化されたテキスト
        raise_yomi_error (bool, optional): False の場合、読めない文字が「'」として発音される。Defaults to False.

    Returns:
        tuple[list[str], list[str]]: 分割された単語リストと、その読み（カタカナ or 記号1文字）のリスト
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
        ここで `yomi` の取りうる値は以下の通りのはず。
        - `word` が通常単語 → 通常の読み（カタカナ）
            （カタカナからなり、長音記号も含みうる、`アー` 等）
        - `word` が `ー` から始まる → `ーラー` や `ーーー` など
        - `word` が句読点や空白等 → `、`
        - `word` が punctuation の繰り返し → 全角にしたもの
        基本的に punctuation は1文字ずつ分かれるが、何故かある程度連続すると1つにまとまる。
        他にも `word` が読めないキリル文字アラビア文字等が来ると `、` になるが、正規化でこの場合は起きないはず。
        また元のコードでは `yomi` が空白の場合の処理があったが、これは起きないはず。
        処理すべきは `yomi` が `、` の場合のみのはず。
        """
        assert yomi != "", f"Empty yomi: {word}"
        if yomi == "、":
            # word は正規化されているので、`.`, `,`, `!`, `'`, `-`, `--` のいずれか
            if not set(word).issubset(set(PUNCTUATIONS)):  # 記号繰り返しか判定
                # ここは pyopenjtalk が読めない文字等のときに起こる
                ## 例外を送出する場合
                if raise_yomi_error:
                    raise YomiError(f"Cannot read: {word} in:\n{norm_text}")
                ## 例外を送出しない場合
                ## 読めない文字は「'」として扱う
                logger.warning(
                    f'Cannot read: {word} in:\n{norm_text}, replaced with "\'"'
                )
                # word の文字数分「'」を追加
                yomi = "'" * len(word)
            else:
                # yomi は元の記号のままに変更
                yomi = word
        elif yomi == "？":
            assert word == "?", f"yomi `？` comes from: {word}"
            yomi = "?"
        sep_text.append(word)
        sep_kata.append(yomi)

    return sep_text, sep_kata


def adjust_word2ph(
    word2ph: list[int],
    generated_phone: list[str],
    given_phone: list[str],
) -> list[int]:
    """
    `g2p()` で得られた `word2ph` を、generated_phone と given_phone の差分情報を使っていい感じに調整する。
    generated_phone は正規化された読み上げテキストから生成された読みの情報だが、
    given_phone で 同じ読み上げテキストに異なる読みが与えられた場合、正規化された読み上げテキストの各文字に
    音素が何文字割り当てられるかを示す word2ph の合計値が given_phone の長さ (音素数) と一致しなくなりうる
    そこで generated_phone と given_phone の差分を取り変更箇所に対応する word2ph の要素の値だけを増減させ、
    アクセントへの影響を最低限に抑えつつ word2ph の合計値を given_phone の長さ (音素数) に一致させる。

    Args:
        word2ph (list[int]): 単語ごとの音素の数のリスト
        generated_phone (list[str]): 生成された音素のリスト
        given_phone (list[str]): 与えられた音素のリスト

    Returns:
        list[int]: 修正された word2ph のリスト
    """

    # word2ph・generated_phone・given_phone 全ての先頭と末尾にダミー要素が入っているので、処理の都合上それらを削除
    # word2ph は先頭と末尾に 1 が入っている (返す際に再度追加する)
    word2ph = word2ph[1:-1]
    generated_phone = generated_phone[1:-1]
    given_phone = given_phone[1:-1]

    class DiffDetail(TypedDict):
        begin_index: int
        end_index: int
        value: list[str]

    class Diff(TypedDict):
        generated: DiffDetail
        given: DiffDetail

    def extract_differences(
        generated_phone: list[str], given_phone: list[str]
    ) -> list[Diff]:
        """
        最長共通部分列を基にして、二つのリストの異なる部分を抽出する。
        """

        def longest_common_subsequence(
            X: list[str], Y: list[str]
        ) -> list[tuple[int, int]]:
            """
            二つのリストの最長共通部分列のインデックスのペアを返す。
            """
            m, n = len(X), len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]
            # LCSの長さを構築
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i - 1] == Y[j - 1]:
                        L[i][j] = L[i - 1][j - 1] + 1
                    else:
                        L[i][j] = max(L[i - 1][j], L[i][j - 1])
            # LCSを逆方向にトレースしてインデックスのペアを取得
            index_pairs = []
            i, j = m, n
            while i > 0 and j > 0:
                if X[i - 1] == Y[j - 1]:
                    index_pairs.append((i - 1, j - 1))
                    i -= 1
                    j -= 1
                elif L[i - 1][j] >= L[i][j - 1]:
                    i -= 1
                else:
                    j -= 1
            index_pairs.reverse()
            return index_pairs

        differences = []
        common_indices = longest_common_subsequence(generated_phone, given_phone)
        prev_x, prev_y = -1, -1

        # 共通部分のインデックスを基にして差分を抽出
        for x, y in common_indices:
            diff_X = {
                "begin_index": prev_x + 1,
                "end_index": x,
                "value": generated_phone[prev_x + 1 : x],
            }
            diff_Y = {
                "begin_index": prev_y + 1,
                "end_index": y,
                "value": given_phone[prev_y + 1 : y],
            }
            if diff_X or diff_Y:
                differences.append({"generated": diff_X, "given": diff_Y})
            prev_x, prev_y = x, y
        # 最後の非共通部分を追加
        if prev_x < len(generated_phone) - 1 or prev_y < len(given_phone) - 1:
            differences.append(
                {
                    "generated": {
                        "begin_index": prev_x + 1,
                        "end_index": len(generated_phone) - 1,
                        "value": generated_phone[prev_x + 1 : len(generated_phone) - 1],
                    },
                    "given": {
                        "begin_index": prev_y + 1,
                        "end_index": len(given_phone) - 1,
                        "value": given_phone[prev_y + 1 : len(given_phone) - 1],
                    },
                }
            )
        # generated.value と given.value の両方が空の要素を diffrences から削除
        for diff in differences[:]:
            if (
                len(diff["generated"]["value"]) == 0
                and len(diff["given"]["value"]) == 0
            ):
                differences.remove(diff)

        return differences

    # 二つのリストの差分を抽出
    differences = extract_differences(generated_phone, given_phone)

    # word2ph をもとにして新しく作る word2ph のリスト
    ## 長さは word2ph と同じだが、中身は 0 で初期化されている
    adjusted_word2ph: list[int] = [0] * len(word2ph)
    # 現在処理中の generated_phone のインデックス
    current_generated_index = 0

    # word2ph の要素数 (=正規化された読み上げテキストの文字数) を維持しながら、差分情報を使って word2ph を修正
    ## 音素数が generated_phone と given_phone で異なる場合にこの align_word2ph() が呼び出される
    ## word2ph は正規化された読み上げテキストの文字数に対応しているので、要素数はそのまま given_phone で増減した音素数に合わせて各要素の値を増減する
    for word2ph_element_index, word2ph_element in enumerate(word2ph):
        # ここの word2ph_element は、正規化された読み上げテキストの各文字に割り当てられる音素の数を示す
        # 例えば word2ph_element が 2 ならば、その文字には 2 つの音素 (例: "k", "a") が割り当てられる
        # 音素の数だけループを回す
        for _ in range(word2ph_element):
            # difference の中に 処理中の generated_phone から始まる差分があるかどうかを確認
            current_diff: Diff | None = None
            for diff in differences:
                if diff["generated"]["begin_index"] == current_generated_index:
                    current_diff = diff
                    break
            # current_diff が None でない場合、generated_phone から始まる差分がある
            if current_diff is not None:
                # generated から given で変わった音素数の差分を取得 (2増えた場合は +2 だし、2減った場合は -2)
                diff_in_phonemes = \
                    len(current_diff["given"]["value"]) - len(current_diff["generated"]["value"])  # fmt: skip
                # adjusted_word2ph[(読み上げテキストの各文字のインデックス)] に上記差分を反映
                adjusted_word2ph[word2ph_element_index] += diff_in_phonemes
            # adjusted_word2ph[(読み上げテキストの各文字のインデックス)] に処理が完了した分の音素として 1 を加える
            adjusted_word2ph[word2ph_element_index] += 1
            # 処理中の generated_phone のインデックスを進める
            current_generated_index += 1

    # この時点で given_phone の長さと adjusted_word2ph に記録されている音素数の合計が一致しているはず
    assert len(given_phone) == sum(adjusted_word2ph), f"{len(given_phone)} != {sum(adjusted_word2ph)}"  # fmt: skip

    # generated_phone から given_phone の間で音素が減った場合 (例: a, sh, i, t, a -> a, s, u) 、
    # adjusted_word2ph の要素の値が 1 未満になることがあるので、1 になるように値を増やす
    ## この時、adjusted_word2ph に記録されている音素数の合計を変えないために、
    ## 値を 1 にした分だけ右隣の要素から増やした分の差分を差し引く
    for adjusted_word2ph_element_index, adjusted_word2ph_element in enumerate(adjusted_word2ph):  # fmt: skip
        # もし現在の要素が 1 未満ならば
        if adjusted_word2ph_element < 1:
            # 値を 1 にするためにどれだけ足せばいいかを計算
            diff = 1 - adjusted_word2ph_element
            # adjusted_word2ph[(読み上げテキストの各文字のインデックス)] を 1 にする
            # これにより、当該文字に最低ラインとして 1 つの音素が割り当てられる
            adjusted_word2ph[adjusted_word2ph_element_index] = 1
            # 次の要素のうち、一番近くてかつ 1 以上の要素から diff を引く
            # この時、diff を引いた結果引いた要素が 1 未満になる場合は、その要素の次の要素の中から一番近くてかつ 1 以上の要素から引く
            # 上記を繰り返していって、diff が 0 になるまで続ける
            for i in range(1, len(adjusted_word2ph)):
                if adjusted_word2ph_element_index + i >= len(adjusted_word2ph):
                    break  # adjusted_word2ph の最後に達した場合は諦める
                if adjusted_word2ph[adjusted_word2ph_element_index + i] - diff >= 1:
                    adjusted_word2ph[adjusted_word2ph_element_index + i] -= diff
                    break
                else:
                    diff -= adjusted_word2ph[adjusted_word2ph_element_index + i] - 1
                    adjusted_word2ph[adjusted_word2ph_element_index + i] = 1
                    if diff == 0:
                        break

    # 逆に、generated_phone から given_phone の間で音素が増えた場合 (例: a, s, u -> a, sh, i, t, a) 、
    # 1文字あたり7音素以上も割り当てられてしまう場合があるので、最大6音素にした上で削った分の差分を次の要素に加える
    # 次の要素に差分を加えた結果7音素以上になってしまう場合は、その差分をさらに次の要素に加える
    for adjusted_word2ph_element_index, adjusted_word2ph_element in enumerate(adjusted_word2ph):  # fmt: skip
        if adjusted_word2ph_element > 6:
            diff = adjusted_word2ph_element - 6
            adjusted_word2ph[adjusted_word2ph_element_index] = 6
            for i in range(1, len(adjusted_word2ph)):
                if adjusted_word2ph_element_index + i >= len(adjusted_word2ph):
                    break  # adjusted_word2ph の最後に達した場合は諦める
                if adjusted_word2ph[adjusted_word2ph_element_index + i] + diff <= 6:
                    adjusted_word2ph[adjusted_word2ph_element_index + i] += diff
                    break
                else:
                    diff -= 6 - adjusted_word2ph[adjusted_word2ph_element_index + i]
                    adjusted_word2ph[adjusted_word2ph_element_index + i] = 6
                    if diff == 0:
                        break

    # この時点で given_phone の長さと adjusted_word2ph に記録されている音素数の合計が一致していない場合、
    # 正規化された読み上げテキストと given_phone が著しく乖離していることを示す
    # このとき、この関数の呼び出し元の get_text() にて InvalidPhoneError が送出される

    # 最初に削除した前後のダミー要素を追加して返す
    return [1] + adjusted_word2ph + [1]


def __g2phone_tone_wo_punct(text: str) -> list[tuple[str, int]]:
    """
    テキストに対して、音素とアクセント（0か1）のペアのリストを返す。
    ただし「!」「.」「?」等の非音素記号 (punctuation) は全て消える（ポーズ記号も残さない）。
    非音素記号を含める処理は `align_tones()` で行われる。
    また「っ」は「q」に、「ん」は「N」に変換される。
    例: "こんにちは、世界ー。。元気？！" →
    [('k', 0), ('o', 0), ('N', 1), ('n', 1), ('i', 1), ('ch', 1), ('i', 1), ('w', 1), ('a', 1), ('s', 1), ('e', 1), ('k', 0), ('a', 0), ('i', 0), ('i', 0), ('g', 1), ('e', 1), ('N', 0), ('k', 0), ('i', 0)]

    Args:
        text (str): テキスト

    Returns:
        list[tuple[str, int]]: 音素とアクセントのペアのリスト
    """

    prosodies = __pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True)
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
            # 保持しているフレーズを、アクセント数値を 0-1 に修正し結果に追加
            result.extend(__fix_phone_tone(current_phrase))
            # 末尾に来る終了記号、無視（文中の疑問文は `_` になる）
            if letter in ("$", "?"):
                assert i == len(prosodies) - 1, f"Unexpected {letter}"
            # あとは "_"（ポーズ）と "#"（アクセント句の境界）のみ
            # これらは残さず、次のアクセント句に備える。
            current_phrase = []
            # 0 を基準点にしてそこから上昇・下降する（負の場合は上の `fix_phone_tone` で直る）
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


__PYOPENJTALK_G2P_PROSODY_A1_PATTERN = re.compile(r"/A:([0-9\-]+)\+")
__PYOPENJTALK_G2P_PROSODY_A2_PATTERN = re.compile(r"\+(\d+)\+")
__PYOPENJTALK_G2P_PROSODY_A3_PATTERN = re.compile(r"\+(\d+)/")
__PYOPENJTALK_G2P_PROSODY_E3_PATTERN = re.compile(r"!(\d+)_")
__PYOPENJTALK_G2P_PROSODY_F1_PATTERN = re.compile(r"/F:(\d+)_")
__PYOPENJTALK_G2P_PROSODY_P3_PATTERN = re.compile(r"\-(.*?)\+")


def __pyopenjtalk_g2p_prosody(
    text: str, drop_unvoiced_vowels: bool = True
) -> list[str]:
    """
    ESPnet の実装から引用、概ね変更点無し。「ん」は「N」なことに注意。
    ref: https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py
    ------------------------------------------------------------------------------------------

    Extract phoneme + prosody symbol sequence from input full-context labels.

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

    def _numeric_feature_by_regex(pattern: re.Pattern[str], s: str) -> int:
        match = pattern.search(s)
        if match is None:
            return -50
        return int(match.group(1))

    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = __PYOPENJTALK_G2P_PROSODY_P3_PATTERN.search(lab_curr).group(1)  # type: ignore
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
                e3 = _numeric_feature_by_regex(
                    __PYOPENJTALK_G2P_PROSODY_E3_PATTERN, lab_curr
                )
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
        a1 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A1_PATTERN, lab_curr)
        a2 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A2_PATTERN, lab_curr)
        a3 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_A3_PATTERN, lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(__PYOPENJTALK_G2P_PROSODY_F1_PATTERN, lab_curr)

        a2_next = _numeric_feature_by_regex(
            __PYOPENJTALK_G2P_PROSODY_A2_PATTERN, labels[n + 1]
        )
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


def __fix_phone_tone(phone_tone_list: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    `phone_tone_list` の tone（アクセントの値）を 0 か 1 の範囲に修正する。
    例: [(a, 0), (i, -1), (u, -1)] → [(a, 1), (i, 0), (u, 0)]

    Args:
        phone_tone_list (list[tuple[str, int]]): 音素とアクセントのペアのリスト

    Returns:
        list[tuple[str, int]]: 修正された音素とアクセントのペアのリスト
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


def __handle_long(sep_phonemes: list[list[str]]) -> list[list[str]]:
    """
    フレーズごとに分かれた音素（長音記号がそのまま）のリストのリスト `sep_phonemes` を受け取り、
    その長音記号を処理して、音素のリストのリストを返す。
    基本的には直前の音素を伸ばすが、直前の音素が母音でない場合もしくは冒頭の場合は、
    おそらく長音記号とダッシュを勘違いしていると思われるので、ダッシュに対応する音素 `-` に変換する。

    Args:
        sep_phonemes (list[list[str]]): フレーズごとに分かれた音素のリストのリスト

    Returns:
        list[list[str]]: 長音記号を処理した音素のリストのリスト
    """

    for i in range(len(sep_phonemes)):
        if len(sep_phonemes[i]) == 0:
            # 空白文字等でリストが空の場合
            continue
        if sep_phonemes[i][0] == "ー":
            if i != 0:
                prev_phoneme = sep_phonemes[i - 1][-1]
                if prev_phoneme in VOWELS:
                    # 母音と「ん」のあとの伸ばし棒なので、その母音に変換
                    sep_phonemes[i][0] = sep_phonemes[i - 1][-1]
                else:
                    # 「。ーー」等おそらく予期しない長音記号
                    # ダッシュの勘違いだと思われる
                    sep_phonemes[i][0] = "-"
            else:
                # 冒頭に長音記号が来ていおり、これはダッシュの勘違いと思われる
                sep_phonemes[i][0] = "-"
        if "ー" in sep_phonemes[i]:
            for j in range(len(sep_phonemes[i])):
                if sep_phonemes[i][j] == "ー":
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1][-1]

    return sep_phonemes


__KATAKANA_PATTERN = re.compile(r"[\u30A0-\u30FF]+")
__MORA_PATTERN = re.compile(
    "|".join(
        map(re.escape, sorted(MORA_KATA_TO_MORA_PHONEMES.keys(), key=len, reverse=True))
    )
)
__LONG_PATTERN = re.compile(r"(\w)(ー*)")


def __kata_to_phoneme_list(text: str) -> list[str]:
    """
    原則カタカナの `text` を受け取り、それをそのままいじらずに音素記号のリストに変換。
    注意点：
    - punctuation かその繰り返しが来た場合、punctuation たちをそのままリストにして返す。
    - 冒頭に続く「ー」はそのまま「ー」のままにする（`handle_long()` で処理される）
    - 文中の「ー」は前の音素記号の最後の音素記号に変換される。
    例：
    `ーーソーナノカーー` → ["ー", "ー", "s", "o", "o", "n", "a", "n", "o", "k", "a", "a", "a"]
    `?` → ["?"]
    `!?!?!?!?!` → ["!", "?", "!", "?", "!", "?", "!", "?", "!"]

    Args:
        text (str): カタカナのテキスト

    Returns:
        list[str]: 音素記号のリスト
    """

    if set(text).issubset(set(PUNCTUATIONS)):
        return list(text)
    # `text` がカタカナ（`ー`含む）のみからなるかどうかをチェック
    if __KATAKANA_PATTERN.fullmatch(text) is None:
        raise ValueError(f"Input must be katakana only: {text}")

    def mora2phonemes(mora: str) -> str:
        consonant, vowel = MORA_KATA_TO_MORA_PHONEMES[mora]
        if consonant is None:
            return f" {vowel}"
        return f" {consonant} {vowel}"

    spaced_phonemes = __MORA_PATTERN.sub(lambda m: mora2phonemes(m.group()), text)

    # 長音記号「ー」の処理
    long_replacement = lambda m: m.group(1) + (" " + m.group(1)) * len(m.group(2))  # type: ignore
    spaced_phonemes = __LONG_PATTERN.sub(long_replacement, spaced_phonemes)

    return spaced_phonemes.strip().split(" ")


def __align_tones(
    phones_with_punct: list[str], phone_tone_list: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    """
    例: …私は、、そう思う。
    phones_with_punct:
        [".", ".", ".", "w", "a", "t", "a", "sh", "i", "w", "a", ",", ",", "s", "o", "o", "o", "m", "o", "u", "."]
    phone_tone_list:
        [("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), ("_", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0))]
    Return:
        [(".", 0), (".", 0), (".", 0), ("w", 0), ("a", 0), ("t", 1), ("a", 1), ("sh", 1), ("i", 1), ("w", 1), ("a", 1), (",", 0), (",", 0), ("s", 0), ("o", 0), ("o", 1), ("o", 1), ("m", 1), ("o", 1), ("u", 0), (".", 0)]

    Args:
        phones_with_punct (list[str]): punctuation を含む音素のリスト
        phone_tone_list (list[tuple[str, int]]): punctuation を含まない音素とアクセントのペアのリスト

    Returns:
        list[tuple[str, int]]: punctuation を含む音素とアクセントのペアのリスト
    """

    result: list[tuple[str, int]] = []
    tone_index = 0
    for phone in phones_with_punct:
        if tone_index >= len(phone_tone_list):
            # 余った punctuation がある場合 → (punctuation, 0) を追加
            result.append((phone, 0))
        elif phone == phone_tone_list[tone_index][0]:
            # phone_tone_list の現在の音素と一致する場合 → tone をそこから取得、(phone, tone) を追加
            result.append((phone, phone_tone_list[tone_index][1]))
            # 探す index を1つ進める
            tone_index += 1
        elif phone in PUNCTUATIONS:
            # phone が punctuation の場合 → (phone, 0) を追加
            result.append((phone, 0))
        else:
            logger.debug(f"phones: {phones_with_punct}")
            logger.debug(f"phone_tone_list: {phone_tone_list}")
            logger.debug(f"result: {result}")
            logger.debug(f"tone_index: {tone_index}")
            logger.debug(f"phone: {phone}")
            raise ValueError(f"Unexpected phone: {phone}")

    return result


def __distribute_phone(n_phone: int, n_word: int) -> list[int]:
    """
    左から右に 1 ずつ振り分け、次にまた左から右に1ずつ増やし、というふうに、
    音素の数 `n_phone` を単語の数 `n_word` に分配する。

    Args:
        n_phone (int): 音素の数
        n_word (int): 単語の数

    Returns:
        list[int]: 単語ごとの音素の数のリスト
    """

    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1

    return phones_per_word


class YomiError(Exception):
    """
    OpenJTalk で、読みが正しく取得できない箇所があるときに発生する例外。
    基本的に「学習の前処理のテキスト処理時」には発生させ、そうでない場合は、
    ignore_yomi_error=True にしておいて、この例外を発生させないようにする。
    """
