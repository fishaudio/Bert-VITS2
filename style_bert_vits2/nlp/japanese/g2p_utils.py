from style_bert_vits2.nlp.japanese.g2p import g2p
from style_bert_vits2.nlp.japanese.mora_list import (
    CONSONANTS,
    MORA_KATA_TO_MORA_PHONEMES,
    MORA_PHONEMES_TO_MORA_KATA,
)
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


def g2kata_tone(norm_text: str) -> list[tuple[str, int]]:
    """
    テキストからカタカナとアクセントのペアのリストを返す。
    推論時のみに使われる関数のため、常に `raise_yomi_error=False` を指定して g2p() を呼ぶ仕様になっている。

    Args:
        norm_text: 正規化されたテキスト。

    Returns:
        カタカナと音高のリスト。
    """

    phones, tones, _ = g2p(norm_text, use_jp_extra=True, raise_yomi_error=False)
    return phone_tone2kata_tone(list(zip(phones, tones)))


def phone_tone2kata_tone(phone_tone: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    phone_tone の phone 部分をカタカナに変換する。ただし最初と最後の ("_", 0) は無視する。

    Args:
        phone_tone: 音素と音高のリスト。

    Returns:
        カタカナと音高のリスト。
    """

    phone_tone = phone_tone[1:]  # 最初の("_", 0)を無視
    phones = [phone for phone, _ in phone_tone]
    tones = [tone for _, tone in phone_tone]
    result: list[tuple[str, int]] = []
    current_mora = ""
    for phone, next_phone, tone, next_tone in zip(phones, phones[1:], tones, tones[1:]):
        # zip の関係で最後の ("_", 0) は無視されている
        if phone in PUNCTUATIONS:
            result.append((phone, tone))
            continue
        if phone in CONSONANTS:  # n以外の子音の場合
            assert current_mora == "", f"Unexpected {phone} after {current_mora}"
            assert tone == next_tone, f"Unexpected {phone} tone {tone} != {next_tone}"
            current_mora = phone
        else:
            # phoneが母音もしくは「N」
            current_mora += phone
            result.append((MORA_PHONEMES_TO_MORA_KATA[current_mora], tone))
            current_mora = ""

    return result


def kata_tone2phone_tone(kata_tone: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    `phone_tone2kata_tone()` の逆の変換を行う。

    Args:
        kata_tone: カタカナと音高のリスト。

    Returns:
        音素と音高のリスト。
    """

    result: list[tuple[str, int]] = [("_", 0)]
    for mora, tone in kata_tone:
        if mora in PUNCTUATIONS:
            result.append((mora, tone))
        else:
            consonant, vowel = MORA_KATA_TO_MORA_PHONEMES[mora]
            if consonant is None:
                result.append((vowel, tone))
            else:
                result.append((consonant, tone))
                result.append((vowel, tone))
    result.append(("_", 0))

    return result
