from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp.symbols import (
    LANGUAGE_ID_MAP,
    LANGUAGE_TONE_START_MAP,
    SYMBOLS,
)


# __init__.py は配下のモジュールをインポートした時点で実行される
# PyTorch のインポートは重いので、型チェック時以外はインポートしない
if TYPE_CHECKING:
    import torch


__symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    language: Languages,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    テキストから BERT の特徴量を抽出する (PyTorch 推論)

    Args:
        text (str): テキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        language (Languages): テキストの言語
        device (str): 推論に利用するデバイス
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        torch.Tensor: BERT の特徴量
    """

    if language == Languages.JP:
        from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature
    elif language == Languages.EN:
        from style_bert_vits2.nlp.english.bert_feature import extract_bert_feature
    elif language == Languages.ZH:
        from style_bert_vits2.nlp.chinese.bert_feature import extract_bert_feature
    else:
        raise ValueError(f"Language {language} not supported")

    return extract_bert_feature(text, word2ph, device, assist_text, assist_text_weight)


def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    language: Languages,
    onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]],
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> NDArray[Any]:
    """
    テキストから BERT の特徴量を抽出する (ONNX 推論)

    Args:
        text (str): テキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        language (Languages): テキストの言語
        onnx_providers (list[str]): ONNX 推論で利用する ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider など)
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        NDArray[Any]: BERT の特徴量
    """

    if language == Languages.JP:
        from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature_onnx
    else:
        raise ValueError(f"Language {language} not supported")

    return extract_bert_feature_onnx(
        text,
        word2ph,
        onnx_providers,
        assist_text,
        assist_text_weight,
    )


def clean_text(
    text: str,
    language: Languages,
    use_jp_extra: bool = True,
    raise_yomi_error: bool = False,
) -> tuple[str, list[str], list[int], list[int]]:
    """
    テキストをクリーニングし、音素に変換する

    Args:
        text (str): クリーニングするテキスト
        language (Languages): テキストの言語
        use_jp_extra (bool, optional): テキストが日本語の場合に JP-Extra モデルを利用するかどうか。Defaults to True.
        raise_yomi_error (bool, optional): False の場合、読めない文字が消えたような扱いとして処理される。Defaults to False.

    Returns:
        tuple[str, list[str], list[int], list[int]]: クリーニングされたテキストと、音素・アクセント・元のテキストの各文字に音素が何個割り当てられるかのリスト
    """

    # Changed to import inside if condition to avoid unnecessary import
    if language == Languages.JP:
        from style_bert_vits2.nlp.japanese.g2p import g2p
        from style_bert_vits2.nlp.japanese.normalizer import normalize_text

        norm_text = normalize_text(text)
        phones, tones, word2ph = g2p(norm_text, use_jp_extra, raise_yomi_error)
    elif language == Languages.EN:
        from style_bert_vits2.nlp.english.g2p import g2p
        from style_bert_vits2.nlp.english.normalizer import normalize_text

        norm_text = normalize_text(text)
        phones, tones, word2ph = g2p(norm_text)
    elif language == Languages.ZH:
        from style_bert_vits2.nlp.chinese.g2p import g2p
        from style_bert_vits2.nlp.chinese.normalizer import normalize_text

        norm_text = normalize_text(text)
        phones, tones, word2ph = g2p(norm_text)
    else:
        raise ValueError(f"Language {language} not supported")

    return norm_text, phones, tones, word2ph


def clean_text_with_given_phone_tone(
    text: str,
    language: Languages,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
    use_jp_extra: bool = True,
    raise_yomi_error: bool = False,
) -> tuple[str, list[str], list[int], list[int]]:
    """
    テキストをクリーニングし、音素に変換する
    変換時、given_phone や given_tone が与えられた場合はそれを調整して使う

    Args:
        text (str): クリーニングするテキスト
        language (Languages): テキストの言語
        given_phone (Optional[list[int]], optional): 読み上げテキストの読みを表す音素列。指定する場合は given_tone も別途指定が必要. Defaults to None.
        given_tone (Optional[list[int]], optional): アクセントのトーンのリスト. Defaults to None.
        use_jp_extra (bool, optional): テキストが日本語の場合に JP-Extra モデルを利用するかどうか。Defaults to True.
        raise_yomi_error (bool, optional): False の場合、読めない文字が消えたような扱いとして処理される。Defaults to False.

    Returns:
        tuple[str, list[str], list[int], list[int]]: クリーニングされたテキストと、音素・アクセント・元のテキストの各文字に音素が何個割り当てられるかのリスト
    """

    # 与えられたテキストをクリーニング
    norm_text, phone, tone, word2ph = clean_text(
        text,
        language,
        use_jp_extra=use_jp_extra,
        raise_yomi_error=raise_yomi_error,
    )

    # phone と tone の両方が与えられた場合はそれを使う
    if given_phone is not None and given_tone is not None:
        # 指定された phone と指定された tone 両方の長さが一致していなければならない
        if len(given_phone) != len(given_tone):
            raise InvalidPhoneError(
                f"Length of given_phone ({len(given_phone)}) != length of given_tone ({len(given_tone)})"
            )
        # 与えられた音素数と pyopenjtalk で生成した読みの音素数が一致しない
        if len(given_phone) != sum(word2ph):
            # 日本語の場合、len(given_phone) と sum(word2ph) が一致するように word2ph を適切に調整する
            # 他の言語は word2ph の調整方法が思いつかないのでエラー
            if language == Languages.JP:
                from style_bert_vits2.nlp.japanese.g2p import adjust_word2ph

                # use_jp_extra でない場合は given_phone 内の「N」を「n」に変換
                if not use_jp_extra:
                    given_phone = [p if p != "N" else "n" for p in given_phone]
                # clean_text() から取得した word2ph を調整結果で上書き
                word2ph = adjust_word2ph(word2ph, phone, given_phone)
                # 上記処理により word2ph の合計が given_phone の長さと一致するはず
                # それでも一致しない場合、大半は読み上げテキストと given_phone が著しく乖離していて調整し切れなかったことを意味する
                if len(given_phone) != sum(word2ph):
                    raise InvalidPhoneError(
                        f"Length of given_phone ({len(given_phone)}) != sum of word2ph ({sum(word2ph)})"
                    )
            else:
                raise InvalidPhoneError(
                    f"Length of given_phone ({len(given_phone)}) != sum of word2ph ({sum(word2ph)})"
                )
        phone = given_phone
        # 生成あるいは指定された phone と指定された tone 両方の長さが一致していなければならない
        if len(phone) != len(given_tone):
            raise InvalidToneError(
                f"Length of phone ({len(phone)}) != length of given_tone ({len(given_tone)})"
            )
        tone = given_tone

    # tone だけが与えられた場合は clean_text() で生成した phone と合わせて使う
    elif given_tone is not None:
        # 生成した phone と指定された tone 両方の長さが一致していなければならない
        if len(phone) != len(given_tone):
            raise InvalidToneError(
                f"Length of phone ({len(phone)}) != length of given_tone ({len(given_tone)})"
            )
        tone = given_tone

    return norm_text, phone, tone, word2ph


def cleaned_text_to_sequence(
    cleaned_phones: list[str], tones: list[int], language: Languages
) -> tuple[list[int], list[int], list[int]]:
    """
    音素リスト・アクセントリスト・言語を、テキスト内の対応する ID に変換する

    Args:
        cleaned_phones (list[str]): clean_text() でクリーニングされた音素のリスト
        tones (list[int]): 各音素のアクセント
        language (Languages): テキストの言語

    Returns:
        tuple[list[int], list[int], list[int]]: List of integers corresponding to the symbols in the text
    """

    phones = [__symbol_to_id[symbol] for symbol in cleaned_phones]
    tone_start = LANGUAGE_TONE_START_MAP[language]
    tones = [i + tone_start for i in tones]
    lang_id = LANGUAGE_ID_MAP[language]
    lang_ids = [lang_id for i in phones]

    return phones, tones, lang_ids


class InvalidPhoneError(ValueError):
    pass


class InvalidToneError(ValueError):
    pass
