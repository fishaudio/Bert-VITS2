from typing import TYPE_CHECKING, Optional

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
) -> "torch.Tensor":
    """
    テキストから BERT の特徴量を抽出する

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
