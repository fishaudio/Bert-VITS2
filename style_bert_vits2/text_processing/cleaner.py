from typing import Literal


def clean_text(
    text: str,
    language: Literal["JP", "EN", "ZH"],
    use_jp_extra: bool = True,
    raise_yomi_error: bool = False,
) -> tuple[str, list[str], list[int], list[int]]:
    """
    テキストをクリーニングし、音素に変換する

    Args:
        text (str): クリーニングするテキスト
        language (Literal["JP", "EN", "ZH"]): テキストの言語
        use_jp_extra (bool, optional): テキストが日本語の場合に JP-Extra モデルを利用するかどうか。Defaults to True.
        raise_yomi_error (bool, optional): False の場合、読めない文字が消えたような扱いとして処理される。Defaults to False.

    Returns:
        tuple[str, list[str], list[int], list[int]]: クリーニングされたテキストと、音素・アクセント・元のテキストの各文字に音素が何個割り当てられるかのリスト
    """

    # Changed to import inside if condition to avoid unnecessary import
    if language == "JP":
        from transformers import AutoTokenizer
        from style_bert_vits2.text_processing.japanese.g2p import g2p
        from style_bert_vits2.text_processing.japanese.normalizer import normalize_text
        norm_text = normalize_text(text)
        phones, tones, word2ph = g2p(
            norm_text,
            tokenizer = AutoTokenizer.from_pretrained("./bert/deberta-v2-large-japanese-char-wwm"),  # 暫定的にここで指定
            use_jp_extra = use_jp_extra,
            raise_yomi_error = raise_yomi_error,
        )
    elif language == "EN":
        from ...text import english as language_module
        norm_text = language_module.normalize_text(text)
        phones, tones, word2ph = language_module.g2p(norm_text)
    elif language == "ZH":
        from ...text import chinese as language_module
        norm_text = language_module.normalize_text(text)
        phones, tones, word2ph = language_module.g2p(norm_text)
    else:
        raise ValueError(f"Language {language} not supported")

    return norm_text, phones, tones, word2ph
