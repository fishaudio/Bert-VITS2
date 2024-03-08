from .japanese_bert import get_bert_feature as get_japanese_bert_feature


def get_bert(text, word2ph, language, device, assist_text=None, assist_text_weight=0.7):
    if language == "ZH":
        from .chinese_bert import get_bert_feature
    elif language == "EN":
        from .english_bert_mock import get_bert_feature
    elif language == "JP":
        # pyopenjtalkのworkerを1度だけ起動するため、ここでのimportは避ける
        # 他言語のようにimportすると、get_bertが呼ばれるたびにpyopenjtalkのworkerが起動してしまう
        get_bert_feature = get_japanese_bert_feature
    else:
        raise ValueError(f"Language {language} not supported")

    return get_bert_feature(text, word2ph, device, assist_text, assist_text_weight)
