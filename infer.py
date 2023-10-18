"""
推理实现
"""
import torch
import commons
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
from oldVersion import V111, V110, V101


def get_text(text, language_str, hps, device):
    # 在此处实现当前版本的get_text
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str, device)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JP":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


def infer(
    text,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    sid,
    language,
    hps,
    net_g,
    device,
):
    # 支持中日双语版本
    inferMap_V2 = {
        "1.1.1": V111.infer,
        "1.1": V110.infer,
        "1.1.0": V110.infer,
    }
    # 仅支持中文版本
    # 在测试中，并未发现两个版本的模型不能互相通用
    inferMap_V1 = {
        "1.0.1": V101.infer,
        "1.0": V101.infer,
        "1.0.0": V101.infer,
    }
    version = hps.version if hasattr(hps, "version") else "1.1.1-dev"
    # 根据版本号选择合适的infer
    if version != "1.1.1-dev":
        if version in inferMap_V2.keys():
            return inferMap_V2[version](
                text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                sid,
                language,
                hps,
                net_g,
                device,
            )
        if version in inferMap_V1.keys():
            return inferMap_V1[version](
                text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                sid,
                hps,
                net_g,
                device,
            )
    # 在此处实现当前版本的推理
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps, device)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        torch.cuda.empty_cache()
        return audio
