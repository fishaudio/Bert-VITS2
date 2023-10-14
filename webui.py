# flake8: noqa: E402
import re
import sys, os
import logging
import re_matching
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
import gradio as gr
import webbrowser
import numpy as np

net_g = None

if sys.platform == "darwin" and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = "cuda"


def get_text(text, language_str, hps):
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


def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language):
    global net_g
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
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


def generate_audio(slices, sdp_ratio, noise_scale, noise_scale_w, length_scale, speaker, language):
    audio_list = []
    silence = np.zeros(hps.data.sampling_rate // 2)
    with torch.no_grad():
        for piece in slices:
            audio = infer(
                piece,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
            )
            audio_list.append(audio)
            audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def tts_fn(text: str, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language):
    audio_list = []
    if speaker == "mix":
        bool_valid, str_valid = re_matching.validate_text(text)
        if not bool_valid:
            return str_valid, (hps.data.sampling_rate, np.concatenate([np.zeros(hps.data.sampling_rate // 2)]))
        result = re_matching.text_matching(text)
        for one in result:
            _speaker = one.pop()
            for lang, content in one:
                audio_list.extend(
                    generate_audio(content.split("|"), sdp_ratio, noise_scale,
                                   noise_scale_w, length_scale, _speaker+'_'+lang.lower(), lang)
                )
    else:
        audio_list.extend(
            generate_audio(text.split("|"), sdp_ratio, noise_scale, noise_scale_w, length_scale, speaker, language)
        )

    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="./logs/as/G_8000.pth", help="path of your model"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/config.json",
        help="path of your config file",
    )
    parser.add_argument(
        "--share", default=False, help="make link public", action="store_true"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="enable DEBUG-LEVEL log"
    )

    args = parser.parse_args()
    if args.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(args.config)

    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else (
            "mps"
            if sys.platform == "darwin" and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model, net_g, None, skip_optimizer=True)

    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP", "mix"]
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                text = gr.TextArea(
                    label="输入文本内容",
                    placeholder="""
                    如果你选择语言为\'mix\'，必须按照格式输入，否则报错:
                        格式举例(zh是中文，jp是日语，不区分大小写；说话人举例:gongzi):
                         [说话人1]<zh>你好，こんにちは！ <jp>こんにちは，世界。
                         [说话人2]<zh>你好吗？<jp>元気ですか？
                         [说话人3]<zh>谢谢。<jp>どういたしまして。
                         ...
                    另外，所有的语言选项都可以用'|'分割长段实现分句生成。
                    """
                )
                speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="选择说话人"
                )
                sdp_ratio = gr.Slider(
                    minimum=0, maximum=1, value=0.2, step=0.1, label="SDP/DP混合比"
                )
                noise_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=0.2, step=0.1, label="感情"
                )
                noise_scale_w = gr.Slider(
                    minimum=0.1, maximum=2, value=0.9, step=0.1, label="音素长度"
                )
                length_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=0.8, step=0.1, label="语速"
                )
                language = gr.Dropdown(
                    choices=languages, value=languages[0], label="选择语言(新增mix混合选项)"
                )
                btn = gr.Button("生成音频！", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="状态信息")
                audio_output = gr.Audio(label="输出音频")
                explain_image = gr.Image(label="参数解释信息",
                                         show_label=True,
                                         show_share_button=False,
                                         show_download_button=False,
                                         value=os.path.abspath("./img/参数说明.png"))
        btn.click(
            tts_fn,
            inputs=[
                text,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
            ],
            outputs=[text_output, audio_output],
        )

    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=args.share, server_port=7860)
