# flake8: noqa: E402

import sys, os
import logging

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
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from infer_utils import infer
import gradio as gr
import webbrowser
import numpy as np
import jieba
import re
import MeCab

net_g = None

if sys.platform == "darwin" and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = "cuda"

MAX_LENGTH = 512

# 静音时长映射
PUNCTUATIONS_SILENCE = {
    ",": 0.2, "，": 0.2,
    ".": 0.5, "。": 0.5,
    "?": 0.5, "？": 0.5,
    "!": 0.5, "！": 0.5,
    # 可以继续添加其他标点符号
}
pattern = re.compile(r'([^,，.。?？!！]*[,，.。?？!！]?)')
mecab = MeCab.Tagger("-Owakati")

def split_by_tokenizer(text, language):
    """
    根据语言对过长的文本进行分词处理，以确保片段不超过MAX_LENGTH。
    :param text: 待分词的文本。
    :param language: 指定文本的语言。
    :return: 返回一个分词后的文本片段列表。
    """

    # 根据不同的语言进行分词
    if language == "ZH":
        tokens = jieba.lcut(text)  # 中文使用jieba分词
    elif language == "JP":
        tokens = mecab.parse(text).strip().split()
    else:
        tokens = text.split()  # 英文简单用空格分词

    slices = []
    temp_slice = []

    for token in tokens:
        if len(''.join(temp_slice + [token])) > MAX_LENGTH:
            slices.append(''.join(temp_slice))
            temp_slice = []
        temp_slice.append(token)

    # 添加剩余的片段
    if temp_slice:
        slices.append(''.join(temp_slice))

    return slices


def split_text(text, language):
    """
    根据标点符号对文本进行分割，同时确保每个片段接近MAX_LENGTH。
    :param text: 待分割的文本。
    :param language: 指定文本的语言。
    :return: 返回一个包含文本片段和对应静音时长的列表。
    """

    # 使用正则表达式按照标点符号对文本进行初步分割
    prelim_slices = pattern.findall(text)

    current_slice = ""
    slices = []

    for slice in prelim_slices:
        # 检查添加新片段是否会超出上限
        if len(current_slice + slice) <= MAX_LENGTH:
            current_slice += slice
        else:
            slices.append(current_slice)
            current_slice = slice

            # 如果当前片段仍超出上限，则使用分词器进行切分
            if len(current_slice) > MAX_LENGTH:
                extended_slices = split_by_tokenizer(current_slice, language)
                slices.extend(extended_slices[:-1])  # 将除了最后一个部分的所有部分添加到slices中
                current_slice = extended_slices[-1]  # 将最后一个部分设置为当前片段

    if current_slice:
        if len(current_slice) > MAX_LENGTH:
            # 如果剩余片段过长，则进一步使用分词器进行分割
            slices.extend(split_by_tokenizer(current_slice, language))
        else:
            slices.append(current_slice)
    slices = [slice.strip() for slice in slices if slice.strip()]
    return slices

def tts_fn(
    text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language
):
    slices = split_text(text, language)
    #logger.info(slices)
    audio_list = []
    with torch.no_grad():
        for slice in slices:
            audio = infer(
                slice,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
            )
            audio_list.append(audio)

            # 根据最后标点符号添加静音
            silence_duration = PUNCTUATIONS_SILENCE.get(slice[-1], 0.1)
            silence = np.zeros(hps.data.sampling_rate * silence_duration)
            audio_list.append(silence)  # 将静音添加到列表中
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
    languages = ["ZH", "JP"]
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                text = gr.TextArea(
                    label="Text",
                    placeholder="Input Text Here",
                    value="吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。",
                )
                speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="Speaker"
                )
                sdp_ratio = gr.Slider(
                    minimum=0, maximum=1, value=0.2, step=0.1, label="SDP Ratio"
                )
                noise_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=0.6, step=0.1, label="Noise Scale"
                )
                noise_scale_w = gr.Slider(
                    minimum=0.1, maximum=2, value=0.8, step=0.1, label="Noise Scale W"
                )
                length_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=1, step=0.1, label="Length Scale"
                )
                language = gr.Dropdown(
                    choices=languages, value=languages[0], label="Language"
                )
                btn = gr.Button("Generate!", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio")

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
    app.launch(share=args.share)
