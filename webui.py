# flake8: noqa: E402
import os
import logging
import re_matching
import argparse
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

import torch
import utils
from infer import infer, latest_version, get_net_g
import gradio as gr
import webbrowser
import numpy as np
from config import config
from tools.translate import translate

net_g = None

device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
):
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
                hps=hps,
                net_g=net_g,
                device=device,
            )
            audio_list.append(audio)
            audio_list.append(silence)  # 将静音添加到列表中
    return audio_list


def tts_split(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    cut_by_sent,
    interval_between_para,
    interval_between_sent,
):
    if language == "mix":
        return ("invalid", None)
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    para_list = re_matching.cut_para(text)
    audio_list = []
    if not cut_by_sent:
        for p in para_list:
            audio = infer(
                p,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
            )
            audio_list.append(audio)
            silence = np.zeros((int)(44100 * interval_between_para))
            audio_list.append(silence)
    else:
        for p in para_list:
            sent_list = re_matching.cut_sent(p)
            for s in sent_list:
                audio = infer(
                    s,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                    sid=speaker,
                    language=language,
                    hps=hps,
                    net_g=net_g,
                    device=device,
                )
                audio_list.append(audio)
                silence = np.zeros((int)(44100 * interval_between_sent))
                audio_list.append(silence)
            if (interval_between_para - interval_between_sent) > 0:
                silence = np.zeros(
                    (int)(44100 * (interval_between_para - interval_between_sent))
                )
                audio_list.append(silence)
    audio_concat = np.concatenate(audio_list)
    return ("Success", (44100, audio_concat))


def tts_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
):
    audio_list = []
    if language == "mix":
        bool_valid, str_valid = re_matching.validate_text(text)
        if not bool_valid:
            return str_valid, (
                hps.data.sampling_rate,
                np.concatenate([np.zeros(hps.data.sampling_rate // 2)]),
            )
        result = re_matching.text_matching(text)
        for one in result:
            _speaker = one.pop()
            for lang, content in one:
                audio_list.extend(
                    generate_audio(
                        content.split("|"),
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        _speaker + "_" + lang.lower(),
                        lang,
                    )
                )
    else:
        audio_list.extend(
            generate_audio(
                text.split("|"),
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                language,
            )
        )

    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)

def file_infer(file,speaker,sdp_ratio,noise_scale,noise_scale_w,length_scale,language,cut_by_sent,interval_between_para,interval_between_sent):
    try:
      with open(file.name, "r", encoding="utf-8") as file:
         text = file.read()
         return tts_split(text,speaker,sdp_ratio,noise_scale,noise_scale_w,length_scale,language,cut_by_sent,interval_between_para,interval_between_sent)
    except Exception as error:
        return error,None

if __name__ == "__main__":
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default=config.webui_config.model, help="path of your model"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=config.webui_config.config_path,
        help="path of your config file",
    )
    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config)
    # 若config.json中未指定版本则默认为最新版本
    version = hps.version if hasattr(hps, "version") else latest_version
    net_g = get_net_g(
        model_path=args.model, version=version, device=device, hps=hps
    )
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP", "EN", "mix"]
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
                    """,
                )
                trans = gr.Button("中翻日", variant="primary")
                speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="选择说话人"
                )
                sdp_ratio = gr.Slider(
                    minimum=0, maximum=1, value=0.2, step=0.1, label="SDP/DP混合比"
                )
                noise_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=0.6, step=0.1, label="感情"
                )
                noise_scale_w = gr.Slider(
                    minimum=0.1, maximum=2, value=0.8, step=0.1, label="音素长度"
                )
                length_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=1.0, step=0.1, label="语速"
                )
                language = gr.Dropdown(
                    choices=languages, value=languages[0], label="选择语言(新增mix混合选项)"
                )
                btn = gr.Button("生成音频！", variant="primary")
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        interval_between_sent = gr.Slider(
                            minimum=0,
                            maximum=5,
                            value=0.2,
                            step=0.1,
                            label="句间停顿(秒)，勾选按句切分才生效",
                        )
                        interval_between_para = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=1,
                            step=0.1,
                            label="段间停顿(秒)，需要大于句间停顿才有效",
                        )
                        opt_cut_by_sent = gr.Checkbox(
                            label="按句切分    在按段落切分的基础上再按句子切分文本"
                        )
                        input_file = gr.Files(label="上传txt纯文本文件",file_types=['text'],file_count='single')
                        slicer = gr.Button("文本框切分生成", variant="primary")
                        slicer_txt_file = gr.Button("从文件切分生成", variant="primary")
                text_output = gr.Textbox(label="状态信息")
                audio_output = gr.Audio(label="输出音频")
                # explain_image = gr.Image(
                #     label="参数解释信息",
                #     show_label=True,
                #     show_share_button=False,
                #     show_download_button=False,
                #     value=os.path.abspath("./img/参数说明.png"),
                # )
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

        trans.click(
            translate,
            inputs=[text],
            outputs=[text],
        )
        slicer.click(
            tts_split,
            inputs=[
                text,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
                opt_cut_by_sent,
                interval_between_para,
                interval_between_sent,
            ],
            outputs=[text_output, audio_output],
        )
        slicer_txt_file.click(
            file_infer,
            inputs=[
                input_file,#text
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
                opt_cut_by_sent,
                interval_between_para,
                interval_between_sent,
            ],
            outputs=[text_output, audio_output],
        )       

    print("推理页面已开启!")
    webbrowser.open(f"http://127.0.0.1:{config.webui_config.port}")
    app.launch(share=config.webui_config.share, server_port=config.webui_config.port)
