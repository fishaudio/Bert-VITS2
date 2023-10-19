# flake8: noqa: E402
import os
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
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from infer import infer
import gradio as gr
import webbrowser
import numpy as np
from config import config
from tools.translate import translate
from oldVersion.V111.models import SynthesizerTrn as V111SynthesizerTrn
from oldVersion.V111.text import symbols as V111symbols
from oldVersion.V110.models import SynthesizerTrn as V110SynthesizerTrn
from oldVersion.V110.text import symbols as V110symbols
from oldVersion.V101.models import SynthesizerTrn as V101SynthesizerTrn
from oldVersion.V101.text import symbols as V101symbols

net_g = None

device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def generate_audio(
    slices, sdp_ratio, noise_scale, noise_scale_w, length_scale, speaker, language
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
    text: str, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language
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


if __name__ == "__main__":
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    version = hps.version if hasattr(hps, "version") else "1.1.1-dev"
    SynthesizerTrnMap = {
        "1.1.1": V111SynthesizerTrn,
        "1.1": V110SynthesizerTrn,
        "1.1.0": V110SynthesizerTrn,
        "1.0.1": V101SynthesizerTrn,
        "1.0": V101SynthesizerTrn,
        "1.0.0": V101SynthesizerTrn,
    }
    symbolsMap = {
        "1.1.1": V111symbols,
        "1.1": V110symbols,
        "1.1.0": V110symbols,
        "1.0.1": V101symbols,
        "1.0": V101symbols,
        "1.0.0": V101symbols,
    }
    if version != "1.1.1-dev":
        net_g = SynthesizerTrnMap[version](
            len(symbolsMap[version]),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)
    else:
        # 当前版本模型 net_g
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)

    _ = net_g.eval()
    _ = utils.load_checkpoint(
        config.webui_config.model, net_g, None, skip_optimizer=True
    )

    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP", "EN","mix"]
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
                slicer = gr.Button("快速切分", variant="primary")
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
                        slicer = gr.Button("切分生成", variant="primary")
                text_output = gr.Textbox(label="状态信息")
                audio_output = gr.Audio(label="输出音频")
                explain_image = gr.Image(
                    label="参数解释信息",
                    show_label=True,
                    show_share_button=False,
                    show_download_button=False,
                    value=os.path.abspath("./img/参数说明.png"),
                )
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
    print("推理页面已开启!")
    webbrowser.open(f"http://127.0.0.1:{config.webui_config.port}")
    app.launch(share=config.webui_config.share, server_port=config.webui_config.port)
