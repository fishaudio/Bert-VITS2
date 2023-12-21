# flake8: noqa: E402
import os
import logging
import re_matching
from tools.sentence import split_by_language

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
from infer import infer, latest_version, get_net_g, infer_multilang
import gradio as gr
import webbrowser
import numpy as np
from config import config
from tools.translate import translate
import librosa

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
    reference_audio,
    emotion,
    style_text,
    style_weight,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = idx != 0
            skip_end = idx != len(slices) - 1
            audio = infer(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
                style_text=style_text,
                style_weight=style_weight,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
    return audio_list


def generate_audio_multilang(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    # silence = np.zeros(hps.data.sampling_rate // 2, dtype=np.int16)
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = idx != 0
            skip_end = idx != len(slices) - 1
            audio = infer_multilang(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language[idx],
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
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
    reference_audio,
    emotion,
    style_text,
    style_weight,
):
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    text = text.replace("|", "")
    para_list = re_matching.cut_para(text)
    para_list = [p for p in para_list if p != ""]
    audio_list = []
    for p in para_list:
        if not cut_by_sent:
            audio_list += process_text(
                p,
                speaker,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                language,
                reference_audio,
                emotion,
                style_text,
                style_weight,
            )
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            audio_list.append(silence)
        else:
            audio_list_sent = []
            sent_list = re_matching.cut_sent(p)
            sent_list = [s for s in sent_list if s != ""]
            for s in sent_list:
                audio_list_sent += process_text(
                    s,
                    speaker,
                    sdp_ratio,
                    noise_scale,
                    noise_scale_w,
                    length_scale,
                    language,
                    reference_audio,
                    emotion,
                    style_text,
                    style_weight,
                )
                silence = np.zeros((int)(44100 * interval_between_sent))
                audio_list_sent.append(silence)
            if (interval_between_para - interval_between_sent) > 0:
                silence = np.zeros(
                    (int)(44100 * (interval_between_para - interval_between_sent))
                )
                audio_list_sent.append(silence)
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(
                np.concatenate(audio_list_sent)
            )  # 对完整句子做音量归一
            audio_list.append(audio16bit)
    audio_concat = np.concatenate(audio_list)
    return ("Success", (hps.data.sampling_rate, audio_concat))


def process_mix(slice):
    _speaker = slice.pop()
    _text, _lang = [], []
    for lang, content in slice:
        content = content.split("|")
        content = [part for part in content if part != ""]
        if len(content) == 0:
            continue
        if len(_text) == 0:
            _text = [[part] for part in content]
            _lang = [[lang] for part in content]
        else:
            _text[-1].append(content[0])
            _lang[-1].append(lang)
            if len(content) > 1:
                _text += [[part] for part in content[1:]]
                _lang += [[lang] for part in content[1:]]
    return _text, _lang, _speaker


def process_auto(text):
    _text, _lang = [], []
    for slice in text.split("|"):
        if slice == "":
            continue
        temp_text, temp_lang = [], []
        sentences_list = split_by_language(slice, target_languages=["zh", "ja", "en"])
        for sentence, lang in sentences_list:
            if sentence == "":
                continue
            temp_text.append(sentence)
            if lang == "ja":
                lang = "jp"
            temp_lang.append(lang.upper())
        _text.append(temp_text)
        _lang.append(temp_lang)
    return _text, _lang


def process_text(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
    style_text=None,
    style_weight=0,
):
    audio_list = []
    if language == "mix":
        bool_valid, str_valid = re_matching.validate_text(text)
        if not bool_valid:
            return str_valid, (
                hps.data.sampling_rate,
                np.concatenate([np.zeros(hps.data.sampling_rate // 2)]),
            )
        for slice in re_matching.text_matching(text):
            _text, _lang, _speaker = process_mix(slice)
            if _speaker is None:
                continue
            print(f"Text: {_text}\nLang: {_lang}")
            audio_list.extend(
                generate_audio_multilang(
                    _text,
                    sdp_ratio,
                    noise_scale,
                    noise_scale_w,
                    length_scale,
                    _speaker,
                    _lang,
                    reference_audio,
                    emotion,
                )
            )
    elif language.lower() == "auto":
        _text, _lang = process_auto(text)
        print(f"Text: {_text}\nLang: {_lang}")
        audio_list.extend(
            generate_audio_multilang(
                _text,
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                _lang,
                reference_audio,
                emotion,
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
                reference_audio,
                emotion,
                style_text,
                style_weight,
            )
        )
    return audio_list


def tts_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
    prompt_mode,
    style_text=None,
    style_weight=0,
):
    if style_text == "":
        style_text = None
    if prompt_mode == "Audio prompt":
        if reference_audio == None:
            return ("Invalid audio prompt", None)
        else:
            reference_audio = load_audio(reference_audio)[1]
    else:
        reference_audio = None

    audio_list = process_text(
        text,
        speaker,
        sdp_ratio,
        noise_scale,
        noise_scale_w,
        length_scale,
        language,
        reference_audio,
        emotion,
        style_text,
        style_weight,
    )

    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)


def format_utils(text, speaker):
    _text, _lang = process_auto(text)
    res = f"[{speaker}]"
    for lang_s, content_s in zip(_lang, _text):
        for lang, content in zip(lang_s, content_s):
            res += f"<{lang.lower()}>{content}"
        res += "|"
    return "mix", res[:-1]


def load_audio(path):
    audio, sr = librosa.load(path, 48000)
    # audio = librosa.resample(audio, 44100, 48000)
    return sr, audio


def gr_util(item):
    if item == "Text prompt":
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }
    else:
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }


if __name__ == "__main__":
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    # 若config.json中未指定版本则默认为最新版本
    version = hps.version if hasattr(hps, "version") else latest_version
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP", "EN", "mix", "auto"]
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
                formatter = gr.Button("检测语言，并整理为 MIX 格式", variant="primary")
                speaker = gr.Dropdown(
                    choices=speakers, value=speakers[0], label="Speaker"
                )
                _ = gr.Markdown(
                    value="提示模式（Prompt mode）：可选文字提示或音频提示，用于生成文字或音频指定风格的声音。\n",
                    visible=False,
                )
                prompt_mode = gr.Radio(
                    ["Text prompt", "Audio prompt"],
                    label="Prompt Mode",
                    value="Text prompt",
                    visible=False,
                )
                text_prompt = gr.Textbox(
                    label="Text prompt",
                    placeholder="用文字描述生成风格。如：Happy",
                    value="Happy",
                    visible=False,
                )
                audio_prompt = gr.Audio(
                    label="Audio prompt", type="filepath", visible=False
                )
                sdp_ratio = gr.Slider(
                    minimum=0, maximum=1, value=0.5, step=0.1, label="SDP Ratio"
                )
                noise_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=0.6, step=0.1, label="Noise"
                )
                noise_scale_w = gr.Slider(
                    minimum=0.1, maximum=2, value=0.9, step=0.1, label="Noise_W"
                )
                length_scale = gr.Slider(
                    minimum=0.1, maximum=2, value=1.0, step=0.1, label="Length"
                )
                language = gr.Dropdown(
                    choices=languages, value=languages[0], label="Language"
                )
                btn = gr.Button("生成音频！", variant="primary")
            with gr.Column():
                with gr.Accordion("融合文本语义", open=False):
                    gr.Markdown(
                        value="使用辅助文本的语意来辅助生成对话（语言保持与主文本相同）\n\n"
                        "**注意**：不要使用**指令式文本**（如：开心），要使用**带有强烈情感的文本**（如：我好快乐！！！）\n\n"
                        "效果较不明确，留空即为不使用该功能"
                    )
                    style_text = gr.Textbox(label="辅助文本")
                    style_weight = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.7,
                        step=0.1,
                        label="Weight",
                        info="主文本和辅助文本的bert混合比率，0表示仅主文本，1表示仅辅助文本",
                    )
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
                audio_prompt,
                text_prompt,
                prompt_mode,
                style_text,
                style_weight,
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
                audio_prompt,
                text_prompt,
                style_text,
                style_weight,
            ],
            outputs=[text_output, audio_output],
        )

        prompt_mode.change(
            lambda x: gr_util(x),
            inputs=[prompt_mode],
            outputs=[text_prompt, audio_prompt],
        )

        audio_prompt.upload(
            lambda x: load_audio(x),
            inputs=[audio_prompt],
            outputs=[audio_prompt],
        )

        formatter.click(
            format_utils,
            inputs=[text, speaker],
            outputs=[language, text],
        )

    print("推理页面已开启!")
    webbrowser.open(f"http://127.0.0.1:{config.webui_config.port}")
    app.launch(share=config.webui_config.share, server_port=config.webui_config.port)
