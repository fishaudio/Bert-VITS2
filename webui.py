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

net_g = None

if sys.platform == "darwin" and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = "cuda"


def tts_fn(
    text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language
):
    slices = text.split("|")
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
            silence = np.zeros(hps.data.sampling_rate)  # 生成1秒的静音
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
