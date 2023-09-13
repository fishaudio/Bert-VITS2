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
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence
from text.parser import parse_text_to_segments, segments_g2p, get_bert_alignment
import gradio as gr
import webbrowser

net_g = None

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else (
        "mps"
        if sys.platform == "darwin" and torch.backends.mps.is_available()
        else "cpu"
    )
)


def get_text(text, hps):
    segments = parse_text_to_segments(text)
    words, phones, tones, word2ph, languages = segments_g2p(segments)
    complex_tokens = get_bert_alignment(words, phones, word2ph)
    token_ids = [i["token_id"] for i in complex_tokens]
    offsets = [i["offset"] for i in complex_tokens]

    # Convert offsets to mapping
    phones2tokens = [0] * len(phones)  # All use CLS by default
    for i in range(len(offsets)):
        if offsets[i] is None:
            continue

        start, end = offsets[i]
        for j in range(start, end):
            phones2tokens[j] = i

    phones, tones, languages = cleaned_text_to_sequence(phones, tones, languages)

    if hps.data.add_blank:
        phones = commons.intersperse(phones, 0)
        tones = commons.intersperse(tones, 0)
        languages = commons.intersperse(languages, 0)
        phones2tokens = commons.intersperse(phones2tokens, 0)

    assert len(phones) == len(phones2tokens) == len(tones) == len(languages)

    return dict(
        phones=torch.LongTensor(phones),
        tones=torch.LongTensor(tones),
        languages=torch.LongTensor(languages),
        phones2tokens=torch.LongTensor(phones2tokens),
        token_ids=torch.LongTensor(token_ids),
        token_attention_masks=torch.ones(len(token_ids)),
    )


def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid):
    global net_g

    data = get_text(text, hps)
    for k in list(data.keys()):
        data[k] = data[k][None].to(device)

    with torch.no_grad():
        x_tst_lengths = torch.LongTensor([data["phones"].size(1)]).to(device)
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)

        audio = (
            net_g.infer(
                data["phones"],
                x_tst_lengths,
                speakers,
                data["tones"],
                data["languages"],
                data["token_ids"],
                data["token_attention_masks"],
                data["phones2tokens"],
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )

        return audio


def tts_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale):
    with torch.no_grad():
        audio = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
            sid=speaker,
        )
        torch.cuda.empty_cache()

    return "Success", (hps.data.sampling_rate, audio)


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
            ],
            outputs=[text_output, audio_output],
        )

    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=args.share)
