from flask import Flask, request, Response
from io import BytesIO
import torch
from av import open as avopen

import utils
from models import SynthesizerTrn
from text.symbols import symbols
from infer_utils import infer
from scipy.io import wavfile

# Flask Init
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


def replace_punctuation(text, i=2):
    punctuation = "，。？！"
    for char in punctuation:
        text = text.replace(char, char * i)
    return text


def wav2(i, o, format):
    inp = avopen(i, "rb")
    out = avopen(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


# Load Generator
hps = utils.get_hparams_from_file("./configs/config.json")

dev = "cuda"
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model,
).to(dev)
_ = net_g.eval()

_ = utils.load_checkpoint("logs/G_649000.pth", net_g, None, skip_optimizer=True)


@app.route("/")
def main():
    try:
        speaker = request.args.get("speaker")
        text = request.args.get("text").replace("/n", "")
        sdp_ratio = float(request.args.get("sdp_ratio", 0.2))
        noise = float(request.args.get("noise", 0.5))
        noisew = float(request.args.get("noisew", 0.6))
        length = float(request.args.get("length", 1.2))
        language = request.args.get("language")
        if length >= 2:
            return "Too big length"
        if len(text) >= 250:
            return "Too long text"
        fmt = request.args.get("format", "wav")
        if None in (speaker, text):
            return "Missing Parameter"
        if fmt not in ("mp3", "wav", "ogg"):
            return "Invalid Format"
        if language not in ("JP", "ZH"):
            return "Invalid language"
    except:
        return "Invalid Parameter"

    with torch.no_grad():
        audio = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise,
            noise_scale_w=noisew,
            length_scale=length,
            sid=speaker,
            language=language,
            hps=hps,
            net_g=net_g,
            device=dev,
        )

    with BytesIO() as wav:
        wavfile.write(wav, hps.data.sampling_rate, audio)
        torch.cuda.empty_cache()
        if fmt == "wav":
            return Response(wav.getvalue(), mimetype="audio/wav")
        wav.seek(0, 0)
        with BytesIO() as ofp:
            wav2(wav, ofp, fmt)
            return Response(
                ofp.getvalue(), mimetype="audio/mpeg" if fmt == "mp3" else "audio/ogg"
            )
