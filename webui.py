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
    bert = get_bert(norm_text, word2ph, language_str)

    assert bert.shape[-1] == len(phone)

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)

    return bert, phone, tone, language

def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid):
    bert, phones, tones, lang_ids = get_text(text, "ZH", hps,)
    with torch.no_grad():
        x_tst=phones.to(device).unsqueeze(0)
        tones=tones.to(device).unsqueeze(0)
        lang_ids=lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, speakers, tones, lang_ids,bert, sdp_ratio=sdp_ratio
                           , noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
        return audio

def tts_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale):
    with torch.no_grad():
        audio = infer(text, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale, sid=speaker)
    return "Success", (hps.data.sampling_rate, audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="./logs/as/G_8000.pth", help="path of your model")
    parser.add_argument("-c", "--config", default="./configs/config.json", help="path of your config file")
    parser.add_argument("--share", default=False, help="make link public")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model, net_g, None,skip_optimizer=True)

    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                text = gr.TextArea(label="Text", placeholder="Input Text Here",
                                      value="吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。")
                speaker = gr.Dropdown(choices=speakers, value=speakers[0], label='Speaker')
                sdp_ratio = gr.Slider(minimum=0.1, maximum=2, value=0.2, step=0.1, label='SDP Ratio')
                noise_scale = gr.Slider(minimum=0.1, maximum=2, value=0.5, step=0.1, label='Noise Scale')
                noise_scale_w = gr.Slider(minimum=0.1, maximum=2, value=0.6, step=0.1, label='Noise Scale W')
                length_scale = gr.Slider(minimum=0.1, maximum=2, value=1.2, step=0.1, label='Length Scale')
                btn = gr.Button("Generate!", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio")

        btn.click(tts_fn,
                inputs=[text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale],
                outputs=[text_output, audio_output])
    
    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=args.share)
