
import streamlit as st
import numpy as np
import soundfile as sf
import sys, os
import logging
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
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
import gradio as gr
import numpy as np
import soundfile as sf
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
    
       # 保存音频数据到本地文件
    sf.write(out_wav, audio_concat, samplerate=hps.data.sampling_rate)  # 请替换YOUR_SAMPLERATE为实际的采样率

    return "Success", (hps.data.sampling_rate,audio_concat )




def prepare(hps):
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
    return net_g
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(    
    "-m", "--model_path", default="./logs/c1", help="path of your model"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="",
        help="path of your config file",
    )
    parser.add_argument(
        "--out_wav", default="out/output_audio.wav", help="make link public", action="store_true"
    )
    
    st.set_page_config(
        page_title="鸿蒙Vits",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': 
                """# 这是一个*非常*酷的应用！
                    - 目前已经可以进行多角色推理
                """  # 使用Markdown格式的字符串
        }
    )
    
    st.title("欢迎来到奇思妙想妙妙屋")
    args = parser.parse_args()
    out_wav=args.out_wav    
    model_path=args.model_path
    if 'config_file' not in st.session_state:
            st.session_state.config_file= ""
    if 'model_file' not in st.session_state:
            st.session_state.model_file= "" 
    if 'hps' not in st.session_state:
            st.session_state.hps= ""
    if 'model_path' not in st.session_state:
            st.session_state.model_path= ""
    if 'net_g' not in st.session_state:
            st.session_state.net_g= ""            
            
            
    if args.config =="":
        config_path="./configs"
        flist=[it[:-5].split('_')[-1] for it in os.listdir(config_path)]
        config_file=st.selectbox("选择配置",flist)
        if config_file != st.session_state.config_file:
            st.session_state.config_file=config_file
            st.session_state.model_file= "" 
            hps = utils.get_hparams_from_file(os.path.join(config_path,"config_"+config_file+".json")) 
            st.session_state.hps=hps 
            model_path=os.path.join("./logs",st.session_state.config_file)
            st.session_state.model_path=model_path 
            logger.info("更换配置***重新加载生成器.......................")
            # print("更换生成器")
            net_g=prepare(hps)
            st.session_state.net_g=net_g 
            _ = st.session_state.net_g.eval()
    else: 
        hps = utils.get_hparams_from_file(args.config)
        net_g=prepare(hps) 
        _ = net_g.eval() 
    
    
    model_file = st.selectbox("选择模型",sorted([it for it in os.listdir(st.session_state.model_path) if it.endswith('.pth') and it[0]=="G"])[1:] )
    if model_file != st.session_state.model_file:
        logger.info("更换模型***重新加载模型.......................") 
        st.session_state.model_file = model_file
        # print(os.path.join(model_path,model_file))
        _ = utils.load_checkpoint(os.path.join(st.session_state.model_path,model_file), st.session_state.net_g, None, skip_optimizer=True)

    speaker_ids = st.session_state.hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP"]
    

    text = st.text_area(
        "Text", "欢迎来到奇思妙想妙妙屋~"
    )
    speaker = st.selectbox("Speaker", speakers)
    sdp_ratio = st.slider("SDP Ratio", 0.0, 1.0, 0.2, 0.01)
    noise_scale = st.slider("Noise Scale", 0.1, 2.0, 0.6, 0.01)
    noise_scale_w = st.slider("Noise Scale W", 0.1, 2.0, 0.8, 0.01)
    length_scale = st.slider("Length Scale", 0.1, 2.0, 1.0, 0.01)
    language = st.selectbox("Language", languages)

    message=None
    if st.button("Generate!"):
        message, audio_data = tts_fn(
            text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language
        )
        
        # 显示消息
    st.text("Message:")
    st.write(message)

    # 播放音频
    st.text("Generated Audio:")
    st.audio(out_wav, format="audio/wav")

