from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from io import BytesIO
import torch
from av import open as avopen
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text
from scipy.io import wavfile
import os
app = FastAPI()


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
    bert = get_bert(norm_text, word2ph, language_str,dev)
    del word2ph
    assert bert.shape[-1] == len(phone), phone

    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JA":
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
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
    # print("---------------") 
    with torch.no_grad():
        x_tst = phones.to(dev).unsqueeze(0)
        tones = tones.to(dev).unsqueeze(0)
        lang_ids = lang_ids.to(dev).unsqueeze(0)
        bert = bert.to(dev).unsqueeze(0)
        ja_bert = ja_bert.to(dev).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(dev)
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(dev)
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
        return audio


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
@app.get("/get_modelstatus")
def get_modelstatus():
    return model 
@app.get("/get_config")
def get_config(): 
    flist=[it[:-5].split('_')[-1] for it in os.listdir(config_path)]   
    return flist
@app.get("/get_model")
def get_model(config_name): 
    return sorted([it for it in os.listdir(os.path.join("logs",config_name)) if it.endswith('.pth') and it[0]=="G"])[1:] 
@app.get("/get_speaker")
def get_speaker():
    return  list(hps.data.spk2id.keys())
@app.get("/switch_model")
def switch_model(config_name,model_name):
    print("开始切换模型")
    config=os.path.join(config_path,"config_"+config_name+".json")
    global model
    model=os.path.join("./logs",config_name,model_name)
    # print(model)
    print("************{},{}已加载**************".format(config,model))
    global hps,net_g
    hps = utils.get_hparams_from_file(config)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(dev)
    _ = net_g.eval()
    _ = utils.load_checkpoint(model, net_g, None, skip_optimizer=True) 
    return "切换模型成功"


def get_query_parameters(
    speaker: str = Query(0, description="Speaker's name"),
    text: str = Query(None, description="Text to be synthesized"),
    sdp_ratio: float = Query(0.2, description="SDP ratio"),
    noise: float = Query(0.5, description="Noise scale"),
    noisew: float = Query(0.6, description="Noise scale for waveform"),
    length: float = Query(1.2, description="Length scale"),
    language: str = Query("ZH", description="Language code"),
    fmt: str = Query("wav", description="Output format")
) -> dict:
    return {
        "speaker": speaker,
        "text": text,
        "sdp_ratio": sdp_ratio,
        "noise": noise,
        "noisew": noisew,
        "length": length,
        "language": language,
        "fmt": fmt
    }

@app.get("/get_audio")
def get_audio(params: dict = Depends(get_query_parameters)):
    # print("坏坏坏")
    speaker=params["speaker"]
    print(speaker)
    text = params["text"].replace("/n", "")
    sdp_ratio = params["sdp_ratio"]
    noise = params["noise"]
    noisew = params["noisew"]
    length = params["length"]
    fmt = params["fmt"]
    language = params["language"]
    # print(params)
    if length >= 2:
        raise HTTPException(status_code=400, detail="Too big length")
    if None in (speaker, text):
        raise HTTPException(status_code=400, detail="Missing Parameter")
    if fmt not in ("mp3", "wav", "ogg"):
        raise HTTPException(status_code=400, detail="Invalid Format")
    if language not in ("JA", "ZH"):
        raise HTTPException(status_code=400, detail="Invalid language")

    with torch.no_grad():
        audio = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise,
            noise_scale_w=noisew,
            length_scale=length,
            sid=speaker,
            language=language
        )
    # wavfile.write('output.wav', hps.data.sampling_rate, audio)
    buffer = BytesIO()
    wavfile.write(buffer, hps.data.sampling_rate, audio)
    # 移动到文件的起始位置，以便从头开始读取
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")
    



@app.get("/")
def main(params: dict = Depends(get_query_parameters)):
    print("好好好")
    global hps,net_g
    hps = utils.get_hparams_from_file("./configs/config_c1.json")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(dev)
    _ = net_g.eval()

    _ = utils.load_checkpoint("./logs/c1/G_84000.pth", net_g, None, skip_optimizer=True)
    # speaker = params["speaker"]
    speaker="巴老师"
    
    text = params["text"].replace("/n", "")
    sdp_ratio = params["sdp_ratio"]
    noise = params["noise"]
    noisew = params["noisew"]
    length = params["length"]
    fmt = params["fmt"]
    language = params["language"]
    
    
    if length >= 2:
        raise HTTPException(status_code=400, detail="Too big length")
    if None in (speaker, text):
        raise HTTPException(status_code=400, detail="Missing Parameter")
    if fmt not in ("mp3", "wav", "ogg"):
        raise HTTPException(status_code=400, detail="Invalid Format")
    if language not in ("JA", "ZH"):
        raise HTTPException(status_code=400, detail="Invalid language")

    with torch.no_grad():
        audio = infer(
            text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise,
            noise_scale_w=noisew,
            length_scale=length,
            sid=speaker,
            language=language
        )
    # wavfile.write('output.wav', hps.data.sampling_rate, audio)
    
    buffer = BytesIO()
    wavfile.write(buffer, hps.data.sampling_rate, audio)
    # 移动到文件的起始位置，以便从头开始读取
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")
    
  

if __name__ == "__main__":
    #配置一些全局变量    
    dev = "cuda" 
    hps=None
    net_g=None
    model=None
    config_path="./configs"
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
