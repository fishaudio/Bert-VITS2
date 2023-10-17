"""
api服务 多版本多模型 fastapi实现
"""
import utils
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from io import BytesIO
from scipy.io import wavfile
import uvicorn
import torch
from typing import Dict, List

from infer import infer


from oldVersion.V111.models import SynthesizerTrn as V111SynthesizerTrn
from oldVersion.V111.text import symbols as V111symbols
from oldVersion.V110.models import SynthesizerTrn as V110SynthesizerTrn
from oldVersion.V110.text import symbols as V110symbols
from oldVersion.V101.models import SynthesizerTrn as V101SynthesizerTrn
from oldVersion.V101.text import symbols as V101symbols

from config import config

net_g_List = []
hps_List = []

# 版本兼容
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

# 模型角色字典
# 使用方法 chr_name = chrsMap[model_id][chr_id]
chrsMap: List[Dict[int, str]] = list()

# 加载模型
models = config.server_config.models
for model in models:
    hps_List.append(utils.get_hparams_from_file(model["config"]))
    # 添加角色字典
    chrsMap.append(dict())
    for name, cid in hps_List[-1].data.spk2id.items():
        chrsMap[-1][cid] = name
    version = hps_List[-1].version if hasattr(hps_List[-1], "version") else "1.1.1-dev"
    device = model["device"]
    net_g_List.append(
        SynthesizerTrnMap[version](
            len(symbolsMap[version]),
            hps_List[-1].data.filter_length // 2 + 1,
            hps_List[-1].train.segment_size // hps_List[-1].data.hop_length,
            n_speakers=hps_List[-1].data.n_speakers,
            **hps_List[-1].model
        ).to(device)
    )
    _ = net_g_List[-1].eval()
    _ = utils.load_checkpoint(model["model"], net_g_List[-1], None, skip_optimizer=True)

app = FastAPI()


@app.get("/voice")
async def voice(
    text: str,
    model_id: int,
    chr_name: str = None,  # chr_name与 chr_id二者选其一
    chr_id: int = None,  # chr_id 既可以输入人名也可以输入id
    sdp_ratio: float = 0.2,
    noise: float = 0.5,
    noisew: float = 0.6,
    length: float = 1.0,
):
    if chr_name is None and chr_id is None:
        return HTTPException(status_code=400, detail="请提供chr_name或chr_id")
    elif chr_name is None:
        chr_name = chrsMap[model_id][chr_id]
    with torch.no_grad():
        audio = infer(
            text=text,
            sdp_ratio=sdp_ratio,
            noise_scale=noise,
            noise_scale_w=noisew,
            length_scale=length,
            sid=chr_name,
            language=models[model_id]["language"],
            hps=hps_List[model_id],
            net_g=net_g_List[model_id],
            device=models[model_id]["device"],
        )
    wavContent = BytesIO()
    wavfile.write(wavContent, hps_List[model_id].data.sampling_rate, audio)
    response = Response(content=wavContent.getvalue(), media_type="audio/wav")
    return response


uvicorn.run(app, port=config.server_config.port)
