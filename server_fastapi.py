"""
api服务 多版本多模型 fastapi实现
"""
import argparse
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response, FileResponse
from io import BytesIO
from scipy.io import wavfile
import uvicorn
import torch
import psutil
import GPUtil
from typing import Dict, Optional, List, Union
import os, sys
from tools.log import logger
from urllib.parse import unquote
from config import config
from app import (
    Model,
    ModelHolder,
    languages,
    DEFAULT_SDP_RATIO,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE_WEIGHT,
    DEFAULT_EMOTION_WEIGHT,
)
from webui_style_vectors import DEFAULT_EMOTION

ln = config.server_config.language
available_languages = languages + ["mix", "auto"]

def load_models(model_holder: ModelHolder):
    model_holder.models = []
    for model_name, model_paths in model_holder.model_files_dict.items():
        model = Model(
            model_path=model_paths[0],
            config_path=os.path.join(model_holder.root_dir, model_name, "config.json"),
            style_vec_path=os.path.join(model_holder.root_dir, model_name, "style_vectors.npy"),
            device=model_holder.device,
        )
        model.load_net_g()
        model_holder.models.append(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=config.out_dir
    )
    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = args.dir
    model_holder = ModelHolder(model_dir, device)
    if len(model_holder.model_names) == 0:
        logger.error(f"Models not found in {model_dir}.")
        sys.exit(1)

    logger.info('Loading models...')
    load_models(model_holder)
    limit = config.server_config.limit
    app = FastAPI()
    app.logger = logger

    async def _voice(
        text: str,
        model_id: int = 0,
        speaker_name: str = None,
        speaker_id: int = 0,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noisew: float = DEFAULT_NOISEW,
        length: float = DEFAULT_LENGTH,
        language: str = ln,
        auto_split: bool = DEFAULT_LINE_SPLIT,
        split_interval: float = DEFAULT_SPLIT_INTERVAL,
        style_text: Optional[str] = None,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        emotion: Optional[Union[int, str]] = DEFAULT_EMOTION,
        emotion_weight: float = DEFAULT_EMOTION_WEIGHT,
        reference_audio_path: str = None,
    ) -> Union[Response, Dict[str, any]]:
        if model_id >= len(model_holder.models):
            return {"status": 10, "detail": f"model_id={model_id} not found"}
        elif len(text) > limit:
            return {"status": 9, "detail": f"too long text: over {limit}"}

        model = model_holder.models[model_id]
        if speaker_name is None:
            if speaker_id is None:
                return {"status": 11, "detail": "Required speaker_name or speaker_id"}
            if speaker_id not in model.id2spk.keys():
                return {"status": 12, "detail": f"peaker_id={speaker_id} not found"}
        else:
            if speaker_name not in model.spk2id.keys():
                return {"status": 13, "detail": f"speaker_name={speaker_name} not found"}
            speaker_id = model.spk2id[speaker_name]
        if emotion not in model.style2id.keys():
            return {"status": 14, "detail": f"emotion={speaker_name} not found"}
        if language not in available_languages:
            language = ln
        sr, audio = model.infer(
            text=text,
            language=language,
            sid=speaker_id,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noisew=noisew,
            length=length,
            line_split=auto_split,
            split_interval=split_interval,
            style_text=style_text,
            style_weight=style_weight,
            use_style_text=bool(style_text),
            style=emotion,
            emotion_weight=emotion_weight,
        )
        with BytesIO() as wavContent:
            wavfile.write(
                wavContent, sr, audio
            )
            response = Response(content=wavContent.getvalue(), media_type="audio/wav")
            return response

    @app.get("/voice")
    async def voice(
        request: Request,
        text: str = Query(..., description="セリフ"),
        model_id: int = Query(0, description="モデルID。`GET /models/info`のkeyの値を指定ください。"),
        speaker_name: str = Query(
            None, description="話者名(speaker_idより優先)。esd.listの2列目記載の文字列を指定。"
        ),
        speaker_id: int = Query(
            0, description="話者ID。model_assets.[model].config.json内のspk2idを確認。"),
        sdp_ratio: float = Query(DEFAULT_SDP_RATIO, description="SDP(Stochastic Duration Predictor)/DP混合比。比率が高くなるほど、トーンのばらつきが大きくなる。"),
        noise: float = Query(DEFAULT_NOISE, description="サンプルノイズの割合。大きくするほどランダム性が高まる"),
        noisew: float = Query(DEFAULT_NOISEW, description="SDPノイズ。大きくするほど発音の間隔にばらつきが出やすくなる。"),
        length: float = Query(DEFAULT_LENGTH, description="話速。基準は1で大きくするほど音声は長くなり読み上げが遅まる。"),
        language: str = Query(ln, description=f"{'/'.join(available_languages)}のいずれか。"),
        auto_split: bool = Query(True, description="改行で分けて生成"),
        split_interval: float = Query(DEFAULT_SPLIT_INTERVAL, description="分けた場合に挟む無音の長さ（秒）"),
        style_text: Optional[str] = Query(
            None, description="このテキストの読み上げと似た声音・感情になりやすくなる。ただし抑揚やテンポ等が犠牲になる傾向がある。"
        ),
        style_weight: float = Query(DEFAULT_STYLE_WEIGHT, description="style_textの強さ"),
        emotion: Optional[Union[int, str]] = Query(DEFAULT_EMOTION, description="スタイル"),
        emotion_weight: float = Query(DEFAULT_EMOTION_WEIGHT, description="emotionの強さ"),
        reference_audio_path: Optional[str] = Query(None, description="emotionを音声ファイルで行う"),
    ):
        """Infer text to speech(テキストから感情付き音声を生成する)"""
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}"
        )
        return await _voice(
            text=text,
            model_id=model_id,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noisew=noisew,
            length=length,
            language=language,
            auto_split=auto_split,
            split_interval=split_interval,
            style_text=style_text,
            style_weight=style_weight,
            emotion=emotion,
            emotion_weight=emotion_weight,
            reference_audio_path=reference_audio_path,
        )

    @app.get("/models/info")
    def get_loaded_models_info(request: Request):
        """ロードされたモデル情報の取得"""

        result: Dict[str, Dict] = dict()
        for model_id, model in enumerate(model_holder.models):
            result[str(model_id)] = {
                "config_path": model.config_path,
                "model_path": model.model_path,
                "device": model.device,
                "spk2id": model.spk2id,
                "id2spk": model.id2spk,
                "style2id": model.style2id,
            }
        return result

    @app.get("/models/refresh")
    def refresh():
        """モデルをパスに追加/削除した際などに読み込ませる"""
        model_holder.refresh()
        load_models(model_holder)
        return {}

    @app.get("/status")
    def get_status():
        """実行環境のステータスを取得"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_total = memory_info.total
        memory_available = memory_info.available
        memory_used = memory_info.used
        memory_percent = memory_info.percent
        gpuInfo = []
        devices = ["cpu"]
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpuInfo.append(
                {
                    "gpu_id": gpu.id,
                    "gpu_load": gpu.load,
                    "gpu_memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                    },
                }
            )
        return {
            "devices": devices,
            "cpu_percent": cpu_percent,
            "memory_total": memory_total,
            "memory_available": memory_available,
            "memory_used": memory_used,
            "memory_percent": memory_percent,
            "gpu": gpuInfo,
        }

    @app.get("/tools/get_audio")
    def get_audio(request: Request, path: str = Query(..., description="local wav path")):
        """wavデータを取得する"""
        logger.info(
            f"{request.client.host}:{request.client.port}/tools/get_audio  { unquote(str(request.query_params) )}"
        )
        if not os.path.isfile(path):
            logger.error(f"/tools/get_audio 获取音频错误：指定音频{path}不存在")
            return {"status": 18, "detail": "指定音频不存在"}
        if not path.lower().endswith(".wav"):
            logger.error(f"/tools/get_audio 获取音频错误：音频{path}非wav文件")
            return {"status": 19, "detail": "非wav格式文件"}
        return FileResponse(path=path)


    logger.info(f"server listen: http://127.0.0.1:{config.server_config.port}")
    logger.info(f"API docs: http://127.0.0.1:{config.server_config.port}/docs")
    uvicorn.run(
        app, port=config.server_config.port, host="0.0.0.0", log_level="warning"
    )
