"""
API server for TTS
"""
import argparse
import os
import sys
from io import BytesIO
from typing import Dict, Optional, Union
from urllib.parse import unquote

import GPUtil
import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from scipy.io import wavfile

from common.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from common.log import logger
from common.tts_model import Model, ModelHolder
from config import config

ln = config.server_config.language


def raise_validation_error(msg: str, param: str):
    logger.warning(f"Validation error: {msg}")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[dict(type="invalid_params", msg=msg, loc=["query", param])],
    )


class AudioResponse(Response):
    media_type = "audio/wav"


def load_models(model_holder: ModelHolder):
    model_holder.models = []
    for model_name, model_paths in model_holder.model_files_dict.items():
        model = Model(
            model_path=model_paths[0],
            config_path=os.path.join(model_holder.root_dir, model_name, "config.json"),
            style_vec_path=os.path.join(
                model_holder.root_dir, model_name, "style_vectors.npy"
            ),
            device=model_holder.device,
        )
        model.load_net_g()
        model_holder.models.append(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=config.assets_root
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

    logger.info("Loading models...")
    load_models(model_holder)
    limit = config.server_config.limit
    app = FastAPI()
    allow_origins = config.server_config.origins
    if allow_origins:
        logger.warning(
            f"CORS allow_origins={config.server_config.origins}. If you don't want, modify config.yml"
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server_config.origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.logger = logger

    @app.get("/voice", response_class=AudioResponse)
    async def voice(
        request: Request,
        text: str = Query(..., min_length=1, max_length=limit, description=f"セリフ"),
        encoding: str = Query(None, description="textをURLデコードする(ex, `utf-8`)"),
        model_id: int = Query(0, description="モデルID。`GET /models/info`のkeyの値を指定ください"),
        speaker_name: str = Query(
            None, description="話者名(speaker_idより優先)。esd.listの2列目の文字列を指定"
        ),
        speaker_id: int = Query(
            0, description="話者ID。model_assets>[model]>config.json内のspk2idを確認"
        ),
        sdp_ratio: float = Query(
            DEFAULT_SDP_RATIO,
            description="SDP(Stochastic Duration Predictor)/DP混合比。比率が高くなるほどトーンのばらつきが大きくなる",
        ),
        noise: float = Query(DEFAULT_NOISE, description="サンプルノイズの割合。大きくするほどランダム性が高まる"),
        noisew: float = Query(
            DEFAULT_NOISEW, description="SDPノイズ。大きくするほど発音の間隔にばらつきが出やすくなる"
        ),
        length: float = Query(
            DEFAULT_LENGTH, description="話速。基準は1で大きくするほど音声は長くなり読み上げが遅まる"
        ),
        language: Languages = Query(ln, description=f"textの言語"),
        auto_split: bool = Query(DEFAULT_LINE_SPLIT, description="改行で分けて生成"),
        split_interval: float = Query(
            DEFAULT_SPLIT_INTERVAL, description="分けた場合に挟む無音の長さ（秒）"
        ),
        assist_text: Optional[str] = Query(
            None, description="このテキストの読み上げと似た声音・感情になりやすくなる。ただし抑揚やテンポ等が犠牲になる傾向がある"
        ),
        assist_text_weight: float = Query(
            DEFAULT_ASSIST_TEXT_WEIGHT, description="assist_textの強さ"
        ),
        style: Optional[Union[int, str]] = Query(DEFAULT_STYLE, description="スタイル"),
        style_weight: float = Query(DEFAULT_STYLE_WEIGHT, description="スタイルの強さ"),
        reference_audio_path: Optional[str] = Query(None, description="スタイルを音声ファイルで行う"),
    ):
        """Infer text to speech(テキストから感情付き音声を生成する)"""
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}"
        )
        if model_id >= len(model_holder.models):  # /models/refresh があるためQuery(le)で表現不可
            raise_validation_error(f"model_id={model_id} not found", "model_id")

        model = model_holder.models[model_id]
        if speaker_name is None:
            if speaker_id not in model.id2spk.keys():
                raise_validation_error(
                    f"speaker_id={speaker_id} not found", "speaker_id"
                )
        else:
            if speaker_name not in model.spk2id.keys():
                raise_validation_error(
                    f"speaker_name={speaker_name} not found", "speaker_name"
                )
            speaker_id = model.spk2id[speaker_name]
        if style not in model.style2id.keys():
            raise_validation_error(f"style={style} not found", "style")
        if encoding is not None:
            text = unquote(text, encoding=encoding)
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
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=bool(assist_text),
            style=style,
            style_weight=style_weight,
        )
        logger.success("Audio data generated and sent successfully")
        with BytesIO() as wavContent:
            wavfile.write(wavContent, sr, audio)
            return Response(content=wavContent.getvalue(), media_type="audio/wav")

    @app.get("/models/info")
    def get_loaded_models_info():
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

    @app.post("/models/refresh")
    def refresh():
        """モデルをパスに追加/削除した際などに読み込ませる"""
        model_holder.refresh()
        load_models(model_holder)
        return get_loaded_models_info()

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

    @app.get("/tools/get_audio", response_class=AudioResponse)
    def get_audio(
        request: Request, path: str = Query(..., description="local wav path")
    ):
        """wavデータを取得する"""
        logger.info(
            f"{request.client.host}:{request.client.port}/tools/get_audio  { unquote(str(request.query_params) )}"
        )
        if not os.path.isfile(path):
            raise_validation_error(f"path={path} not found", "path")
        if not path.lower().endswith(".wav"):
            raise_validation_error(f"wav file not found in {path}", "path")
        return FileResponse(path=path, media_type="audio/wav")

    logger.info(f"server listen: http://127.0.0.1:{config.server_config.port}")
    logger.info(f"API docs: http://127.0.0.1:{config.server_config.port}/docs")
    uvicorn.run(
        app, port=config.server_config.port, host="0.0.0.0", log_level="warning"
    )
