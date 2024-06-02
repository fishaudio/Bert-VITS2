"""
Style-Bert-VITS2-Editor用のサーバー。
次のリポジトリ
https://github.com/litagin02/Style-Bert-VITS2-Editor
をビルドしてできあがったファイルをWebフォルダに入れて実行する。

TODO: リファクタリングやドキュメンテーションやAPI整理、辞書周りの改善などが必要。
"""

import argparse
import io
import shutil
import sys
import webbrowser
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.io import wavfile

from config import get_path_config
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    VERSION,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.g2p_utils import g2kata_tone, kata_tone2phone_tone
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.nlp.japanese.user_dict import (
    apply_word,
    delete_word,
    read_dict,
    rewrite_word,
    update_dict,
)
from style_bert_vits2.tts_model import TTSModelHolder, TTSModelInfo


# ---フロントエンド部分に関する処理---

# エディターのビルドファイルを配置するディレクトリ
STATIC_DIR = Path("static")
# エディターの最新のビルドファイルのダウンロード日時を記録するファイル
LAST_DOWNLOAD_FILE = STATIC_DIR / "last_download.txt"


def download_static_files(user, repo, asset_name):
    """Style-Bert-VITS2エディターの最新のビルドzipをダウンロードして展開する。"""

    logger.info("Checking for new release...")
    latest_release = get_latest_release(user, repo)
    if latest_release is None:
        logger.warning(
            "Failed to fetch the latest release. Proceeding without static files."
        )
        return

    if not new_release_available(latest_release):
        logger.info("No new release available. Proceeding with existing static files.")
        return

    logger.info("New release available. Downloading static files...")
    asset_url = get_asset_url(latest_release, asset_name)
    if asset_url:
        if STATIC_DIR.exists():
            shutil.rmtree(STATIC_DIR)
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        download_and_extract(asset_url, STATIC_DIR)
        save_last_download(latest_release)
    else:
        logger.warning("Asset not found. Proceeding without static files.")


def get_latest_release(user, repo):
    url = f"https://api.github.com/repos/{user}/{repo}/releases/latest"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def get_asset_url(release, asset_name):
    for asset in release["assets"]:
        if asset["name"] == asset_name:
            return asset["browser_download_url"]
    return None


def download_and_extract(url, extract_to: Path):
    response = requests.get(url)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_to)

    # 展開先が1つのディレクトリだけの場合、その中身を直下に移動する
    extracted_dirs = list(extract_to.iterdir())
    if len(extracted_dirs) == 1 and extracted_dirs[0].is_dir():
        for file in extracted_dirs[0].iterdir():
            file.rename(extract_to / file.name)
        extracted_dirs[0].rmdir()

    # index.htmlが存在するかチェック
    if not (extract_to / "index.html").exists():
        logger.warning("index.html not found in the extracted files.")


def new_release_available(latest_release):
    if LAST_DOWNLOAD_FILE.exists():
        with open(LAST_DOWNLOAD_FILE) as file:
            last_download_str = file.read().strip()
            # 'Z'を除去して日時オブジェクトに変換
            last_download_str = last_download_str.replace("Z", "+00:00")
            last_download = datetime.fromisoformat(last_download_str)
        return (
            datetime.fromisoformat(
                latest_release["published_at"].replace("Z", "+00:00")
            )
            > last_download
        )
    return True


def save_last_download(latest_release):
    with open(LAST_DOWNLOAD_FILE, "w") as file:
        file.write(latest_release["published_at"])


# ---フロントエンド部分に関する処理ここまで---
# 以降はAPIの設定

# pyopenjtalk_worker を起動
## pyopenjtalk_worker は TCP ソケットサーバーのため、ここで起動する
pyopenjtalk.initialize_worker()

# pyopenjtalk の辞書を更新
update_dict()

# 事前に BERT モデル/トークナイザーをロードしておく
## ここでロードしなくても必要になった際に自動ロードされるが、時間がかかるため事前にロードしておいた方が体験が良い
## server_editor.py は日本語にしか対応していないため、日本語の BERT モデル/トークナイザーのみロードする
bert_models.load_model(Languages.JP)
bert_models.load_tokenizer(Languages.JP)


class AudioResponse(Response):
    media_type = "audio/wav"


origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

path_config = get_path_config()
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default=path_config.assets_root)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--inbrowser", action="store_true")
parser.add_argument("--line_length", type=int, default=None)
parser.add_argument("--line_count", type=int, default=None)
# parser.add_argument("--skip_default_models", action="store_true")
parser.add_argument("--skip_static_files", action="store_true")
args = parser.parse_args()
device = args.device
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"
model_dir = Path(args.model_dir)
port = int(args.port)
# if not args.skip_default_models:
#     download_default_models()
skip_static_files = bool(args.skip_static_files)

model_holder = TTSModelHolder(model_dir, device)
if len(model_holder.model_names) == 0:
    logger.error(f"Models not found in {model_dir}.")
    sys.exit(1)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


@router.get("/version")
def version() -> str:
    return VERSION


class MoraTone(BaseModel):
    mora: str
    tone: int


class TextRequest(BaseModel):
    text: str


@router.post("/g2p")
async def read_item(item: TextRequest):
    try:
        # 最初に正規化しないと整合性がとれない
        text = normalize_text(item.text)
        kata_tone_list = g2kata_tone(text)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to convert {item.text} to katakana and tone, {e}",
        )
    return [MoraTone(mora=kata, tone=tone) for kata, tone in kata_tone_list]


@router.post("/normalize")
async def normalize(item: TextRequest):
    return normalize_text(item.text)


@router.get("/models_info", response_model=list[TTSModelInfo])
def models_info():
    return model_holder.models_info


class SynthesisRequest(BaseModel):
    model: str
    modelFile: str
    text: str
    moraToneList: list[MoraTone]
    style: str = DEFAULT_STYLE
    styleWeight: float = DEFAULT_STYLE_WEIGHT
    assistText: str = ""
    assistTextWeight: float = DEFAULT_ASSIST_TEXT_WEIGHT
    speed: float = 1.0
    noise: float = DEFAULT_NOISE
    noisew: float = DEFAULT_NOISEW
    sdpRatio: float = DEFAULT_SDP_RATIO
    language: Languages = Languages.JP
    silenceAfter: float = 0.5
    pitchScale: float = 1.0
    intonationScale: float = 1.0
    speaker: Optional[str] = None


@router.post("/synthesis", response_class=AudioResponse)
def synthesis(request: SynthesisRequest):
    if args.line_length is not None and len(request.text) > args.line_length:
        raise HTTPException(
            status_code=400,
            detail=f"1行の文字数は{args.line_length}文字以下にしてください。",
        )
    try:
        model = model_holder.get_model(
            model_name=request.model, model_path_str=request.modelFile
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model {request.model} from {request.modelFile}, {e}",
        )
    text = request.text
    kata_tone_list = [
        (mora_tone.mora, mora_tone.tone) for mora_tone in request.moraToneList
    ]
    phone_tone = kata_tone2phone_tone(kata_tone_list)
    tone = [t for _, t in phone_tone]
    try:
        sid = 0 if request.speaker is None else model.spk2id[request.speaker]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Speaker {request.speaker} not found in {model.spk2id}",
        )
    sr, audio = model.infer(
        text=text,
        language=request.language,
        sdp_ratio=request.sdpRatio,
        noise=request.noise,
        noise_w=request.noisew,
        length=1 / request.speed,
        given_tone=tone,
        style=request.style,
        style_weight=request.styleWeight,
        assist_text=request.assistText,
        assist_text_weight=request.assistTextWeight,
        use_assist_text=bool(request.assistText),
        line_split=False,
        pitch_scale=request.pitchScale,
        intonation_scale=request.intonationScale,
        speaker_id=sid,
    )

    with BytesIO() as wavContent:
        wavfile.write(wavContent, sr, audio)
        return Response(content=wavContent.getvalue(), media_type="audio/wav")


class MultiSynthesisRequest(BaseModel):
    lines: list[SynthesisRequest]


@router.post("/multi_synthesis", response_class=AudioResponse)
def multi_synthesis(request: MultiSynthesisRequest):
    lines = request.lines
    if args.line_count is not None and len(lines) > args.line_count:
        raise HTTPException(
            status_code=400,
            detail=f"行数は{args.line_count}行以下にしてください。",
        )
    audios = []
    sr = None
    for i, req in enumerate(lines):
        if args.line_length is not None and len(req.text) > args.line_length:
            raise HTTPException(
                status_code=400,
                detail=f"1行の文字数は{args.line_length}文字以下にしてください。",
            )
        try:
            model = model_holder.get_model(
                model_name=req.model, model_path_str=req.modelFile
            )
        except Exception as e:
            logger.error(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model {req.model} from {req.modelFile}, {e}",
            )
        text = req.text
        kata_tone_list = [
            (mora_tone.mora, mora_tone.tone) for mora_tone in req.moraToneList
        ]
        phone_tone = kata_tone2phone_tone(kata_tone_list)
        tone = [t for _, t in phone_tone]
        sr, audio = model.infer(
            text=text,
            language=req.language,
            sdp_ratio=req.sdpRatio,
            noise=req.noise,
            noise_w=req.noisew,
            length=1 / req.speed,
            given_tone=tone,
            style=req.style,
            style_weight=req.styleWeight,
            assist_text=req.assistText,
            assist_text_weight=req.assistTextWeight,
            use_assist_text=bool(req.assistText),
            line_split=False,
            pitch_scale=req.pitchScale,
            intonation_scale=req.intonationScale,
        )
        audios.append(audio)
        if i < len(lines) - 1:
            silence = int(sr * req.silenceAfter)
            audios.append(np.zeros(silence, dtype=np.int16))
    audio = np.concatenate(audios)

    with BytesIO() as wavContent:
        wavfile.write(wavContent, sr, audio)
        return Response(content=wavContent.getvalue(), media_type="audio/wav")


class UserDictWordRequest(BaseModel):
    surface: str
    pronunciation: str
    accent_type: int  # アクセント核位置（存在しない場合は0、1文字目は1）
    priority: int = 5


@router.get("/user_dict")
def get_user_dict():
    return read_dict()


@router.post("/user_dict_word")
def add_user_dict_word(request: UserDictWordRequest):
    uuid = apply_word(
        surface=request.surface,
        pronunciation=request.pronunciation,
        accent_type=request.accent_type,
        priority=request.priority,
    )
    update_dict()

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"uuid": uuid},
    )


@router.put("/user_dict_word/{uuid}")
def update_user_dict_word(uuid: str, request: UserDictWordRequest):
    rewrite_word(
        word_uuid=uuid,
        surface=request.surface,
        pronunciation=request.pronunciation,
        accent_type=request.accent_type,
        priority=request.priority,
    )
    update_dict()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"uuid": uuid})


@router.delete("/user_dict_word/{uuid}")
def delete_user_dict_word(uuid: str):
    delete_word(uuid)
    update_dict()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"uuid": uuid})


app.include_router(router, prefix="/api")

if __name__ == "__main__":
    if not skip_static_files:
        download_static_files("litagin02", "Style-Bert-VITS2-Editor", "out.zip")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    if args.inbrowser:
        webbrowser.open(f"http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
