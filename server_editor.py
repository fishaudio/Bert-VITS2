import csv
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pyopenjtalk
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.io import wavfile

from common.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    USER_DICT_CSV_PATH,
    USER_DICT_PATH,
    Languages,
    LATEST_VERSION,
)
from common.log import logger
from common.tts_model import ModelHolder
from text.japanese import g2kata_tone, kata_tone2phone_tone


class AudioResponse(Response):
    media_type = "audio/wav"


origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

device = "cuda"
model_dir = Path("model_assets/server_test")
model_holder = ModelHolder(model_dir, device)
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


@app.get("/version")
def version() -> str:
    return LATEST_VERSION


class MoraTone(BaseModel):
    mora: str
    tone: int


class G2PRequest(BaseModel):
    text: str


@app.post("/g2p")
async def read_item(item: G2PRequest):
    try:
        kata_tone_list = g2kata_tone(item.text, ignore_unknown=True)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to convert {item.text} to katakana and tone, {e}",
        )
    return [MoraTone(mora=kata, tone=tone) for kata, tone in kata_tone_list]


@app.get("/models_info")
def models_info():
    return model_holder.models_info()


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


@app.post("/synthesis", response_class=AudioResponse)
def synthesis(request: SynthesisRequest):
    try:
        model = model_holder.load_model(
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
    sr, audio = model.infer(
        text=text,
        language=request.language.value,
        sdp_ratio=request.sdpRatio,
        noise=request.noise,
        noisew=request.noisew,
        length=1 / request.speed,
        given_tone=tone,
        style=request.style,
        style_weight=request.styleWeight,
        assist_text=request.assistText,
        assist_text_weight=request.assistTextWeight,
        use_assist_text=bool(request.assistText),
        line_split=False,
        ignore_unknown=True,
        pitch_scale=request.pitchScale,
        intonation_scale=request.intonationScale,
    )

    with BytesIO() as wavContent:
        wavfile.write(wavContent, sr, audio)
        return Response(content=wavContent.getvalue(), media_type="audio/wav")


class MultiSynthesisRequest(BaseModel):
    lines: list[SynthesisRequest]


@app.post("/multi-synthesis", response_class=AudioResponse)
def multi_synthesis(request: MultiSynthesisRequest):
    lines = request.lines
    audios = []
    for i, req in enumerate(lines):
        # Loade model
        try:
            model = model_holder.load_model(
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
            language=req.language.value,
            sdp_ratio=req.sdpRatio,
            noise=req.noise,
            noisew=req.noisew,
            length=1 / req.speed,
            given_tone=tone,
            style=req.style,
            style_weight=req.styleWeight,
            assist_text=req.assistText,
            assist_text_weight=req.assistTextWeight,
            use_assist_text=bool(req.assistText),
            line_split=False,
            ignore_unknown=True,
            pitch_scale=req.pitchScale,
            intonation_scale=req.intonationScale,
        )
        audios.append(audio)
        if i < len(lines) - 1:
            silence = int(sr * req.silenceAfter)
            audios.append(np.zeros(silence, dtype=np.int16))
    audio = np.concatenate(audios)
    logger.debug(audio.dtype)

    with BytesIO() as wavContent:
        wavfile.write(wavContent, sr, audio)
        return Response(content=wavContent.getvalue(), media_type="audio/wav")


class AddUserDictWordRequest(BaseModel):
    surface: str
    pronunciation: str
    accentType: str  # アクセント核位置/モーラ数、例：1/3, アクセント核位置は1から始まる
    priority: int = 5


# コストの値は以下の値を参考にしている
# https://github.com/VOICEVOX/voicevox_engine/blob/master/voicevox_engine/user_dict/part_of_speech_data.py
COST_CANDIDATES = [-988, 3488, 4768, 6048, 7328, 8609, 8734, 8859, 8984, 9110, 14176]


@app.post("/user_dict_word")
def add_user_dict_word(request: AddUserDictWordRequest):
    if request.surface == "" or request.pronunciation == "":
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"message": "単語か読みが空です。"},
        )
    user_dict_file = Path(USER_DICT_CSV_PATH)
    user_dict_file.parent.mkdir(parents=True, exist_ok=True)

    # 新規追加または更新する単語のCSV行
    new_csv_row = f"{request.surface},,,{COST_CANDIDATES[request.priority]},名詞,固有名詞,一般,*,*,*,{request.surface},{request.pronunciation},{request.pronunciation},{request.accentType},*\n"
    found = False
    updated = False

    # ユーザー辞書ファイルが存在する場合、既存の単語をチェックし、必要に応じて更新する
    if user_dict_file.exists():
        with user_dict_file.open(encoding="utf-8") as f:
            lines = f.readlines()

        with user_dict_file.open("w", encoding="utf-8") as f:
            for line in lines:
                if line.split(",")[0] == request.surface:
                    found = True
                    if line.strip() != new_csv_row.strip():
                        f.write(new_csv_row)
                        updated = True
                    else:
                        f.write(line)
                else:
                    f.write(line)
    # 単語が新しい場合、新規に追加
    if not found:
        with user_dict_file.open("a", encoding="utf-8") as f:
            f.write(new_csv_row)

    pyopenjtalk.unset_user_dict()
    pyopenjtalk.mecab_dict_index(USER_DICT_CSV_PATH, USER_DICT_PATH)
    pyopenjtalk.update_global_jtalk_with_user_dict(USER_DICT_PATH)

    if updated:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "単語が既に存在し、更新されました。"},
        )
    elif found:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "既に登録されている単語です。更新はありません。"},
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"message": "単語が新規追加されました。"},
        )


app.mount("/", StaticFiles(directory="Web"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
