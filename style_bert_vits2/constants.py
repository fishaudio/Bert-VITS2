from enum import Enum
from pathlib import Path


# Style-Bert-VITS2 のバージョン
VERSION = "2.3.1"

# ユーザー辞書ディレクトリ
USER_DICT_DIR = Path("dict_data")

# Gradio のテーマ
## Built-in theme: "default", "base", "monochrome", "soft", "glass"
## See https://huggingface.co/spaces/gradio/theme-gallery for more themes
GRADIO_THEME = "NoCrypt/miku"

# 利用可能な言語
class Languages(str, Enum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"

# 推論パラメータのデフォルト値
DEFAULT_STYLE = "Neutral"
DEFAULT_STYLE_WEIGHT = 5.0
DEFAULT_SDP_RATIO = 0.2
DEFAULT_NOISE = 0.6
DEFAULT_NOISEW = 0.8
DEFAULT_LENGTH = 1.0
DEFAULT_LINE_SPLIT = True
DEFAULT_SPLIT_INTERVAL = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT = 0.7
DEFAULT_ASSIST_TEXT_WEIGHT = 1.0
