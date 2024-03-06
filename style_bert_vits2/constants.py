from enum import Enum
from pathlib import Path


# Style-Bert-VITS2 のバージョン
VERSION = "2.3.1"

# Gradio のテーマ
## Built-in theme: "default", "base", "monochrome", "soft", "glass"
## See https://huggingface.co/spaces/gradio/theme-gallery for more themes
GRADIO_THEME = "NoCrypt/miku"

# デフォルトのユーザー辞書ディレクトリ
## style_bert_vits2.text_processing.japanese.user_dict モジュールのデフォルト値として利用される
## ライブラリとしての利用などで外部のユーザー辞書を指定したい場合は、user_dict 以下の各関数の実行時、引数に辞書データファイルのパスを指定する
DEFAULT_USER_DICT_DIR = Path(__file__).parent.parent / "dict_data"

# デフォルトの推論パラメータ
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

# 利用可能な言語
## JP-Extra モデル利用時は JP 以外の言語の音声合成はできない
class Languages(str, Enum):
    JP = "JP"
    EN = "EN"
    ZH = "ZH"
