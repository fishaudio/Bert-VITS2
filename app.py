import argparse
from pathlib import Path

import gradio as gr
import torch
import yaml

from style_bert_vits2.constants import GRADIO_THEME, VERSION
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModelHolder
from webui.dataset import create_dataset_app
from webui.inference import create_inference_app
from webui.merge import create_merge_app
from webui.style_vectors import create_style_vectors_app
from webui.train import create_train_app


# このプロセスからはワーカーを起動して辞書を使いたいので、ここで初期化
pyopenjtalk_worker.initialize_worker()

# dict_data/ 以下の辞書データを pyopenjtalk に適用
update_dict()

# Get path settings
with Path("configs/paths.yml").open("r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    # dataset_root = path_config["dataset_root"]
    assets_root = path_config["assets_root"]

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=None)
parser.add_argument("--no_autolaunch", action="store_true")
parser.add_argument("--share", action="store_true")

args = parser.parse_args()
device = args.device
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

model_holder = TTSModelHolder(Path(assets_root), device)

with gr.Blocks(theme=GRADIO_THEME) as app:
    gr.Markdown(f"# Style-Bert-VITS2 WebUI (version {VERSION})")
    with gr.Tabs():
        with gr.Tab("音声合成"):
            create_inference_app(model_holder=model_holder)
        with gr.Tab("データセット作成"):
            create_dataset_app()
        with gr.Tab("学習"):
            create_train_app()
        with gr.Tab("スタイル作成"):
            create_style_vectors_app()
        with gr.Tab("マージ"):
            create_merge_app(model_holder=model_holder)


app.launch(
    server_name=args.host,
    server_port=args.port,
    inbrowser=not args.no_autolaunch,
    share=args.share,
)
