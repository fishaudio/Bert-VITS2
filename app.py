import argparse
from pathlib import Path

import gradio as gr
import torch
import yaml

from style_bert_vits2.constants import GRADIO_THEME, LATEST_VERSION
from style_bert_vits2.tts_model import ModelHolder
from webui import (
    create_dataset_app,
    create_inference_app,
    create_merge_app,
    create_style_vectors_app,
    create_train_app,
)


# Get path settings
with Path("configs/paths.yml").open("r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    # dataset_root = path_config["dataset_root"]
    assets_root = path_config["assets_root"]

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--no_autolaunch", action="store_true")
parser.add_argument("--share", action="store_true")

args = parser.parse_args()
device = args.device
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

model_holder = ModelHolder(Path(assets_root), device)

with gr.Blocks(theme=GRADIO_THEME) as app:
    gr.Markdown(f"# Style-Bert-VITS2 WebUI (version {LATEST_VERSION})")
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


app.launch(inbrowser=not args.no_autolaunch, share=args.share)
