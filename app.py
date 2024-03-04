import pyopenjtalk
import gradio as gr
from webui import (
    create_dataset_app,
    create_train_app,
    create_merge_app,
    create_style_vectors_app,
)
from pathlib import Path

pyopenjtalk.unset_user_dict()

setting_json = Path("webui/setting.json")

with gr.Blocks() as app:
    with gr.Tabs():
        with gr.Tab("Hello"):
            gr.Markdown("## Hello, Gradio!")
            gr.Textbox("input", label="Input Text")
        with gr.Tab("Dataset"):
            create_dataset_app()
        with gr.Tab("Train"):
            create_train_app()
        with gr.Tab("Merge"):
            create_merge_app()
        with gr.Tab("Create Style Vectors"):
            create_style_vectors_app()


app.launch(inbrowser=True)
