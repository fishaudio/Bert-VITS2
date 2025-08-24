from pathlib import Path

import gradio as gr

from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import NullModelParam, TTSModelHolder
from style_bert_vits2.utils.subprocess import run_script_with_log


def call_convert_onnx(
    model: str,
):
    if model == "":
        return "Error: モデル名を入力してください。"
    logger.info("Start converting model to onnx...")
    cmd = [
        "convert_onnx.py",
        "--model",
        model,
    ]
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}"
    return "ONNX変換が完了しました。"


initial_md = """
safetensors形式のモデルをONNX形式に変換します。
このONNXモデルは、[AIVM Generator](https://aivm-generator.aivis-project.com/) 等でさらにAIVM形式・AIVMX形式に変換して[AivisSpeech](https://aivis-project.com/)で利用できます。

変換には5分以上ほどの時間がかかります。進捗状況はターミナルのログを参照してください。
"""


def create_onnx_app(model_holder: TTSModelHolder) -> gr.Blocks:
    def get_model_files(model_name: str):
        return [str(f) for f in model_holder.model_files_dict[model_name]]

    model_names = model_holder.model_names
    if len(model_names) == 0:
        logger.error(
            f"モデルが見つかりませんでした。{model_holder.root_dir}にモデルを置いてください。"
        )
        with gr.Blocks() as app:
            gr.Markdown(
                f"Error: モデルが見つかりませんでした。{model_holder.root_dir}にモデルを置いてください。"
            )
        return app
    initial_id = 0
    initial_pth_files = get_model_files(model_names[initial_id])

    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(initial_md)
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    label="モデル一覧",
                    choices=model_names,
                    value=model_names[initial_id],
                )
                model_path = gr.Dropdown(
                    label="モデルファイル",
                    choices=initial_pth_files,
                    value=initial_pth_files[0],
                )
            refresh_button = gr.Button("更新")
        convert_button = gr.Button("ONNX形式に変換", variant="primary")
        info = gr.Textbox(label="情報")

        model_name.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name],
            outputs=[model_path],
        )

        def refresh_fn() -> tuple[gr.Dropdown, gr.Dropdown]:
            names, files, _ = model_holder.update_model_names_for_gradio()
            return names, files

        refresh_button.click(
            refresh_fn,
            outputs=[model_name, model_path],
        )
        convert_button.click(
            call_convert_onnx,
            inputs=[model_path],
            outputs=[info],
        )

    return app


if __name__ == "__main__":
    from config import get_path_config

    path_config = get_path_config()
    assets_root = path_config.assets_root
    model_holder = TTSModelHolder(assets_root, "cpu", "", ignore_onnx=True)
    app = create_onnx_app(model_holder)
    app.launch(inbrowser=True)
