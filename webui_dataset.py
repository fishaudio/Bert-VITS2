import os

import gradio as gr
import yaml

from common.log import logger
from common.subprocess_utils import run_script_with_log

# Get path settings
with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    dataset_root = path_config["dataset_root"]
    # assets_root = path_config["assets_root"]


def do_slice(model_name, normalize):
    logger.info("Start slicing...")
    input_dir = "inputs"
    output_dir = os.path.join(dataset_root, model_name, "raw")
    cmd = [
        "slice.py",
        "--input_dir",
        input_dir,
        "--output_dir",
        output_dir,
    ]
    if normalize:
        cmd.append("--normalize")
    success, message = run_script_with_log(cmd)
    if not success:
        return f"Error: {message}"
    return "音声のスライスが完了しました。"


def do_transcribe(model_name):
    input_dir = os.path.join(dataset_root, model_name, "raw")
    output_file = os.path.join(dataset_root, model_name, "esd.list")
    result = run_script_with_log(
        [
            "transcribe.py",
            "--input_dir",
            input_dir,
            "--output_file",
            output_file,
            "--speaker_name",
            model_name,
        ]
    )
    return "音声の文字起こしが完了しました。"


initial_md = """
# 簡易学習用データセット作成ツール

**注意**：より精密で高品質なデータセットを作成したい・書き起こしをいろいろ修正したい場合は、[Aivis Dataset](https://github.com/litagin02/Aivis-Dataset)をおすすめします。書き起こし部分もかなり工夫されています。このツールはあくまでスライスして書き起こすという簡易的なことしかしていません。

Style-Bert-VITS2の学習用データセットを作成するためのツールです。与えられた音声からちょうどいい長さの発話区間を切り取りスライスし、それぞれの音声に対して文字起こしを行います。

## 必要なもの
学習したい音声が入ったwavファイルいくつか。
合計時間がある程度はあったほうがいいかも、10分とかでも大丈夫だったとの報告あり。単一ファイルでも良いし複数ファイルでもよい。

## 使い方
1. `inputs`フォルダ直下にwavファイルをすべて入れる
2. `モデル名`を入力して、`音声のスライス`ボタンを押す
3. 完了したら、`音声の文字起こし`ボタンを押す

細かいパラメータ調整とかがしたい人は、`slice.py`と`transcribe.py`を眺めて直接実行してください。

また、出来上がった音声ファイルたちは`Data/{モデル名}/raw`に、書き起こしファイルは`Data/{モデル名}/esd.list`に保存されます。
書き起こしの結果をどれだけ修正すればいいかはデータセットに依存しそうです。
"""

with gr.Blocks(theme="NoCrypt/miku") as app:
    gr.Markdown(initial_md)
    model_name = gr.Textbox(label="モデル名を入力してください（話者名としても使われます）。")
    with gr.Accordion("音声のスライス"):
        with gr.Row():
            with gr.Column():
                normalize = gr.Checkbox(label="スライスされた音声の音量を正規化する", value=True)
                slice_button = gr.Button("スライスを実行")
            result1 = gr.Textbox(label="結果")
    with gr.Row():
        transcribe_button = gr.Button("音声の文字起こし")
        result2 = gr.Textbox(label="結果")
    slice_button.click(
        do_slice,
        inputs=[model_name, normalize],
        outputs=[result1],
    )
    transcribe_button.click(
        do_transcribe,
        inputs=[model_name],
        outputs=[result2],
    )

app.launch(inbrowser=True)
