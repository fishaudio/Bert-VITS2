import os
import subprocess
import sys

import gradio as gr

python = sys.executable


def subprocess_wrapper(cmd):
    return subprocess.run(
        cmd,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        text=True,
    )


def do_slice(model_name):
    input_dir = "inputs"
    output_dir = os.path.join("Data", model_name, "raw")
    result = subprocess_wrapper(
        [
            python,
            "slice.py",
            "--input_dir",
            input_dir,
            "--output_dir",
            output_dir,
        ]
    )
    return "ターミナルを見て結果を確認してください。"


def do_transcribe(model_name):
    input_dir = os.path.join("Data", model_name, "raw")
    output_file = os.path.join("Data", model_name, "esd.list")
    result = subprocess_wrapper(
        [
            python,
            "transcribe.py",
            "--input_dir",
            input_dir,
            "--output_file",
            output_file,
            "--speaker_name",
            model_name,
        ]
    )
    if result.stderr:
        return f"{result.stderr}"
    return "音声の文字起こしが完了しました。"


initial_md = """
# 学習用データセット作成ツール

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

**ffmpeg のインストールが別途必要のよう**です、「Couldn't find ffmpeg」とか怒られたら、「Windows ffmpeg インストール」等でググって別途インストールしてください。
"""

with gr.Blocks(theme="NoCrypt/miku") as app:
    gr.Markdown(initial_md)
    model_name = gr.Textbox(label="モデル名を入力してください（話者名としても使われます）。")
    with gr.Row():
        slice_button = gr.Button("音声のスライス")
        result1 = gr.Textbox(label="結果")
    with gr.Row():
        transcribe_button = gr.Button("2. 音声の文字起こし")
        result2 = gr.Textbox(label="結果")
    slice_button.click(
        do_slice,
        inputs=[model_name],
        outputs=[result1],
    )
    transcribe_button.click(
        do_transcribe,
        inputs=[model_name],
        outputs=[result2],
    )

app.launch(inbrowser=True)
