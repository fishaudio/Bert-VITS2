import gradio as gr

from style_bert_vits2.logging import logger
from style_bert_vits2.utils.subprocess import run_script_with_log


def do_slice(
    model_name: str,
    min_sec: float,
    max_sec: float,
    min_silence_dur_ms: int,
    time_suffix: bool,
    input_dir: str,
    num_processes: int = 3,
):
    if model_name == "":
        return "Error: モデル名を入力してください。"
    logger.info("Start slicing...")
    cmd = [
        "slice.py",
        "--model_name",
        model_name,
        "--min_sec",
        str(min_sec),
        "--max_sec",
        str(max_sec),
        "--min_silence_dur_ms",
        str(min_silence_dur_ms),
        "--num_processes",
        str(num_processes),
    ]
    if time_suffix:
        cmd.append("--time_suffix")
    if input_dir != "":
        cmd += ["--input_dir", input_dir]
    # onnxの警告が出るので無視する
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}"
    return "音声のスライスが完了しました。"


def do_transcribe(
    model_name, whisper_model, compute_type, language, initial_prompt, device
):
    if model_name == "":
        return "Error: モデル名を入力してください。"

    success, message = run_script_with_log(
        [
            "transcribe.py",
            "--model_name",
            model_name,
            "--model",
            whisper_model,
            "--compute_type",
            compute_type,
            "--device",
            device,
            "--language",
            language,
            "--initial_prompt",
            f'"{initial_prompt}"',
        ]
    )
    if not success:
        return f"Error: {message}. しかし何故かエラーが起きても正常に終了している場合がほとんどなので、書き起こし結果を確認して問題なければ学習に使えます。"
    return "音声の文字起こしが完了しました。"


how_to_md = """
Style-Bert-VITS2の学習用データセットを作成するためのツールです。以下の2つからなります。

- 与えられた音声からちょうどいい長さの発話区間を切り取りスライス
- 音声に対して文字起こし

このうち両方を使ってもよいし、スライスする必要がない場合は後者のみを使ってもよいです。

## 必要なもの

学習したい音声が入ったwavファイルいくつか。
合計時間がある程度はあったほうがいいかも、10分とかでも大丈夫だったとの報告あり。単一ファイルでも良いし複数ファイルでもよい。

## スライス使い方
1. `inputs`フォルダにwavファイルをすべて入れる
2. `モデル名`を入力して、設定を必要なら調整して`音声のスライス`ボタンを押す
3. 出来上がった音声ファイルたちは`Data/{モデル名}/raw`に保存される

## 書き起こし使い方

1. 書き起こしたい音声ファイルのあるフォルダを指定（デフォルトは`Data/{モデル名}/raw`なのでスライス後に行う場合は省略してよい）
2. 設定を必要なら調整してボタンを押す
3. 書き起こしファイルは`Data/{モデル名}/esd.list`に保存される

## 注意

- 長すぎる秒数（12-15秒くらいより長い？）のwavファイルは学習に用いられないようです。また短すぎてもあまりよくない可能性もあります。
- 書き起こしの結果をどれだけ修正すればいいかはデータセットに依存しそうです。
- 手動で書き起こしをいろいろ修正したり結果を細かく確認したい場合は、[Aivis Dataset](https://github.com/litagin02/Aivis-Dataset)もおすすめします。書き起こし部分もかなり工夫されています。ですがファイル数が多い場合などは、このツールで簡易的に切り出してデータセットを作るだけでも十分という気もしています。
"""


def create_dataset_app() -> gr.Blocks:
    with gr.Blocks() as app:
        with gr.Accordion("使い方", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(
            label="モデル名を入力してください（話者名としても使われます）。"
        )
        with gr.Accordion("音声のスライス"):
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(
                        label="元音声の入っているフォルダパス",
                        value="inputs",
                        info="下記フォルダにwavファイルを入れておいてください",
                    )
                    min_sec = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=0.5,
                        label="この秒数未満は切り捨てる",
                    )
                    max_sec = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=12,
                        step=0.5,
                        label="この秒数以上は切り捨てる",
                    )
                    min_silence_dur_ms = gr.Slider(
                        minimum=0,
                        maximum=2000,
                        value=700,
                        step=100,
                        label="無音とみなして区切る最小の無音の長さ（ms）",
                    )
                    time_suffix = gr.Checkbox(
                        value=False,
                        label="WAVファイル名の末尾に元ファイルの時間範囲を付与する",
                    )
                    num_processes = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="並列処理数（速度向上のため）",
                        info="3で十分高速、多くしてもCPU負荷が増すだけでそこまで速度は変わらない",
                    )
                    slice_button = gr.Button("スライスを実行")
                result1 = gr.Textbox(label="結果")
        with gr.Row():
            with gr.Column():
                whisper_model = gr.Dropdown(
                    [
                        "tiny",
                        "base",
                        "small",
                        "medium",
                        "large",
                        "large-v2",
                        "large-v3",
                    ],
                    label="Whisperモデル",
                    value="large-v3",
                )
                compute_type = gr.Dropdown(
                    [
                        "int8",
                        "int8_float32",
                        "int8_float16",
                        "int8_bfloat16",
                        "int16",
                        "float16",
                        "bfloat16",
                        "float32",
                    ],
                    label="計算精度",
                    value="bfloat16",
                )
                device = gr.Radio(["cuda", "cpu"], label="デバイス", value="cuda")
                language = gr.Dropdown(["ja", "en", "zh"], value="ja", label="言語")
                initial_prompt = gr.Textbox(
                    label="初期プロンプト",
                    value="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
                    info="このように書き起こしてほしいという例文（句読点の入れ方・笑い方・固有名詞等）",
                )
            transcribe_button = gr.Button("音声の文字起こし")
            result2 = gr.Textbox(label="結果")
        slice_button.click(
            do_slice,
            inputs=[
                model_name,
                min_sec,
                max_sec,
                min_silence_dur_ms,
                time_suffix,
                input_dir,
                num_processes,
            ],
            outputs=[result1],
        )
        transcribe_button.click(
            do_transcribe,
            inputs=[
                model_name,
                whisper_model,
                compute_type,
                language,
                initial_prompt,
                device,
            ],
            outputs=[result2],
        )

    return app
