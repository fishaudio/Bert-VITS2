import gradio as gr

from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.subprocess import run_script_with_log


def do_slice(
    model_name: str,
    min_sec: float,
    max_sec: float,
    min_silence_dur_ms: int,
    time_suffix: bool,
    input_dir: str,
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
    model_name,
    whisper_model,
    compute_type,
    language,
    initial_prompt,
    use_hf_whisper,
    batch_size,
    num_beams,
    hf_repo_id,
):
    if model_name == "":
        return "Error: モデル名を入力してください。"

    cmd = [
        "transcribe.py",
        "--model_name",
        model_name,
        "--model",
        whisper_model,
        "--compute_type",
        compute_type,
        "--language",
        language,
        "--initial_prompt",
        f'"{initial_prompt}"',
        "--num_beams",
        str(num_beams),
    ]
    if use_hf_whisper:
        cmd.append("--use_hf_whisper")
        cmd.extend(["--batch_size", str(batch_size)])
        if hf_repo_id != "openai/whisper":
            cmd.extend(["--hf_repo_id", hf_repo_id])
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}. エラーメッセージが空の場合、何も問題がない可能性があるので、書き起こしファイルをチェックして問題なければ無視してください。"
    return "音声の文字起こしが完了しました。"


how_to_md = """
Style-Bert-VITS2の学習用データセットを作成するためのツールです。以下の2つからなります。

- 与えられた音声からちょうどいい長さの発話区間を切り取りスライス
- 音声に対して文字起こし

このうち両方を使ってもよいし、スライスする必要がない場合は後者のみを使ってもよいです。**コーパス音源などすでに適度な長さの音声ファイルがある場合はスライスは不要**です。

## 必要なもの

学習したい音声が入った音声ファイルいくつか（形式はwav以外でもmp3等通常の音声ファイル形式なら可能）。
合計時間がある程度はあったほうがいいかも、10分とかでも大丈夫だったとの報告あり。単一ファイルでも良いし複数ファイルでもよい。

## スライス使い方
1. `inputs`フォルダに音声ファイルをすべて入れる（スタイル分けをしたい場合は、サブフォルダにスタイルごとに音声を分けて入れる）
2. `モデル名`を入力して、設定を必要なら調整して`音声のスライス`ボタンを押す
3. 出来上がった音声ファイルたちは`Data/{モデル名}/raw`に保存される

## 書き起こし使い方

1. `Data/{モデル名}/raw`に音声ファイルが入っていることを確認（直下でなくてもよい）
2. 設定を必要なら調整してボタンを押す
3. 書き起こしファイルは`Data/{モデル名}/esd.list`に保存される

## 注意

- ~~長すぎる秒数（12-15秒くらいより長い？）のwavファイルは学習に用いられないようです。また短すぎてもあまりよくない可能性もあります。~~ この制限はVer 2.5では学習時に「カスタムバッチサンプラーを使わない」を選択すればなくなりました。が、長すぎる音声があるとVRAM消費量が増えたり安定しなかったりするので、適度な長さにスライスすることをおすすめします。
- 書き起こしの結果をどれだけ修正すればいいかはデータセットに依存しそうです。
"""


def create_dataset_app() -> gr.Blocks:
    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(
            "**既に1ファイル2-12秒程度の音声ファイル集とその書き起こしデータがある場合は、このタブは使用せずに学習できます。**"
        )
        with gr.Accordion("使い方", open=False):
            gr.Markdown(how_to_md)
        model_name = gr.Textbox(
            label="モデル名を入力してください（話者名としても使われます）。"
        )
        with gr.Accordion("音声のスライス"):
            gr.Markdown(
                "**すでに適度な長さの音声ファイルからなるデータがある場合は、その音声をData/{モデル名}/rawに入れれば、このステップは不要です。**"
            )
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(
                        label="元音声の入っているフォルダパス",
                        value="inputs",
                        info="下記フォルダにwavやmp3等のファイルを入れておいてください",
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
                use_hf_whisper = gr.Checkbox(
                    label="HuggingFaceのWhisperを使う（速度が速いがVRAMを多く使う）",
                    value=True,
                )
                hf_repo_id = gr.Dropdown(
                    ["openai/whisper", "kotoba-tech/kotoba-whisper-v1.1"],
                    label="HuggingFaceのWhisperモデル",
                    value="openai/whisper",
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
                    visible=False,
                )
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=128,
                    value=16,
                    step=1,
                    label="バッチサイズ",
                    info="大きくすると速度が速くなるがVRAMを多く使う",
                )
                language = gr.Dropdown(["ja", "en", "zh"], value="ja", label="言語")
                initial_prompt = gr.Textbox(
                    label="初期プロンプト",
                    value="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
                    info="このように書き起こしてほしいという例文（句読点の入れ方・笑い方・固有名詞等）",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="ビームサーチのビーム数",
                    info="小さいほど速度が上がる（以前は5）",
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
                use_hf_whisper,
                batch_size,
                num_beams,
                hf_repo_id,
            ],
            outputs=[result2],
        )
        use_hf_whisper.change(
            lambda x: (
                gr.update(visible=x),
                gr.update(visible=x),
                gr.update(visible=not x),
            ),
            inputs=[use_hf_whisper],
            outputs=[hf_repo_id, batch_size, compute_type],
        )

    return app


if __name__ == "__main__":
    app = create_dataset_app()
    app.launch(inbrowser=True)
