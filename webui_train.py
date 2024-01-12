import argparse
import json
import os
import shutil
from datetime import datetime
from multiprocessing import cpu_count

import gradio as gr
import yaml

from common.log import logger
from common.subprocess_utils import run_script_with_log, second_elem_of

logger_handler = None

# Get path settings
with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    dataset_root = path_config["dataset_root"]
    # assets_root = path_config["assets_root"]


def get_path(model_name):
    assert model_name != "", "モデル名は空にできません"
    dataset_path = os.path.join(dataset_root, model_name)
    lbl_path = os.path.join(dataset_path, "esd.list")
    train_path = os.path.join(dataset_path, "train.list")
    val_path = os.path.join(dataset_path, "val.list")
    config_path = os.path.join(dataset_path, "config.json")
    return dataset_path, lbl_path, train_path, val_path, config_path


def initialize(model_name, batch_size, epochs, save_every_steps, bf16_run):
    global logger_handler
    dataset_path, _, train_path, val_path, config_path = get_path(model_name)

    # 前処理のログをファイルに保存する
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"preprocess_{timestamp}.log"
    if logger_handler is not None:
        logger.remove(logger_handler)
    logger_handler = logger.add(os.path.join(dataset_path, file_name))

    logger.info(
        f"Step 1: start initialization...\nmodel_name: {model_name}, batch_size: {batch_size}, epochs: {epochs}, save_every_steps: {save_every_steps}, bf16_run: {bf16_run}"
    )
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        # Use default config
        with open("configs/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    config["model_name"] = model_name
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs
    config["train"]["bf16_run"] = bf16_run
    config["train"]["eval_interval"] = save_every_steps

    model_path = os.path.join(dataset_path, "models")
    if os.path.exists(model_path):
        logger.warning(f"Step 1: {model_path} already exists, so copy it to backup.")
        shutil.copytree(
            src=model_path,
            dst=os.path.join(dataset_path, "models_backup"),
            dirs_exist_ok=True,
        )
        shutil.rmtree(model_path)
    try:
        shutil.copytree(
            src="pretrained",
            dst=model_path,
        )
    except FileNotFoundError:
        logger.error("Step 1: `pretrained` folder not found.")
        return False, "Step 1, Error: pretrainedフォルダが見つかりません。"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if not os.path.exists("config.yml"):
        shutil.copy(src="default_config.yml", dst="config.yml")
    # yml_data = safe_load(open("config.yml", "r", encoding="utf-8"))
    with open("config.yml", "r", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = dataset_path
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)
    logger.success("Step 1: initialization finished.")
    return True, "Step 1, Success: 初期設定が完了しました"


def resample(model_name, normalize, trim, num_processes):
    logger.info("Step 2: start resampling...")
    dataset_path, _, _, _, _ = get_path(model_name)
    in_dir = os.path.join(dataset_path, "raw")
    out_dir = os.path.join(dataset_path, "wavs")
    cmd = [
        "resample.py",
        "--in_dir",
        in_dir,
        "--out_dir",
        out_dir,
        "--num_processes",
        str(num_processes),
        "--sr",
        "44100",
    ]
    if normalize:
        cmd.append("--normalize")
    if trim:
        cmd.append("--trim")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error(f"Step 2: resampling failed.")
        return False, f"Step 2, Error: 音声ファイルの前処理に失敗しました:\n{message}"
    elif message:
        logger.warning(f"Step 2: resampling finished with stderr.")
        return True, f"Step 2, Success: 音声ファイルの前処理が完了しました:\n{message}"
    logger.success("Step 2: resampling finished.")
    return True, "Step 2, Success: 音声ファイルの前処理が完了しました"


def preprocess_text(model_name):
    logger.info("Step 3: start preprocessing text...")
    dataset_path, lbl_path, train_path, val_path, config_path = get_path(model_name)
    try:
        lines = open(lbl_path, "r", encoding="utf-8").readlines()
    except FileNotFoundError:
        logger.error(f"Step 3: {lbl_path} not found.")
        return False, f"Step 3, Error: 書き起こしファイル {lbl_path} が見つかりません。"
    with open(lbl_path, "w", encoding="utf-8") as f:
        for line in lines:
            path, spk, language, text = line.strip().split("|")
            path = os.path.join(dataset_path, "wavs", os.path.basename(path)).replace(
                "\\", "/"
            )
            f.writelines(f"{path}|{spk}|{language}|{text}\n")
    success, message = run_script_with_log(
        [
            "preprocess_text.py",
            "--config-path",
            config_path,
            "--transcription-path",
            lbl_path,
            "--train-path",
            train_path,
            "--val-path",
            val_path,
        ]
    )
    if not success:
        logger.error(f"Step 3: preprocessing text failed.")
        return False, f"Step 3, Error: 書き起こしファイルの前処理に失敗しました:\n{message}"
    elif message:
        logger.warning(f"Step 3: preprocessing text finished with stderr.")
        return True, f"Step 3, Success: 書き起こしファイルの前処理が完了しました:\n{message}"
    logger.success("Step 3: preprocessing text finished.")
    return True, "Step 3, Success: 書き起こしファイルの前処理が完了しました"


def bert_gen(model_name):
    logger.info("Step 4: start bert_gen...")
    _, _, _, _, config_path = get_path(model_name)
    success, message = run_script_with_log(
        [
            "bert_gen.py",
            "--config",
            config_path,
            # "--num_processes",  # bert_genは重いのでプロセス数いじらない
            #  str(num_processes),
        ]
    )
    if not success:
        logger.error(f"Step 4: bert_gen failed.")
        return False, f"Step 4, Error: BERT特徴ファイルの生成に失敗しました:\n{message}"
    elif message:
        logger.warning(f"Step 4: bert_gen finished with stderr.")
        return True, f"Step 4, Success: BERT特徴ファイルの生成が完了しました:\n{message}"
    logger.success("Step 4: bert_gen finished.")
    return True, "Step 4, Success: BERT特徴ファイルの生成が完了しました"


def style_gen(model_name, num_processes):
    logger.info("Step 5: start style_gen...")
    _, _, _, _, config_path = get_path(model_name)
    success, message = run_script_with_log(
        [
            "style_gen.py",
            "--config",
            config_path,
            "--num_processes",
            str(num_processes),
        ]
    )
    if not success:
        logger.error(f"Step 5: style_gen failed.")
        return False, f"Step 5, Error: スタイル特徴ファイルの生成に失敗しました:\n{message}"
    elif message:
        logger.warning(f"Step 5: style_gen finished with stderr.")
        return True, f"Step 5, Success: スタイル特徴ファイルの生成が完了しました:\n{message}"
    logger.success("Step 5: style_gen finished.")
    return True, "Step 5, Success: スタイル特徴ファイルの生成が完了しました"


def preprocess_all(
    model_name,
    batch_size,
    epochs,
    save_every_steps,
    bf16_run,
    num_processes,
    normalize,
    trim,
):
    if model_name == "":
        return False, "Error: モデル名を入力してください"
    success, message = initialize(
        model_name, batch_size, epochs, save_every_steps, bf16_run
    )
    if not success:
        return False, message
    success, message = resample(model_name, normalize, trim, num_processes)
    if not success:
        return False, message
    success, message = preprocess_text(model_name)
    if not success:
        return False, message
    success, message = bert_gen(model_name)  # bert_genは重いのでプロセス数いじらない
    if not success:
        return False, message
    success, message = style_gen(model_name, num_processes)
    if not success:
        return False, message
    logger.success("Success: All preprocess finished!")
    return True, "Success: 全ての前処理が完了しました。ターミナルを確認しておかしいところがないか確認するのをおすすめします。"


def train(model_name):
    dataset_path, _, _, _, config_path = get_path(model_name)
    # 学習再開の場合は念のためconfig.ymlの名前等を更新
    with open("config.yml", "r", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = dataset_path
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)
    success, message = run_script_with_log(
        ["train_ms.py", "--config", config_path, "--model", dataset_path]
    )
    if not success:
        logger.error(f"Train failed.")
        return False, f"Error: 学習に失敗しました:\n{message}"
    elif message:
        logger.warning(f"Train finished with stderr.")
        return True, f"Success: 学習が完了しました:\n{message}"
    logger.success("Train finished.")
    return True, "Success: 学習が完了しました"


initial_md = """
# Style-Bert-VITS2 学習用WebUI

## 使い方

- データを準備して、モデル名を入力して、必要なら設定を調整してから、「自動前処理を実行」ボタンを押してください。進捗状況等はターミナルに表示されます。

- 各ステップごとに実行する場合は「手動前処理」を使ってください（基本的には自動でいいはず）。

- 前処理が終わったら、「学習を開始する」ボタンを押すと学習が開始されます。

- 途中から学習を再開する場合は、モデル名を入力してから「学習を開始する」を押せばよいです。

注意: 音声合成で使うには、スタイルベクトルファイル`style_vectors.npy`を作る必要があります。これは、`Style.bat`を実行してそこで作成してください。
動作は軽いはずなので、学習中でも実行でき、何度でも繰り返して試せます。
"""

prepare_md = """
まず音声データ（wavファイルで1ファイルが2-12秒程度の、長すぎず短すぎない発話のものをいくつか）と、書き起こしテキストを用意してください。

それを次のように配置します。
```
├── Data
│   ├── {モデルの名前}
│   │   ├── esd.list
│   │   ├── raw
│   │   │   ├── ****.wav
│   │   │   ├── ****.wav
│   │   │   ├── ...
```

wavファイル名やモデルの名前は空白を含まない半角で、wavファイルの拡張子は小文字`.wav`である必要があります。
`raw` フォルダにはすべてのwavファイルを入れ、`esd.list` ファイルには、以下のフォーマットで各wavファイルの情報を記述してください。
```
****.wav|{話者名}|{言語ID、ZHかJPかEN}|{書き起こしテキスト}
```

例：
```
wav_number1.wav|hanako|JP|こんにちは、聞こえて、いますか？
wav_next.wav|taro|JP|はい、聞こえています……。
english_teacher.wav|Mary|EN|How are you? I'm fine, thank you, and you?
...
```
日本語話者の単一話者データセットでも構いません。
"""

if __name__ == "__main__":
    with gr.Blocks(theme="NoCrypt/miku") as app:
        gr.Markdown(initial_md)
        with gr.Accordion(label="データの前準備", open=False):
            gr.Markdown(prepare_md)
        model_name = gr.Textbox(
            label="モデル名",
        )
        gr.Markdown("### 自動前処理")
        with gr.Row(variant="panel"):
            with gr.Column():
                batch_size = gr.Slider(
                    label="バッチサイズ",
                    value=4,
                    minimum=1,
                    maximum=64,
                    step=1,
                )
                epochs = gr.Slider(
                    label="エポック数",
                    info="100もあれば十分そうだけどもっと回すと質が上がるかもしれない",
                    value=100,
                    minimum=10,
                    maximum=1000,
                    step=10,
                )
                save_every_steps = gr.Slider(
                    label="何ステップごとに結果を保存するか",
                    info="エポック数とは違うことに注意",
                    value=1000,
                    minimum=100,
                    maximum=10000,
                    step=100,
                )
                bf16_run = gr.Checkbox(
                    label="bfloat16を使う",
                    info="bfloat16を使うかどうか。新しめのグラボだと学習が早くなるかも、古いグラボだと動かないかも。",
                    value=True,
                )
                num_processes = gr.Slider(
                    label="プロセス数",
                    info="前処理時の並列処理プロセス数、大きすぎるとフリーズするかも",
                    value=cpu_count() // 2,
                    minimum=1,
                    maximum=cpu_count(),
                    step=1,
                )
                normalize = gr.Checkbox(
                    label="音声の音量を正規化する(音量の大小が揃っていない場合など)",
                    value=False,
                )
                trim = gr.Checkbox(
                    label="音声の最初と最後の無音を取り除く",
                    value=False,
                )
            with gr.Column():
                preprocess_button = gr.Button(value="自動前処理を実行", variant="primary")
                info_all = gr.Textbox(label="状況")
        with gr.Accordion(open=False, label="手動前処理"):
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Step 1: 設定ファイルの生成")
                    batch_size_manual = gr.Slider(
                        label="バッチサイズ",
                        value=4,
                        minimum=1,
                        maximum=64,
                        step=1,
                    )
                    epochs_manual = gr.Slider(
                        label="エポック数",
                        value=100,
                        minimum=1,
                        maximum=1000,
                        step=1,
                    )
                    save_every_steps_manual = gr.Slider(
                        label="何ステップごとに結果を保存するか",
                        value=1000,
                        minimum=100,
                        maximum=10000,
                        step=100,
                    )
                    bf16_run_manual = gr.Checkbox(
                        label="bfloat16を使う",
                        value=True,
                    )
                with gr.Column():
                    generate_config_btn = gr.Button(value="実行", variant="primary")
                    info_init = gr.Textbox(label="状況")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Step 2: 音声ファイルの前処理")
                    num_processes_resample = gr.Slider(
                        label="プロセス数",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                    normalize_resample = gr.Checkbox(
                        label="音声の音量を正規化する",
                        value=False,
                    )
                    trim_resample = gr.Checkbox(
                        label="音声の最初と最後の無音を取り除く",
                        value=False,
                    )
                with gr.Column():
                    resample_btn = gr.Button(value="実行", variant="primary")
                    info_resample = gr.Textbox(label="状況")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Step 3: 書き起こしファイルの前処理")
                with gr.Column():
                    preprocess_text_btn = gr.Button(value="実行", variant="primary")
                    info_preprocess_text = gr.Textbox(label="状況")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Step 4: BERT特徴ファイルの生成")
                with gr.Column():
                    bert_gen_btn = gr.Button(value="実行", variant="primary")
                    info_bert = gr.Textbox(label="状況")
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Step 5: スタイル特徴ファイルの生成")
                    num_processes_style = gr.Slider(
                        label="プロセス数",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                with gr.Column():
                    style_gen_btn = gr.Button(value="実行", variant="primary")
                    info_style = gr.Textbox(label="状況")
        gr.Markdown("## 学習")
        with gr.Row(variant="panel"):
            train_btn = gr.Button(value="学習を開始する", variant="primary")
            info_train = gr.Textbox(label="状況")

        preprocess_button.click(
            second_elem_of(preprocess_all),
            inputs=[
                model_name,
                batch_size,
                epochs,
                save_every_steps,
                bf16_run,
                num_processes,
                normalize,
                trim,
            ],
            outputs=[info_all],
        )
        generate_config_btn.click(
            second_elem_of(initialize),
            inputs=[
                model_name,
                batch_size_manual,
                epochs_manual,
                save_every_steps_manual,
                bf16_run_manual,
            ],
            outputs=[info_init],
        )
        resample_btn.click(
            second_elem_of(resample),
            inputs=[
                model_name,
                normalize_resample,
                trim_resample,
                num_processes_resample,
            ],
            outputs=[info_resample],
        )
        preprocess_text_btn.click(
            second_elem_of(preprocess_text),
            inputs=[model_name],
            outputs=[info_preprocess_text],
        )
        bert_gen_btn.click(
            second_elem_of(bert_gen),
            inputs=[model_name],
            outputs=[info_bert],
        )
        style_gen_btn.click(
            second_elem_of(style_gen),
            inputs=[model_name, num_processes_style],
            outputs=[info_style],
        )
        train_btn.click(
            second_elem_of(train), inputs=[model_name], outputs=[info_train]
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-name",
        type=str,
        default=None,
        help="Server name for Gradio app",
    )
    parser.add_argument(
        "--no-autolaunch",
        action="store_true",
        default=False,
        help="Do not launch app automatically",
    )
    args = parser.parse_args()

    app.launch(inbrowser=not args.no_autolaunch, server_name=args.server_name)
