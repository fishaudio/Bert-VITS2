import json
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import gradio as gr
import yaml

from config import get_path_config
from style_bert_vits2.constants import GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
from style_bert_vits2.utils.subprocess import run_script_with_log, second_elem_of


logger_handler = None
tensorboard_executed = False

path_config = get_path_config()
dataset_root = path_config.dataset_root


@dataclass
class PathsForPreprocess:
    dataset_path: Path
    esd_path: Path
    train_path: Path
    val_path: Path
    config_path: Path


def get_path(model_name: str) -> PathsForPreprocess:
    assert model_name != "", "モデル名は空にできません"
    dataset_path = dataset_root / model_name
    esd_path = dataset_path / "esd.list"
    train_path = dataset_path / "train.list"
    val_path = dataset_path / "val.list"
    config_path = dataset_path / "config.json"
    return PathsForPreprocess(dataset_path, esd_path, train_path, val_path, config_path)


def initialize(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    freeze_EN_bert: bool,
    freeze_JP_bert: bool,
    freeze_ZH_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    use_jp_extra: bool,
    log_interval: int,
):
    global logger_handler
    paths = get_path(model_name)

    # 前処理のログをファイルに保存する
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"preprocess_{timestamp}.log"
    if logger_handler is not None:
        logger.remove(logger_handler)
    logger_handler = logger.add(paths.dataset_path / file_name)

    logger.info(
        f"Step 1: start initialization...\nmodel_name: {model_name}, batch_size: {batch_size}, epochs: {epochs}, save_every_steps: {save_every_steps}, freeze_ZH_bert: {freeze_ZH_bert}, freeze_JP_bert: {freeze_JP_bert}, freeze_EN_bert: {freeze_EN_bert}, freeze_style: {freeze_style}, freeze_decoder: {freeze_decoder}, use_jp_extra: {use_jp_extra}"
    )

    default_config_path = (
        "configs/config.json" if not use_jp_extra else "configs/config_jp_extra.json"
    )

    with open(default_config_path, encoding="utf-8") as f:
        config = json.load(f)
    config["model_name"] = model_name
    config["data"]["training_files"] = str(paths.train_path)
    config["data"]["validation_files"] = str(paths.val_path)
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs
    config["train"]["eval_interval"] = save_every_steps
    config["train"]["log_interval"] = log_interval

    config["train"]["freeze_EN_bert"] = freeze_EN_bert
    config["train"]["freeze_JP_bert"] = freeze_JP_bert
    config["train"]["freeze_ZH_bert"] = freeze_ZH_bert
    config["train"]["freeze_style"] = freeze_style
    config["train"]["freeze_decoder"] = freeze_decoder

    config["train"]["bf16_run"] = False  # デフォルトでFalseのはずだが念のため

    # 今はデフォルトであるが、以前は非JP-Extra版になくバグの原因になるので念のため
    config["data"]["use_jp_extra"] = use_jp_extra

    model_path = paths.dataset_path / "models"
    if model_path.exists():
        logger.warning(
            f"Step 1: {model_path} already exists, so copy it to backup to {model_path}_backup"
        )
        shutil.copytree(
            src=model_path,
            dst=paths.dataset_path / "models_backup",
            dirs_exist_ok=True,
        )
        shutil.rmtree(model_path)
    pretrained_dir = Path("pretrained" if not use_jp_extra else "pretrained_jp_extra")
    try:
        shutil.copytree(
            src=pretrained_dir,
            dst=model_path,
        )
    except FileNotFoundError:
        logger.error(f"Step 1: {pretrained_dir} folder not found.")
        return False, f"Step 1, Error: {pretrained_dir}フォルダが見つかりません。"

    with open(paths.config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if not Path("config.yml").exists():
        shutil.copy(src="default_config.yml", dst="config.yml")
    with open("config.yml", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = str(paths.dataset_path)
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)
    logger.success("Step 1: initialization finished.")
    return True, "Step 1, Success: 初期設定が完了しました"


def resample(model_name: str, normalize: bool, trim: bool, num_processes: int):
    logger.info("Step 2: start resampling...")
    dataset_path = get_path(model_name).dataset_path
    input_dir = dataset_path / "raw"
    output_dir = dataset_path / "wavs"
    cmd = [
        "resample.py",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
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
        logger.error("Step 2: resampling failed.")
        return False, f"Step 2, Error: 音声ファイルの前処理に失敗しました:\n{message}"
    elif message:
        logger.warning("Step 2: resampling finished with stderr.")
        return True, f"Step 2, Success: 音声ファイルの前処理が完了しました:\n{message}"
    logger.success("Step 2: resampling finished.")
    return True, "Step 2, Success: 音声ファイルの前処理が完了しました"


def preprocess_text(
    model_name: str, use_jp_extra: bool, val_per_lang: int, yomi_error: str
):
    logger.info("Step 3: start preprocessing text...")
    paths = get_path(model_name)
    if not paths.esd_path.exists():
        logger.error(f"Step 3: {paths.esd_path} not found.")
        return (
            False,
            f"Step 3, Error: 書き起こしファイル {paths.esd_path} が見つかりません。",
        )

    cmd = [
        "preprocess_text.py",
        "--config-path",
        str(paths.config_path),
        "--transcription-path",
        str(paths.esd_path),
        "--train-path",
        str(paths.train_path),
        "--val-path",
        str(paths.val_path),
        "--val-per-lang",
        str(val_per_lang),
        "--yomi_error",
        yomi_error,
        "--correct_path",  # 音声ファイルのパスを正しいパスに修正する
    ]
    if use_jp_extra:
        cmd.append("--use_jp_extra")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error("Step 3: preprocessing text failed.")
        return (
            False,
            f"Step 3, Error: 書き起こしファイルの前処理に失敗しました:\n{message}",
        )
    elif message:
        logger.warning("Step 3: preprocessing text finished with stderr.")
        return (
            True,
            f"Step 3, Success: 書き起こしファイルの前処理が完了しました:\n{message}",
        )
    logger.success("Step 3: preprocessing text finished.")
    return True, "Step 3, Success: 書き起こしファイルの前処理が完了しました"


def bert_gen(model_name: str):
    logger.info("Step 4: start bert_gen...")
    config_path = get_path(model_name).config_path
    success, message = run_script_with_log(
        ["bert_gen.py", "--config", str(config_path)]
    )
    if not success:
        logger.error("Step 4: bert_gen failed.")
        return False, f"Step 4, Error: BERT特徴ファイルの生成に失敗しました:\n{message}"
    elif message:
        logger.warning("Step 4: bert_gen finished with stderr.")
        return (
            True,
            f"Step 4, Success: BERT特徴ファイルの生成が完了しました:\n{message}",
        )
    logger.success("Step 4: bert_gen finished.")
    return True, "Step 4, Success: BERT特徴ファイルの生成が完了しました"


def style_gen(model_name: str, num_processes: int):
    logger.info("Step 5: start style_gen...")
    config_path = get_path(model_name).config_path
    success, message = run_script_with_log(
        [
            "style_gen.py",
            "--config",
            str(config_path),
            "--num_processes",
            str(num_processes),
        ]
    )
    if not success:
        logger.error("Step 5: style_gen failed.")
        return (
            False,
            f"Step 5, Error: スタイル特徴ファイルの生成に失敗しました:\n{message}",
        )
    elif message:
        logger.warning("Step 5: style_gen finished with stderr.")
        return (
            True,
            f"Step 5, Success: スタイル特徴ファイルの生成が完了しました:\n{message}",
        )
    logger.success("Step 5: style_gen finished.")
    return True, "Step 5, Success: スタイル特徴ファイルの生成が完了しました"


def preprocess_all(
    model_name: str,
    batch_size: int,
    epochs: int,
    save_every_steps: int,
    num_processes: int,
    normalize: bool,
    trim: bool,
    freeze_EN_bert: bool,
    freeze_JP_bert: bool,
    freeze_ZH_bert: bool,
    freeze_style: bool,
    freeze_decoder: bool,
    use_jp_extra: bool,
    val_per_lang: int,
    log_interval: int,
    yomi_error: str,
):
    if model_name == "":
        return False, "Error: モデル名を入力してください"
    success, message = initialize(
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        save_every_steps=save_every_steps,
        freeze_EN_bert=freeze_EN_bert,
        freeze_JP_bert=freeze_JP_bert,
        freeze_ZH_bert=freeze_ZH_bert,
        freeze_style=freeze_style,
        freeze_decoder=freeze_decoder,
        use_jp_extra=use_jp_extra,
        log_interval=log_interval,
    )
    if not success:
        return False, message
    success, message = resample(
        model_name=model_name,
        normalize=normalize,
        trim=trim,
        num_processes=num_processes,
    )
    if not success:
        return False, message

    success, message = preprocess_text(
        model_name=model_name,
        use_jp_extra=use_jp_extra,
        val_per_lang=val_per_lang,
        yomi_error=yomi_error,
    )
    if not success:
        return False, message
    success, message = bert_gen(
        model_name=model_name
    )  # bert_genは重いのでプロセス数いじらない
    if not success:
        return False, message
    success, message = style_gen(model_name=model_name, num_processes=num_processes)
    if not success:
        return False, message
    logger.success("Success: All preprocess finished!")
    return (
        True,
        "Success: 全ての前処理が完了しました。ターミナルを確認しておかしいところがないか確認するのをおすすめします。",
    )


def train(
    model_name: str,
    skip_style: bool = False,
    use_jp_extra: bool = True,
    speedup: bool = False,
    not_use_custom_batch_sampler: bool = False,
):
    paths = get_path(model_name)
    # 学習再開の場合を考えて念のためconfig.ymlの名前等を更新
    with open("config.yml", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = str(paths.dataset_path)
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)

    train_py = "train_ms.py" if not use_jp_extra else "train_ms_jp_extra.py"
    cmd = [
        train_py,
        "--config",
        str(paths.config_path),
        "--model",
        str(paths.dataset_path),
    ]
    if skip_style:
        cmd.append("--skip_default_style")
    if speedup:
        cmd.append("--speedup")
    if not_use_custom_batch_sampler:
        cmd.append("--not_use_custom_batch_sampler")
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        logger.error("Train failed.")
        return False, f"Error: 学習に失敗しました:\n{message}"
    elif message:
        logger.warning("Train finished with stderr.")
        return True, f"Success: 学習が完了しました:\n{message}"
    logger.success("Train finished.")
    return True, "Success: 学習が完了しました"


def wait_for_tensorboard(port: int = 6006, timeout: float = 10):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True  # ポートが開いている場合
        except OSError:
            pass  # ポートがまだ開いていない場合

        if time.time() - start_time > timeout:
            return False  # タイムアウト

        time.sleep(0.1)


def run_tensorboard(model_name: str):
    global tensorboard_executed
    if not tensorboard_executed:
        python = sys.executable
        tensorboard_cmd = [
            python,
            "-m",
            "tensorboard.main",
            "--logdir",
            f"Data/{model_name}/models",
        ]
        subprocess.Popen(
            tensorboard_cmd,
            stdout=SAFE_STDOUT,  # type: ignore
            stderr=SAFE_STDOUT,  # type: ignore
        )
        yield gr.Button("起動中…")
        if wait_for_tensorboard():
            tensorboard_executed = True
        else:
            logger.error("Tensorboard did not start in the expected time.")
    webbrowser.open("http://localhost:6006")
    yield gr.Button("Tensorboardを開く")


change_log_md = """
**Ver 2.5以降の変更点**

- `raw/`フォルダの中で音声をサブディレクトリに分けて配置することで、自動的にスタイルが作成されるようになりました。詳細は下の「使い方/データの前準備」を参照してください。
- これまでは1ファイルあたり14秒程度を超えた音声ファイルは学習には用いられていませんでしたが、Ver 2.5以降では「カスタムバッチサンプラーを無効化」にチェックを入れることでその制限が無しに学習できるようになりました（デフォルトはオフ）。ただし:
    - 音声ファイルが長い場合の学習効率は悪いかもしれず、挙動も確認していません
    - チェックを入れると要求VRAMがかなり増えるようので、学習に失敗したりVRAM不足になる場合は、バッチサイズを小さくするか、チェックを外してください
"""

how_to_md = """
## 使い方

- データを準備して、モデル名を入力して、必要なら設定を調整してから、「自動前処理を実行」ボタンを押してください。進捗状況等はターミナルに表示されます。

- 各ステップごとに実行する場合は「手動前処理」を使ってください（基本的には自動でいいはず）。

- 前処理が終わったら、「学習を開始する」ボタンを押すと学習が開始されます。

- 途中から学習を再開する場合は、モデル名を入力してから「学習を開始する」を押せばよいです。

## JP-Extra版について

元とするモデル構造として [Bert-VITS2 Japanese-Extra](https://github.com/fishaudio/Bert-VITS2/releases/tag/JP-Exta) を使うことができます。
日本語のアクセントやイントネーションや自然性が上がる傾向にありますが、英語と中国語は話せなくなります。
"""

prepare_md = """
まず音声データと、書き起こしテキストを用意してください。

それを次のように配置します。
```
├── Data/
│   ├── {モデルの名前}
│   │   ├── esd.list
│   │   ├── raw/
│   │   │   ├── foo.wav
│   │   │   ├── bar.mp3
│   │   │   ├── style1/
│   │   │   │   ├── baz.wav
│   │   │   │   ├── qux.wav
│   │   │   ├── style2/
│   │   │   │   ├── corge.wav
│   │   │   │   ├── grault.wav
...
```

### 配置の仕方
- 上のように配置すると、`style1/`と`style2/`フォルダの内部（直下以外も含む）に入っている音声ファイルたちから、自動的にデフォルトスタイルに加えて`style1`と`style2`というスタイルが作成されます
- 特にスタイルを作る必要がない場合や、スタイル分類機能等でスタイルを作る場合は、`raw/`フォルダ直下に全てを配置してください。このように`raw/`のサブディレクトリの個数が0または1の場合は、スタイルはデフォルトスタイルのみが作成されます。
- 音声ファイルのフォーマットはwav形式以外にもmp3等の多くの音声ファイルに対応しています

### 書き起こしファイル`esd.list`

`Data/{モデルの名前}/esd.list` ファイルには、以下のフォーマットで各音声ファイルの情報を記述してください。


```
path/to/audio.wav(wavファイル以外でもこう書く)|{話者名}|{言語ID、ZHかJPかEN}|{書き起こしテキスト}
```

- ここで、最初の`path/to/audio.wav`は、`raw/`からの相対パスです。つまり、`raw/foo.wav`の場合は`foo.wav`、`raw/style1/bar.wav`の場合は`style1/bar.wav`となります。
- 拡張子がwavでない場合でも、`esd.list`には`wav`と書いてください、つまり、`raw/bar.mp3`の場合でも`bar.wav`と書いてください。


例：
```
foo.wav|hanako|JP|こんにちは、元気ですか？
bar.wav|taro|JP|はい、聞こえています……。何か用ですか？
style1/baz.wav|hanako|JP|今日はいい天気ですね。
style1/qux.wav|taro|JP|はい、そうですね。
...
english_teacher.wav|Mary|EN|How are you? I'm fine, thank you, and you?
...
```
もちろん日本語話者の単一話者データセットでも構いません。
"""


def create_train_app():
    with gr.Blocks(theme=GRADIO_THEME).queue() as app:
        gr.Markdown(change_log_md)
        with gr.Accordion("使い方", open=False):
            gr.Markdown(how_to_md)
            with gr.Accordion(label="データの前準備", open=False):
                gr.Markdown(prepare_md)
        model_name = gr.Textbox(label="モデル名")
        gr.Markdown("### 自動前処理")
        with gr.Row(variant="panel"):
            with gr.Column():
                use_jp_extra = gr.Checkbox(
                    label="JP-Extra版を使う（日本語の性能が上がるが英語と中国語は話せなくなる）",
                    value=True,
                )
                batch_size = gr.Slider(
                    label="バッチサイズ",
                    info="学習速度が遅い場合は小さくして試し、VRAMに余裕があれば大きくしてください。JP-Extra版でのVRAM使用量目安: 1: 6GB, 2: 8GB, 3: 10GB, 4: 12GB",
                    value=2,
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
                normalize = gr.Checkbox(
                    label="音声の音量を正規化する(音量の大小が揃っていない場合など)",
                    value=False,
                )
                trim = gr.Checkbox(
                    label="音声の最初と最後の無音を取り除く",
                    value=False,
                )
                yomi_error = gr.Radio(
                    label="書き起こしが読めないファイルの扱い",
                    choices=[
                        ("エラー出たらテキスト前処理が終わった時点で中断", "raise"),
                        ("読めないファイルは使わず続行", "skip"),
                        ("読めないファイルも無理やり読んで学習に使う", "use"),
                    ],
                    value="skip",
                )
                with gr.Accordion("詳細設定", open=False):
                    num_processes = gr.Slider(
                        label="プロセス数",
                        info="前処理時の並列処理プロセス数、前処理でフリーズしたら下げてください",
                        value=cpu_count() // 2,
                        minimum=1,
                        maximum=cpu_count(),
                        step=1,
                    )
                    val_per_lang = gr.Slider(
                        label="検証データ数",
                        info="学習には使われず、tensorboardで元音声と合成音声を比較するためのもの",
                        value=0,
                        minimum=0,
                        maximum=100,
                        step=1,
                    )
                    log_interval = gr.Slider(
                        label="Tensorboardのログ出力間隔",
                        info="Tensorboardで詳しく見たい人は小さめにしてください",
                        value=200,
                        minimum=10,
                        maximum=1000,
                        step=10,
                    )
                    gr.Markdown("学習時に特定の部分を凍結させるかどうか")
                    freeze_EN_bert = gr.Checkbox(
                        label="英語bert部分を凍結",
                        value=False,
                    )
                    freeze_JP_bert = gr.Checkbox(
                        label="日本語bert部分を凍結",
                        value=False,
                    )
                    freeze_ZH_bert = gr.Checkbox(
                        label="中国語bert部分を凍結",
                        value=False,
                    )
                    freeze_style = gr.Checkbox(
                        label="スタイル部分を凍結",
                        value=False,
                    )
                    freeze_decoder = gr.Checkbox(
                        label="デコーダ部分を凍結",
                        value=False,
                    )

            with gr.Column():
                preprocess_button = gr.Button(
                    value="自動前処理を実行", variant="primary"
                )
                info_all = gr.Textbox(label="状況")
        with gr.Accordion(open=False, label="手動前処理"):
            with gr.Row(variant="panel"):
                with gr.Column():
                    gr.Markdown(value="#### Step 1: 設定ファイルの生成")
                    use_jp_extra_manual = gr.Checkbox(
                        label="JP-Extra版を使う",
                        value=True,
                    )
                    batch_size_manual = gr.Slider(
                        label="バッチサイズ",
                        value=2,
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
                    log_interval_manual = gr.Slider(
                        label="Tensorboardのログ出力間隔",
                        value=200,
                        minimum=10,
                        maximum=1000,
                        step=10,
                    )
                    freeze_EN_bert_manual = gr.Checkbox(
                        label="英語bert部分を凍結",
                        value=False,
                    )
                    freeze_JP_bert_manual = gr.Checkbox(
                        label="日本語bert部分を凍結",
                        value=False,
                    )
                    freeze_ZH_bert_manual = gr.Checkbox(
                        label="中国語bert部分を凍結",
                        value=False,
                    )
                    freeze_style_manual = gr.Checkbox(
                        label="スタイル部分を凍結",
                        value=False,
                    )
                    freeze_decoder_manual = gr.Checkbox(
                        label="デコーダ部分を凍結",
                        value=False,
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
                    val_per_lang_manual = gr.Slider(
                        label="検証データ数",
                        value=0,
                        minimum=0,
                        maximum=100,
                        step=1,
                    )
                    yomi_error_manual = gr.Radio(
                        label="書き起こしが読めないファイルの扱い",
                        choices=[
                            ("エラー出たらテキスト前処理が終わった時点で中断", "raise"),
                            ("読めないファイルは使わず続行", "skip"),
                            ("読めないファイルも無理やり読んで学習に使う", "use"),
                        ],
                        value="raise",
                    )
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
        with gr.Row():
            skip_style = gr.Checkbox(
                label="スタイルファイルの生成をスキップする",
                info="学習再開の場合の場合はチェックしてください",
                value=False,
            )
            use_jp_extra_train = gr.Checkbox(
                label="JP-Extra版を使う",
                value=True,
            )
            not_use_custom_batch_sampler = gr.Checkbox(
                label="カスタムバッチサンプラーを無効化",
                info="VRAMに余裕がある場合にチェックすると、長い音声ファイルも学習に使われるようになります",
                value=False,
            )
            speedup = gr.Checkbox(
                label="ログ等をスキップして学習を高速化する",
                value=False,
                visible=False,  # Experimental
            )
            train_btn = gr.Button(value="学習を開始する", variant="primary")
            tensorboard_btn = gr.Button(value="Tensorboardを開く")
        gr.Markdown(
            "進捗はターミナルで確認してください。結果は指定したステップごとに随時保存されており、また学習を途中から再開することもできます。学習を終了するには単にターミナルを終了してください。"
        )
        info_train = gr.Textbox(label="状況")

        preprocess_button.click(
            second_elem_of(preprocess_all),
            inputs=[
                model_name,
                batch_size,
                epochs,
                save_every_steps,
                num_processes,
                normalize,
                trim,
                freeze_EN_bert,
                freeze_JP_bert,
                freeze_ZH_bert,
                freeze_style,
                freeze_decoder,
                use_jp_extra,
                val_per_lang,
                log_interval,
                yomi_error,
            ],
            outputs=[info_all],
        )

        # Manual preprocess
        generate_config_btn.click(
            second_elem_of(initialize),
            inputs=[
                model_name,
                batch_size_manual,
                epochs_manual,
                save_every_steps_manual,
                freeze_EN_bert_manual,
                freeze_JP_bert_manual,
                freeze_ZH_bert_manual,
                freeze_style_manual,
                freeze_decoder_manual,
                use_jp_extra_manual,
                log_interval_manual,
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
            inputs=[
                model_name,
                use_jp_extra_manual,
                val_per_lang_manual,
                yomi_error_manual,
            ],
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

        # Train
        train_btn.click(
            second_elem_of(train),
            inputs=[
                model_name,
                skip_style,
                use_jp_extra_train,
                speedup,
                not_use_custom_batch_sampler,
            ],
            outputs=[info_train],
        )
        tensorboard_btn.click(
            run_tensorboard, inputs=[model_name], outputs=[tensorboard_btn]
        )

        use_jp_extra.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_jp_extra],
            outputs=[use_jp_extra_train],
        )
        use_jp_extra_manual.change(
            lambda x: gr.Checkbox(value=x),
            inputs=[use_jp_extra_manual],
            outputs=[use_jp_extra_train],
        )

    return app


if __name__ == "__main__":
    app = create_train_app()
    app.launch(inbrowser=True)
