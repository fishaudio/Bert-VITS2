# Usage: .venv/bin/python convert_onnx.py --model model_assets/koharune-ami/koharune-ami.safetensors
#        .venv/bin/python convert_onnx.py --model model_assets/ (All models in the directory will be converted)
# https://github.com/tuna2134/sbv2-api/blob/main/scripts/convert/convert_model.py を参考に実装した

# MIT License
# Copyright (c) 2024 tuna2134
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import time
import uuid
from argparse import ArgumentParser
from pathlib import Path
from typing import BinaryIO, cast

import onnx
import torch
from onnxsim import model_info, simplify
from rich import print
from rich.rule import Rule
from rich.style import Style

from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.models.infer import get_text
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.tts_model import TTSModel


def generate_aivm_metadata(
    hyper_parameters_file: BinaryIO,
    style_vectors_file: BinaryIO,
    model_file_name: str,
    model_uuid: uuid.UUID,
):
    try:
        import aivmlib
        from aivmlib.schemas.aivm_manifest import ModelArchitecture
    except ImportError:
        raise ImportError(
            "aivmlib is not installed. Please install it using `pip install aivmlib`."
        )

    # AIVM メタデータを生成
    metadata = aivmlib.generate_aivm_metadata(
        # 実際に JP-Extra かどうかはハイパーパラメータの値を元に自動判定されるので、ここでは JP-Extra を指定
        ModelArchitecture.StyleBertVITS2JPExtra,
        hyper_parameters_file,
        style_vectors_file,
    )

    # モデルファイル名からエポック数とステップ数を抽出
    epoch_match = re.search(r"e(\d{2,})", model_file_name)  # "e" の後ろに2桁以上の数字
    step_match = re.search(r"s(\d{2,})", model_file_name)  # "s" の後ろに2桁以上の数字

    # エポック数を設定
    if epoch_match:
        metadata.manifest.training_epochs = int(epoch_match.group(1))
    # ステップ数を設定
    if step_match:
        metadata.manifest.training_steps = int(step_match.group(1))

    # UUID を設定
    metadata.manifest.uuid = model_uuid

    return metadata


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path to the model file or directory"
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Already converted models will be overwritten",
    )
    parser.add_argument(
        "--aivm",
        action="store_true",
        help="Generate AIVM file from Safetensors model",
    )
    parser.add_argument(
        "--aivmx",
        action="store_true",
        help="Generate AIVMX file from ONNX model",
    )
    args = parser.parse_args()

    # --model に指定されたパスがディレクトリの時、配下にある全ての .safetensors ファイルを対象に変換する
    model_paths: list[Path] = []
    if Path(args.model).is_dir():
        for path in Path(args.model).glob("**/*.safetensors"):
            # . から始まるファイルは除外
            if not path.name.startswith("."):
                model_paths.append(path)
    else:
        model_paths.append(Path(args.model))

    for model_path in model_paths:

        # モデルの入出力先ファイルパスを取得
        onnx_temp_model_path = model_path.parent / f"{model_path.stem}_temp.onnx"
        onnx_optimized_model_path = model_path.parent / f"{model_path.stem}.onnx"
        config_path = model_path.parent / "config.json"
        style_vec_path = model_path.parent / "style_vectors.npy"
        aivm_path = model_path.parent / f"{model_path.stem}.aivm"
        aivmx_path = model_path.parent / f"{model_path.stem}.aivmx"
        assert model_path.exists(), "Model file does not exist"
        assert config_path.exists(), "Config file does not exist"
        assert style_vec_path.exists(), "Style vector file does not exist"
        assert model_path.suffix != ".onnx", "Model file is already ONNX"
        print(Rule(characters="=", style=Style(color="blue")))
        print(f"[bold cyan]Model file:[/bold cyan] {model_path}")
        print(f"[bold cyan]Config file:[/bold cyan] {config_path}")
        print(f"[bold cyan]Style vector file:[/bold cyan] {style_vec_path}")
        print(Rule(characters="=", style=Style(color="blue")))

        # すでに ONNX モデルが存在する場合、--force-convert オプションが指定されていない場合はスキップ
        if onnx_optimized_model_path.exists() and not args.force_convert:
            print(
                f"[bold yellow]ONNX model already exists: {onnx_optimized_model_path}[/bold yellow]"
            )
            print(
                "[bold]If you want to overwrite it, use the --force-convert option.[/bold]"
            )
            print(Rule(characters="=", style=Style(color="blue")))

        # ONNX モデルが存在しない場合、ONNX モデルを生成
        else:
            # PyTorch モデルを読み込む
            device = "cpu"
            tts_model = TTSModel(
                model_path=model_path,
                config_path=config_path,
                style_vec_path=style_vec_path,
                device=device,
            )
            tts_model.load()
            if DEFAULT_STYLE in tts_model.style2id:
                style_id = tts_model.style2id[DEFAULT_STYLE]
            else:
                style_id = 0  # 通常デフォルトスタイルのインデックスは 0 になる
            assert tts_model.net_g is not None, "Model is not loaded"

            # 音声合成に必要な BERT 特徴量・音素列・アクセント列・言語 ID を取得
            # JP-Extra モデルアーキテクチャの場合、bert (中国語の BERT 特徴量) や en_bert (英語の BERT 特徴量) は
            # torch.zeros() で適当に埋められており、推論には ja_bert (日本語の BERT 特徴量) のみが使用される
            bert, ja_bert, en_bert, phones, tones, lang_ids = get_text(
                "今日はいい天気ですね。",
                Languages.JP,
                tts_model.hyper_parameters,
                device,
                assist_text=None,
                assist_text_weight=DEFAULT_ASSIST_TEXT_WEIGHT,
                given_phone=None,
                given_tone=None,
            )

            # スタイルベクトルを取得
            style_vector = tts_model.get_style_vector(style_id, DEFAULT_STYLE_WEIGHT)

            # モデルの入力を作成
            x_tst = phones.to(device).unsqueeze(0)
            tones = tones.to(device).unsqueeze(0)
            lang_ids = lang_ids.to(device).unsqueeze(0)
            bert = bert.to(device).unsqueeze(0)
            ja_bert = ja_bert.to(device).unsqueeze(0)
            en_bert = en_bert.to(device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
            style_vec_tensor = torch.from_numpy(style_vector).to(device).unsqueeze(0)
            sid = 0
            sid_tensor = torch.LongTensor([sid]).to(device)
            length_scale = torch.tensor(1.0)
            sdp_ratio = torch.tensor(0.0)
            noise_scale = torch.tensor(0.667)
            noise_scale_w = torch.tensor(0.8)

            # JP-Extra モデルアーキテクチャ向けの ONNX 変換ロジック
            if isinstance(tts_model.net_g, SynthesizerTrnJPExtra):

                # SynthesizerTrnJPExtra の forward メソッドをオーバーライド
                def forward_jp_extra(
                    x: torch.Tensor,
                    x_lengths: torch.Tensor,
                    sid: torch.Tensor,
                    tone: torch.Tensor,
                    language: torch.Tensor,
                    bert: torch.Tensor,
                    style_vec: torch.Tensor,
                    length_scale: float = 1.0,
                    sdp_ratio: float = 0.0,
                    noise_scale: float = 0.667,
                    noise_scale_w: float = 0.8,
                ) -> tuple[
                    torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]
                ]:
                    return cast(SynthesizerTrnJPExtra, tts_model.net_g).infer(
                        x,
                        x_lengths,
                        sid,
                        tone,
                        language,
                        bert,
                        style_vec,
                        length_scale=length_scale,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                    )

                tts_model.net_g.forward = forward_jp_extra  # type: ignore

                # モデルを ONNX に変換
                print(Rule(characters="=", style=Style(color="blue")))
                print(
                    "[bold cyan]Exporting ONNX model... (Architecture: JP-Extra)[/bold cyan]"
                )
                print(Rule(characters="=", style=Style(color="blue")))
                export_start_time = time.time()
                torch.onnx.export(
                    model=tts_model.net_g,
                    args=(
                        x_tst,
                        x_tst_lengths,
                        sid_tensor,
                        tones,
                        lang_ids,
                        ja_bert,
                        style_vec_tensor,
                        length_scale,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                    ),
                    f=str(onnx_temp_model_path),
                    verbose=False,
                    input_names=[
                        "x_tst",
                        "x_tst_lengths",
                        "sid",
                        "tones",
                        "language",
                        "bert",
                        "style_vec",
                        "length_scale",
                        "sdp_ratio",
                        "noise_scale",
                        "noise_scale_w",
                    ],
                    output_names=["output"],
                    dynamic_axes={
                        "x_tst": {0: "batch_size", 1: "x_tst_max_length"},
                        "x_tst_lengths": {0: "batch_size"},
                        "sid": {0: "batch_size"},
                        "tones": {0: "batch_size", 1: "x_tst_max_length"},
                        "language": {0: "batch_size", 1: "x_tst_max_length"},
                        "bert": {0: "batch_size", 2: "x_tst_max_length"},
                        "style_vec": {0: "batch_size"},
                    },
                )
                print(
                    f"[bold green]ONNX model exported to {onnx_temp_model_path} ({time.time() - export_start_time:.2f}s)[/bold green]"
                )

            # 非 JP-Extra モデルアーキテクチャ向けの ONNX 変換ロジック
            else:

                # SynthesizerTrn の forward メソッドをオーバーライド
                def forward_non_jp_extra(
                    x: torch.Tensor,
                    x_lengths: torch.Tensor,
                    sid: torch.Tensor,
                    tone: torch.Tensor,
                    language: torch.Tensor,
                    bert: torch.Tensor,
                    ja_bert: torch.Tensor,
                    en_bert: torch.Tensor,
                    style_vec: torch.Tensor,
                    length_scale: float = 1.0,
                    sdp_ratio: float = 0.0,
                    noise_scale: float = 0.667,
                    noise_scale_w: float = 0.8,
                ) -> tuple[
                    torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]
                ]:
                    return cast(SynthesizerTrn, tts_model.net_g).infer(
                        x,
                        x_lengths,
                        sid,
                        tone,
                        language,
                        bert,
                        ja_bert,
                        en_bert,
                        style_vec,
                        length_scale=length_scale,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                    )

                tts_model.net_g.forward = forward_non_jp_extra  # type: ignore

                # モデルを ONNX に変換
                print(Rule(characters="=", style=Style(color="blue")))
                print(
                    "[bold cyan]Exporting ONNX model... (Architecture: Non-JP-Extra)[/bold cyan]"
                )
                print(Rule(characters="=", style=Style(color="blue")))
                export_start_time = time.time()
                torch.onnx.export(
                    model=tts_model.net_g,
                    args=(
                        x_tst,
                        x_tst_lengths,
                        sid_tensor,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        en_bert,
                        style_vec_tensor,
                        length_scale,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                    ),
                    f=str(onnx_temp_model_path),
                    verbose=False,
                    input_names=[
                        "x_tst",
                        "x_tst_lengths",
                        "sid",
                        "tones",
                        "language",
                        "bert",
                        "ja_bert",
                        "en_bert",
                        "style_vec",
                        "length_scale",
                        "sdp_ratio",
                        "noise_scale",
                        "noise_scale_w",
                    ],
                    output_names=["output"],
                    dynamic_axes={
                        "x_tst": {0: "batch_size", 1: "x_tst_max_length"},
                        "x_tst_lengths": {0: "batch_size"},
                        "sid": {0: "batch_size"},
                        "tones": {0: "batch_size", 1: "x_tst_max_length"},
                        "language": {0: "batch_size", 1: "x_tst_max_length"},
                        "bert": {0: "batch_size", 2: "x_tst_max_length"},
                        "ja_bert": {0: "batch_size", 2: "x_tst_max_length"},
                        "en_bert": {0: "batch_size", 2: "x_tst_max_length"},
                        "style_vec": {0: "batch_size"},
                    },
                )
                print(
                    f"[bold green]ONNX model exported to {onnx_temp_model_path} ({time.time() - export_start_time:.2f}s)[/bold green]"
                )

            # ONNX モデルを最適化
            print(Rule(characters="=", style=Style(color="blue")))
            print("[bold cyan]Optimizing ONNX model...[/bold cyan]")
            print(Rule(characters="=", style=Style(color="blue")))
            optimize_start_time = time.time()
            onnx_model = onnx.load(onnx_temp_model_path)
            simplified_onnx_model, check = simplify(onnx_model)
            onnx.save(simplified_onnx_model, onnx_optimized_model_path)
            print(
                f"[bold green]ONNX model optimized and saved to {onnx_optimized_model_path} ({time.time() - optimize_start_time:.2f}s)[/bold green]"
            )
            print(
                f"[bold]Total Time: {time.time() - start_time:.2f}s / "
                f"Size: {onnx_temp_model_path.stat().st_size / 1000 / 1000:.2f}MB -> "
                f"{onnx_optimized_model_path.stat().st_size / 1000 / 1000:.2f}MB[/bold]"
            )
            onnx_temp_model_path.unlink()
            print(Rule(characters="=", style=Style(color="blue")))
            print("[bold cyan]Optimized model info:[/bold cyan]")
            model_info.print_simplifying_info(onnx_model, simplified_onnx_model)
            print(Rule(characters="=", style=Style(color="blue")))

        # AIVM/AIVMX ファイルを生成
        if args.aivm or args.aivmx:
            try:
                import aivmlib
            except ImportError:
                raise ImportError(
                    "aivmlib is not installed. Please install it using `pip install aivmlib`."
                )

            # 共通の UUID を生成
            model_uuid = uuid.uuid4()

            # AIVM メタデータを生成
            with config_path.open("rb") as hyper_parameters_file:
                with style_vec_path.open("rb") as style_vectors_file:
                    aivm_metadata = generate_aivm_metadata(
                        hyper_parameters_file,
                        style_vectors_file,
                        model_path.name,
                        model_uuid,
                    )

            # AIVM ファイルを生成
            if args.aivm and (not aivm_path.exists() or args.force_convert):
                print("[bold cyan]Generating AIVM file...[/bold cyan]")
                print(Rule(characters="=", style=Style(color="blue")))
                with model_path.open("rb") as safetensors_file:
                    new_aivm_file_content = aivmlib.write_aivm_metadata(safetensors_file, aivm_metadata)  # fmt: skip
                    with aivm_path.open("wb") as f:
                        f.write(new_aivm_file_content)
                print(f"[bold green]Generated AIVM file: {aivm_path}[/bold green]")
                print(Rule(characters="=", style=Style(color="blue")))

            # AIVMX ファイルを生成
            if args.aivmx and (not aivmx_path.exists() or args.force_convert):
                print("[bold cyan]Generating AIVMX file...[/bold cyan]")
                print(Rule(characters="=", style=Style(color="blue")))
                with onnx_optimized_model_path.open("rb") as onnx_file:
                    new_aivmx_file_content = aivmlib.write_aivmx_metadata(onnx_file, aivm_metadata)  # fmt: skip
                    with aivmx_path.open("wb") as f:
                        f.write(new_aivmx_file_content)
                print(f"[bold green]Generated AIVMX file: {aivmx_path}[/bold green]")
                print(Rule(characters="=", style=Style(color="blue")))
