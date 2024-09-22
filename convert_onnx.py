# usage: .venv/bin/python convert_onnx.py --model model_assets/koharune-ami/koharune-ami.safetensors
# usage: .venv/bin/python convert_onnx.py --model model_assets/ (All models in the directory will be converted)
# ref: https://github.com/tuna2134/sbv2-api/blob/main/convert/convert_model.py

import time
from argparse import ArgumentParser
from pathlib import Path
from typing import cast

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
            continue

        # PyTorch モデルを読み込む
        device = "cpu"
        tts_model = TTSModel(
            model_path=model_path,
            config_path=config_path,
            style_vec_path=style_vec_path,
            device=device,
        )
        tts_model.load()
        style_id = tts_model.style2id[DEFAULT_STYLE]
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
                f"[bold cyan]Exporting ONNX model... (Architecture: JP-Extra)[/bold cyan]"
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
                f"[bold cyan]Exporting ONNX model... (Architecture: Non-JP-Extra)[/bold cyan]"
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
        print(f"[bold cyan]Optimizing ONNX model...[/bold cyan]")
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
