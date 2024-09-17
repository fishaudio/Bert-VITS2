# usage: .venv/bin/python convert_onnx.py --model model_assets/amitaro/amitaro.safetensors
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
from style_bert_vits2.models.models_jp_extra import (
    SynthesizerTrn as SynthesizerTrnJPExtra,
)
from style_bert_vits2.tts_model import TTSModel


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    # モデルの入出力先ファイルパスを取得
    model_path = Path(args.model)
    onnx_temp_model_path = Path(args.model).parent / f"{model_path.stem}_temp.onnx"
    onnx_optimized_model_path = Path(args.model).parent / f"{model_path.stem}.onnx"
    config_path = Path(args.model).parent / "config.json"
    style_vec_path = Path(args.model).parent / "style_vectors.npy"
    assert model_path.exists(), "Model file does not exist"
    assert config_path.exists(), "Config file does not exist"
    assert style_vec_path.exists(), "Style vector file does not exist"
    assert model_path.suffix != ".onnx", "Model file is already ONNX"
    print(Rule(characters="=", style=Style(color="blue")))
    print(f"[bold cyan]Model file:[/bold cyan] {model_path}")
    print(f"[bold cyan]Config file:[/bold cyan] {config_path}")
    print(f"[bold cyan]Style vector file:[/bold cyan] {style_vec_path}")
    print(Rule(characters="=", style=Style(color="blue")))

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
    assert (
        tts_model.hyper_parameters.data.use_jp_extra is True
    ), "Normal model is not supported yet"

    # SynthesizerTrnJPExtra の forward メソッドをオーバーライド
    def forward(
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: torch.Tensor,
        tone: torch.Tensor,
        language: torch.Tensor,
        bert: torch.Tensor,
        style_vec: torch.Tensor,
        length_scale: float = 1.0,
        sdp_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        return cast(SynthesizerTrnJPExtra, tts_model.net_g).infer(
            x,
            x_lengths,
            sid,
            tone,
            language,
            bert,
            style_vec,
            sdp_ratio=sdp_ratio,
            length_scale=length_scale,
        )

    tts_model.net_g.forward = forward  # type: ignore

    # 音声合成に必要な BERT 特徴量・音素列・アクセント列・言語 ID を取得
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

    x_tst = phones.to(device).unsqueeze(0)
    tones = tones.to(device).unsqueeze(0)
    lang_ids = lang_ids.to(device).unsqueeze(0)
    bert = bert.to(device).unsqueeze(0)
    ja_bert = ja_bert.to(device).unsqueeze(0)
    en_bert = en_bert.to(device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
    style_vec_tensor = torch.from_numpy(style_vector).to(device).unsqueeze(0)

    # モデルを ONNX に変換
    print(Rule(characters="=", style=Style(color="blue")))
    print(f"[bold cyan]Exporting ONNX model...[/bold cyan]")
    print(Rule(characters="=", style=Style(color="blue")))
    export_start_time = time.time()
    torch.onnx.export(
        model=tts_model.net_g,
        args=(
            x_tst,
            x_tst_lengths,
            torch.LongTensor([0]).to(device),
            tones,
            lang_ids,
            bert,
            style_vec_tensor,
            torch.tensor(1.0),
            torch.tensor(0.0),
        ),
        f=str(onnx_temp_model_path),
        verbose=True,
        dynamic_axes={
            "x_tst": {1: "batch_size"},
            "x_tst_lengths": {0: "batch_size"},
            "tones": {1: "batch_size"},
            "language": {1: "batch_size"},
            "bert": {2: "batch_size"},
        },
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
        ],
        output_names=["output"],
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
    onnx_temp_model_path.unlink()
    print(
        f"[bold green]ONNX model optimized and saved to {onnx_optimized_model_path} ({time.time() - optimize_start_time:.2f}s)[/bold green]"
    )
    print(
        f"[bold]Total Time: {time.time() - start_time:.2f}s / Size: {onnx_optimized_model_path.stat().st_size / 1024 / 1024:.2f}MB[/bold]"
    )
    print(Rule(characters="=", style=Style(color="blue")))
    print("[bold cyan]Optimized model info:[/bold cyan]")
    model_info.print_simplifying_info(onnx_model, simplified_onnx_model)
    print(Rule(characters="=", style=Style(color="blue")))
