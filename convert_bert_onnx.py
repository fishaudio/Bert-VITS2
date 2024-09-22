# usage: .venv/bin/python convert_bert_onnx.py --language JP
# ref: https://github.com/tuna2134/sbv2-api/blob/main/convert/convert_deberta.py

import time
from argparse import ArgumentParser
from pathlib import Path

import onnx
import torch
from onnxsim import model_info, simplify
from rich import print
from rich.rule import Rule
from rich.style import Style
from torch import nn
from transformers.convert_slow_tokenizer import BertConverter

from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.nlp import bert_models


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser()
    parser.add_argument("--language", default=Languages.JP, help="Language of the BERT model to be converted")
    args = parser.parse_args()

    # モデルの入出力先ファイルパスを取得
    language = Languages(args.language)
    pretrained_model_name_or_path = DEFAULT_BERT_MODEL_PATHS[language]
    onnx_temp_model_path = Path(pretrained_model_name_or_path) / f"model_temp.onnx"
    onnx_optimized_model_path = Path(pretrained_model_name_or_path) / f"model.onnx"
    tokenizer_json_path = Path(pretrained_model_name_or_path) / "tokenizer.json"
    print(Rule(characters="=", style=Style(color="blue")))
    print(f"[bold cyan]Language:[/bold cyan] {language.name}")
    print(f"[bold cyan]Pretrained model:[/bold cyan] {pretrained_model_name_or_path}")
    print(Rule(characters="=", style=Style(color="blue")))

    # トークナイザーを Fast Tokenizer 用形式に変換して保存
    tokenizer = bert_models.load_tokenizer(language)
    converter = BertConverter(tokenizer)
    converter.converted().save(str(tokenizer_json_path))
    print(Rule(characters="=", style=Style(color="blue")))
    print(f"[bold green]Tokenizer JSON saved to {tokenizer_json_path}[/bold green]")
    print(Rule(characters="=", style=Style(color="blue")))

    # TODO: JP, ZH は変換できるが、EN は途中で強制終了されてしまい変換できない
    class ONNXBert(nn.Module):
        def __init__(self):
            super(ONNXBert, self).__init__()
            self.model = bert_models.load_model(language)

        def forward(self, input_ids, token_type_ids, attention_mask):
            inputs = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            }
            res = self.model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
            return res

    # ONNX 変換用の BERT モデルをロード
    model = ONNXBert()
    inputs = tokenizer("今日はいい天気ですね", return_tensors="pt")

    # モデルを ONNX に変換
    print(Rule(characters="=", style=Style(color="blue")))
    print(f"[bold cyan]Exporting ONNX model...[/bold cyan]")
    print(Rule(characters="=", style=Style(color="blue")))
    export_start_time = time.time()
    torch.onnx.export(
        model=model,
        args=(
            inputs["input_ids"],
            inputs["token_type_ids"],
            inputs["attention_mask"],
        ),
        f=str(onnx_temp_model_path),
        verbose=False,
        input_names=[
            "input_ids",
            "token_type_ids",
            "attention_mask",
        ],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {1: "batch_size"},
            "attention_mask": {1: "batch_size"},
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
