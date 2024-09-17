# usage: .venv/bin/python convert_bert_onnx.py --language JP
# ref: https://github.com/tuna2134/sbv2-api/blob/main/convert/convert_deberta.py

from argparse import ArgumentParser
from pathlib import Path

import onnx
import torch
from onnxsim import simplify
from torch import nn
from transformers.convert_slow_tokenizer import BertConverter

from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.nlp import bert_models


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--language", default=Languages.JP)
    args = parser.parse_args()

    # モデルの入出力先ファイルパスを取得
    language = Languages(args.language)
    pretrained_model_name_or_path = DEFAULT_BERT_MODEL_PATHS[language]
    onnx_temp_model_path = Path(pretrained_model_name_or_path) / f"model_temp.onnx"
    onnx_optimized_model_path = Path(pretrained_model_name_or_path) / f"model.onnx"
    tokenizer_json_path = Path(pretrained_model_name_or_path) / "tokenizer.json"

    # トークナイザーを Fast Tokenizer 用形式に変換して保存
    tokenizer = bert_models.load_tokenizer(language)
    converter = BertConverter(tokenizer)
    converter.converted().save(str(tokenizer_json_path))

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
    torch.onnx.export(
        model=model,
        args=(inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]),
        f=str(onnx_temp_model_path),
        input_names=["input_ids", "token_type_ids", "attention_mask"],
        output_names=["output"],
        verbose=True,
        dynamic_axes={
            "input_ids": {1: "batch_size"},
            "attention_mask": {1: "batch_size"},
        },
    )

    # ONNX モデルを最適化
    onnx_model = onnx.load(onnx_temp_model_path)
    simplified_onnx_model, check = simplify(onnx_model)
    onnx.save(simplified_onnx_model, onnx_optimized_model_path)

    # 最適化前の ONNX モデルを削除
    onnx_temp_model_path.unlink()
    print(f"ONNX model optimized and saved to {onnx_optimized_model_path}")
