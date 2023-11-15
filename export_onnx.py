from models_onnx import SynthesizerTrn
import utils
from text.symbols import symbols
import os
import json


def export_onnx(export_path, model_path, config_path):
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    net_g.export_onnx(export_path)

    spklist = []
    for key in hps.data.spk2id.keys():
        spklist.append(key)

    MoeVSConf = {
        "Folder": f"{export_path}",
        "Name": f"{export_path}",
        "Type": "BertVits",
        "Symbol": symbols,
        "Cleaner": "",
        "Rate": hps.data.sampling_rate,
        "CharaMix": True,
        "Characters": spklist,
        "LanguageMap": {"ZH": [0, 0], "JP": [1, 6], "EN": [2, 8]},
        "Dict": "BasicDict",
        "BertPath": [
            "chinese-roberta-wwm-ext-large",
            "deberta-v2-large-japanese",
            "bert-base-japanese-v3",
        ],
    }

    with open(f"onnx/{export_path}.json", "w") as MoeVsConfFile:
        json.dump(MoeVSConf, MoeVsConfFile, indent=4)


if __name__ == "__main__":
    print(symbols)
    export_path = "HimenoSena"
    model_path = "G_53000.pth"
    config_path = "config.json"
    if not os.path.exists("onnx"):
        os.makedirs("onnx")
    if not os.path.exists(f"onnx/{export_path}"):
        os.makedirs(f"onnx/{export_path}")
    export_onnx(export_path, model_path, config_path)
