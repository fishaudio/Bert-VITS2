import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from common.log import logger


def download_bert_models():
    with open("bert/bert_models.json", "r") as fp:
        models = json.load(fp)
    for k, v in models.items():
        local_path = Path("bert").joinpath(k)
        for file in v["files"]:
            if not Path(local_path).joinpath(file).exists():
                logger.info(f"Downloading {k} {file}")
                hf_hub_download(
                    v["repo_id"],
                    file,
                    local_dir=local_path,
                    local_dir_use_symlinks=False,
                )


def download_slm_model():
    local_path = Path("slm/wavlm-base-plus/")
    file = "pytorch_model.bin"
    if not Path(local_path).joinpath(file).exists():
        logger.info(f"Downloading wavlm-base-plus {file}")
        hf_hub_download(
            "microsoft/wavlm-base-plus",
            file,
            local_dir=local_path,
            local_dir_use_symlinks=False,
        )


def download_pretrained_models():
    files = ["G_0.safetensors", "D_0.safetensors", "DUR_0.safetensors"]
    local_path = Path("pretrained")
    for file in files:
        if not Path(local_path).joinpath(file).exists():
            logger.info(f"Downloading pretrained {file}")
            hf_hub_download(
                "litagin/Style-Bert-VITS2-1.0-base",
                file,
                local_dir=local_path,
                local_dir_use_symlinks=False,
            )


def download_jp_extra_pretrained_models():
    files = ["G_0.safetensors", "D_0.safetensors", "WD_0.safetensors"]
    local_path = Path("pretrained_jp_extra")
    for file in files:
        if not Path(local_path).joinpath(file).exists():
            logger.info(f"Downloading JP-Extra pretrained {file}")
            hf_hub_download(
                "litagin/Style-Bert-VITS2-2.0-base-JP-Extra",
                file,
                local_dir=local_path,
                local_dir_use_symlinks=False,
            )


def download_jvnv_models():
    files = [
        "jvnv--jp/config.json",
        "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors",
        "jvnv-F1-jp/style_vectors.npy",
        "jvnv-F2-jp/config.json",
        "jvnv-F2-jp/jvnv-F2_e166_s20000.safetensors",
        "jvnv-F2-jp/style_vectors.npy",
        "jvnv-M1-jp/config.json",
        "jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors",
        "jvnv-M1-jp/style_vectors.npy",
        "jvnv-M2-jp/config.json",
        "jvnv-M2-jp/jvnv-M2-jp_e159_s17000.safetensors",
        "jvnv-M2-jp/style_vectors.npy",
    ]
    for file in files:
        if not Path(f"model_assets/{file}").exists():
            logger.info(f"Downloading {file}")
            hf_hub_download(
                "litagin/style_bert_vits2_jvnv",
                file,
                local_dir="model_assets",
                local_dir_use_symlinks=False,
            )


parser = argparse.ArgumentParser()
parser.add_argument("--skip_jvnv", action="store_true")
args = parser.parse_args()

download_bert_models()

download_slm_model()

download_pretrained_models()

download_jp_extra_pretrained_models()

if not args.skip_jvnv:
    download_jvnv_models()
