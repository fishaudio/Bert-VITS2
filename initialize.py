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
        "jvnv-F1/config.json",
        "jvnv-F1/jvnv-F1.safetensors",
        "jvnv-F1/style_vectors.npy",
        "jvnv-F2/config.json",
        "jvnv-F2/jvnv-F2.safetensors",
        "jvnv-F2/style_vectors.npy",
        "jvnv-M1/config.json",
        "jvnv-M1/jvnv-M1.safetensors",
        "jvnv-M1/style_vectors.npy",
        "jvnv-M2/config.json",
        "jvnv-M2/jvnv-M2.safetensors",
        "jvnv-M2/style_vectors.npy",
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


download_bert_models()

download_slm_model()

download_pretrained_models()

download_jp_extra_pretrained_models()

download_jvnv_models()
