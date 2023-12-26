import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from tools.log import logger


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


download_bert_models()

download_pretrained_models()
