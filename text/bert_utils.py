import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from config import config


MIRROR: str = config.mirror


def _check_bert(repo_id, files, local_path, mirror):
    for file in files:
        if not Path(local_path).joinpath(file).exists():
            if mirror.lower() == "openi":
                import openi

                kwargs = {"token": config.openi_token} if config.openi_token else {}
                openi.login(**kwargs)
                openi.model.download_model(
                    "Stardust_minus/Bert-VITS2", repo_id.split("/")[-1], "./bert"
                )
            else:
                hf_hub_download(
                    repo_id, file, local_dir=local_path, local_dir_use_symlinks=False
                )


def check_bert_models():
    with open("./bert/bert_models.json", "r") as fp:
        models = json.load(fp)
        for k, v in models.items():
            local_path = Path("./bert").joinpath(k)
            _check_bert(v["repo_id"], v["files"], local_path, MIRROR)
