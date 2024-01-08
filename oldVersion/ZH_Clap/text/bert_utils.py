from pathlib import Path

from huggingface_hub import hf_hub_download

from config import config


MIRROR: str = config.mirror


def _check_bert(repo_id, files, local_path):
    for file in files:
        if not Path(local_path).joinpath(file).exists():
            if MIRROR.lower() == "openi":
                import openi

                openi.model.download_model(
                    "Stardust_minus/Bert-VITS2", repo_id.split("/")[-1], "./bert"
                )
            else:
                hf_hub_download(
                    repo_id, file, local_dir=local_path, local_dir_use_symlinks=False
                )
