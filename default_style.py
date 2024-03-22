import json
import os
from pathlib import Path
from typing import Union

import numpy as np

from style_bert_vits2.constants import DEFAULT_STYLE
from style_bert_vits2.logging import logger


def set_style_config(json_path: Path, output_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = 1
    json_dict["data"]["style2id"] = {DEFAULT_STYLE: 0}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Save style config (only {DEFAULT_STYLE}) to {output_path}")


def save_neutral_vector(wav_dir: Union[Path, str], output_path: Union[Path, str]):
    wav_dir = Path(wav_dir)
    output_path = Path(output_path)
    embs = []
    for file in wav_dir.rglob("*.npy"):
        xvec = np.load(file)
        embs.append(np.expand_dims(xvec, axis=0))

    x = np.concatenate(embs, axis=0)  # (N, 256)
    mean = np.mean(x, axis=0)  # (256,)
    only_mean = np.stack([mean])  # (1, 256)
    np.save(output_path, only_mean)
    logger.info(f"Saved mean style vector to {output_path}")
