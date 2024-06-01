import json
from pathlib import Path
from typing import Union

import numpy as np

from style_bert_vits2.constants import DEFAULT_STYLE
from style_bert_vits2.logging import logger


def save_neutral_vector(
    wav_dir: Union[Path, str],
    output_dir: Union[Path, str],
    config_path: Union[Path, str],
    config_output_path: Union[Path, str],
):
    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)
    embs = []
    for file in wav_dir.rglob("*.npy"):
        xvec = np.load(file)
        embs.append(np.expand_dims(xvec, axis=0))

    x = np.concatenate(embs, axis=0)  # (N, 256)
    mean = np.mean(x, axis=0)  # (256,)
    only_mean = np.stack([mean])  # (1, 256)
    np.save(output_dir / "style_vectors.npy", only_mean)
    logger.info(f"Saved mean style vector to {output_dir}")

    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = 1
    json_dict["data"]["style2id"] = {DEFAULT_STYLE: 0}
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved style config to {config_output_path}")


def save_styles_by_dirs(
    wav_dir: Union[Path, str],
    output_dir: Union[Path, str],
    config_path: Union[Path, str],
    config_output_path: Union[Path, str],
):
    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)
    config_output_path = Path(config_output_path)

    subdirs = [d for d in wav_dir.iterdir() if d.is_dir()]
    subdirs.sort()
    if len(subdirs) in (0, 1):
        logger.info(
            f"At least 2 subdirectories are required for generating style vectors with respect to them, found {len(subdirs)}."
        )
        logger.info("Generating only neutral style vector instead.")
        save_neutral_vector(wav_dir, output_dir, config_path, config_output_path)
        return

    # First get mean of all for Neutral
    embs = []
    for file in wav_dir.rglob("*.npy"):
        xvec = np.load(file)
        embs.append(np.expand_dims(xvec, axis=0))
    x = np.concatenate(embs, axis=0)  # (N, 256)
    mean = np.mean(x, axis=0)  # (256,)
    style_vectors = [mean]

    names = [DEFAULT_STYLE]
    for style_dir in subdirs:
        npy_files = list(style_dir.rglob("*.npy"))
        if not npy_files:
            continue
        embs = []
        for file in npy_files:
            xvec = np.load(file)
            embs.append(np.expand_dims(xvec, axis=0))

        x = np.concatenate(embs, axis=0)  # (N, 256)
        mean = np.mean(x, axis=0)  # (256,)
        style_vectors.append(mean)
        names.append(style_dir.name)

    # Stack them to make (num_styles, 256)
    style_vectors_npy = np.stack(style_vectors, axis=0)
    np.save(output_dir / "style_vectors.npy", style_vectors_npy)
    logger.info(f"Saved style vectors to {output_dir / 'style_vectors.npy'}")

    # Save style2id config to json
    style2id = {name: i for i, name in enumerate(names)}
    with open(config_path, encoding="utf-8") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = len(names)
    json_dict["data"]["style2id"] = style2id
    with open(config_output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved style config to {config_output_path}")
