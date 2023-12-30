import os
from tools.log import logger

import numpy as np
import json


def set_style_config(json_path, output_path):
    with open(json_path, "r") as f:
        json_dict = json.load(f)
    json_dict["data"]["num_styles"] = 1
    json_dict["data"]["style2id"] = {"Neutral": 0}
    with open(output_path, "w") as f:
        json.dump(json_dict, f, indent=2)
    logger.info(f"Update style config (only Neutral style) to {output_path}")


def save_mean_vector(wav_dir, output_path):
    embs = []
    for file in os.listdir(wav_dir):
        if file.endswith(".npy"):
            xvec = np.load(os.path.join(wav_dir, file))
            embs.append(np.expand_dims(xvec, axis=0))

    x = np.concatenate(embs, axis=0)  # (N, 256)
    mean = np.mean(x, axis=0)  # (256,)
    only_mean = np.stack([mean])  # (1, 256)
    np.save(output_path, only_mean)
    logger.info(f"Saved mean style vector to {output_path}")
