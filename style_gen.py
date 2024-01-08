import argparse
import concurrent.futures
import warnings

import numpy as np
import torch
from tqdm import tqdm

import utils
from common.log import logger
from common.stdout_wrapper import SAFE_STDOUT
from config import config

warnings.filterwarnings("ignore", category=UserWarning)
from pyannote.audio import Inference, Model

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")
device = torch.device(config.style_gen_config.device)
inference.to(device)


# 推論時にインポートするために短いが関数を書く
def get_style_vector(wav_path):
    return inference(wav_path)


def save_style_vector(wav_path):
    try:
        style_vec = get_style_vector(wav_path)
    except Exception as e:
        logger.error(f"\nError occurred with file: {wav_path}, Details:\n{e}\n")
        raise
    np.save(f"{wav_path}.npy", style_vec)  # `test.wav` -> `test.wav.npy`
    return style_vec


def save_average_style_vector(style_vectors, filename="style_vectors.npy"):
    average_vector = np.mean(style_vectors, axis=0)
    np.save(filename, average_vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.style_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes", type=int, default=config.style_gen_config.num_processes
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    num_processes = args.num_processes

    hps = utils.get_hparams_from_file(config_path)

    device = config.style_gen_config.device

    lines = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    wavnames = [line.split("|")[0] for line in lines]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        list(
            tqdm(
                executor.map(save_style_vector, wavnames),
                total=len(wavnames),
                file=SAFE_STDOUT,
            )
        )

    logger.info(f"Finished generating style vectors! total: {len(wavnames)} npy files.")
