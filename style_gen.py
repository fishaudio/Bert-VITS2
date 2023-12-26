import argparse
import concurrent.futures
import sys
import warnings

import numpy as np
import torch
from tqdm import tqdm

import utils
from config import config

warnings.filterwarnings("ignore", category=UserWarning)
from pyannote.audio import Inference, Model

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")
device = torch.device(config.style_gen_config.device)
inference.to(device)


def extract_style_vector(wav_path):
    return inference(wav_path)


def save_style_vector(wav_path):
    style_vec = extract_style_vector(wav_path)
    # `test.wav` -> `test.wav.npy`
    np.save(f"{wav_path}.npy", style_vec)


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
                file=sys.stdout,
            )
        )

    print(f"Finished generating style vectors! total: {len(wavnames)} npy files.")
