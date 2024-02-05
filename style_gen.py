import argparse
from concurrent.futures import ThreadPoolExecutor
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


class NaNValueError(ValueError):
    """カスタム例外クラス。NaN値が見つかった場合に使用されます。"""

    pass


# 推論時にインポートするために短いが関数を書く
def get_style_vector(wav_path):
    return inference(wav_path)


def save_style_vector(wav_path):
    try:
        style_vec = get_style_vector(wav_path)
    except Exception as e:
        print("\n")
        logger.error(f"Error occurred with file: {wav_path}, Details:\n{e}\n")
        raise
    # 値にNaNが含まれていると悪影響なのでチェックする
    if np.isnan(style_vec).any():
        print("\n")
        logger.warning(f"NaN value found in style vector: {wav_path}")
        raise NaNValueError(f"NaN value found in style vector: {wav_path}")
    np.save(f"{wav_path}.npy", style_vec)  # `test.wav` -> `test.wav.npy`


def process_line(line):
    wavname = line.split("|")[0]
    try:
        save_style_vector(wavname)
        return line, None
    except NaNValueError:
        return line, "nan_error"


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

    training_lines = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        training_lines.extend(f.readlines())
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        training_results = list(
            tqdm(
                executor.map(process_line, training_lines),
                total=len(training_lines),
                file=SAFE_STDOUT,
            )
        )
    ok_training_lines = [line for line, error in training_results if error is None]
    nan_training_lines = [
        line for line, error in training_results if error == "nan_error"
    ]
    if nan_training_lines:
        nan_files = [line.split("|")[0] for line in nan_training_lines]
        logger.warning(
            f"Found NaN value in {len(nan_training_lines)} files: {nan_files}, so they will be deleted from training data."
        )

    val_lines = []
    with open(hps.data.validation_files, encoding="utf-8") as f:
        val_lines.extend(f.readlines())

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        val_results = list(
            tqdm(
                executor.map(process_line, val_lines),
                total=len(val_lines),
                file=SAFE_STDOUT,
            )
        )
    ok_val_lines = [line for line, error in val_results if error is None]
    nan_val_lines = [line for line, error in val_results if error == "nan_error"]
    if nan_val_lines:
        nan_files = [line.split("|")[0] for line in nan_val_lines]
        logger.warning(
            f"Found NaN value in {len(nan_val_lines)} files: {nan_files}, so they will be deleted from validation data."
        )

    with open(hps.data.training_files, "w", encoding="utf-8") as f:
        f.writelines(ok_training_lines)

    with open(hps.data.validation_files, "w", encoding="utf-8") as f:
        f.writelines(ok_val_lines)

    ok_num = len(ok_training_lines) + len(ok_val_lines)

    logger.info(f"Finished generating style vectors! total: {ok_num} npy files.")
