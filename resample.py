import argparse
import os
from multiprocessing import Pool, cpu_count

import librosa
import pyloudnorm as pyln
import soundfile
from tqdm import tqdm

from common.log import logger
from common.stdout_wrapper import SAFE_STDOUT
from config import config

DEFAULT_BLOCK_SIZE: float = 0.400  # seconds


class BlockSizeException(Exception):
    pass


def normalize_audio(data, sr):
    meter = pyln.Meter(sr, block_size=DEFAULT_BLOCK_SIZE)  # create BS.1770 meter
    try:
        loudness = meter.integrated_loudness(data)
    except ValueError as e:
        raise BlockSizeException(e)
    # logger.info(f"loudness: {loudness}")
    data = pyln.normalize.loudness(data, loudness, -23.0)
    return data


def process(item):
    spkdir, wav_name, args = item
    wav_path = os.path.join(args.in_dir, spkdir, wav_name)
    if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):
        wav, sr = librosa.load(wav_path, sr=args.sr)
        if args.normalize:
            try:
                wav = normalize_audio(wav, sr)
            except BlockSizeException:
                logger.info(
                    f"Skip normalize due to less than {DEFAULT_BLOCK_SIZE} second audio: {wav_path}"
                )
        if args.trim:
            wav, _ = librosa.effects.trim(wav, top_db=30)
        soundfile.write(os.path.join(args.out_dir, spkdir, wav_name), wav, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sr",
        type=int,
        default=config.resample_config.sampling_rate,
        help="sampling rate",
    )
    parser.add_argument(
        "--in_dir",
        "-i",
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default=config.resample_config.out_dir,
        help="path to target dir",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="cpu_processes",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="loudness normalize audio",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        default=False,
        help="trim silence (start and end only)",
    )
    args, _ = parser.parse_known_args()
    # autodl 无卡模式会识别出46个cpu
    if args.num_processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes = args.num_processes

    tasks = []

    for dirpath, _, filenames in os.walk(args.in_dir):
        # 子级目录
        spk_dir = os.path.relpath(dirpath, args.in_dir)
        spk_dir_out = os.path.join(args.out_dir, spk_dir)
        if not os.path.isdir(spk_dir_out):
            os.makedirs(spk_dir_out, exist_ok=True)
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                twople = (spk_dir, filename, args)
                tasks.append(twople)

    if len(tasks) == 0:
        logger.error(f"No wav files found in {args.in_dir}")
        raise ValueError(f"No wav files found in {args.in_dir}")

    pool = Pool(processes=processes)
    for _ in tqdm(
        pool.imap_unordered(process, tasks), file=SAFE_STDOUT, total=len(tasks)
    ):
        pass

    pool.close()
    pool.join()

    logger.info("Resampling Done!")
