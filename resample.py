import os
import argparse
import librosa
from multiprocessing import Pool, cpu_count

import soundfile
from tqdm import tqdm

from config import config


def process(item):
    spkdir, wav_name, args = item
    wav_path = os.path.join(args.in_dir, spkdir, wav_name)
    if os.path.exists(wav_path) and wav_path.lower().endswith(".wav"):
        wav, sr = librosa.load(wav_path, sr=args.sr)
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
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=config.resample_config.out_dir,
        help="path to target dir",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="cpu_processes",
    )
    args, _ = parser.parse_known_args()
    # autodl 无卡模式会识别出46个cpu
    if args.processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes = args.processes
    pool = Pool(processes=processes)

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

    for _ in tqdm(
        pool.imap_unordered(process, tasks),
    ):
        pass

    pool.close()
    pool.join()

    print("音频重采样完毕!")
