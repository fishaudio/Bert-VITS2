import os
import argparse
import librosa
from multiprocessing import Pool, cpu_count

import soundfile
from tqdm import tqdm

from config import config


def process(item):
    spkdir, wav_name, args = item
    wav_path = os.path.join(spkdir, wav_name)
    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.exists(wav_path) and ".wav" in wav_path:
        # print(wav_path)
        wav, sr = librosa.load(wav_path, sr=args.sr)
        soundfile.write(os.path.join(args.out_dir, wav_name), wav, sr)


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
    print(config.resample_config.sampling_rate)
    print(config.resample_config.in_dir)
    print(config.resample_config.out_dir)
    # autodl 无卡模式会识别出46个cpu
    if args.processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes = args.processes
    print("processes: ", processes)
    pool = Pool(processes=processes)

    spk_dir = args.in_dir

    tasks = []

    for dirpath, _, filenames in os.walk(spk_dir):
        for filename in filenames:
            if filename.endswith(".wav"):
                twople = (os.path.abspath(dirpath), filename, args)
                print(twople)
                tasks.append(twople)

    for _ in tqdm(
        pool.imap_unordered(process, tasks),
    ):
        pass

    pool.close()
    pool.join()

    print("音频重采样完毕!")
