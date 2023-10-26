import os
import argparse
import librosa
from multiprocessing import Pool, cpu_count

import soundfile
from tqdm import tqdm

from config import config


def process(item):
    spkdir, wav_name, args = item
    wav_path = os.path.join(args.in_dir, wav_name)
    if os.path.exists(wav_path) and ".wav" in wav_path:
        os.makedirs(args.out_dir, exist_ok=True)
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
    args, _ = parser.parse_known_args()
    print(config.resample_config.sampling_rate)
    print(config.resample_config.in_dir)
    print(config.resample_config.out_dir)
    # processes = 8
    processes = cpu_count() - 2 if cpu_count() > 4 else 1
    pool = Pool(processes=processes)
    
    spk_dir = args.in_dir
    if os.path.isdir(spk_dir):
        print(spk_dir)
        for _ in tqdm(
            pool.imap_unordered(
                process,
                [
                    (spk_dir, i, args)
                    for i in os.listdir(spk_dir)
                    if i.endswith("wav")
                ],
            )
        ):
            pass

    print("音频重采样完毕!")
