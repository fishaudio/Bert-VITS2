import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count

import soundfile
from scipy.io import wavfile
from tqdm import tqdm


def process(item):
    spkdir, wav_name, args = item
    speaker = spkdir.replace("\\", "/").split("/")[-1]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and ".wav" in wav_path:
        os.makedirs(os.path.join(args.out_dir, speaker), exist_ok=True)
        wav, sr = librosa.load(wav_path, sr=args.sr)
        soundfile.write(os.path.join(args.out_dir, speaker, wav_name), wav, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=44100, help="sampling rate")
    parser.add_argument(
        "--in_dir", type=str, default="./raw", help="path to source dir"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./dataset", help="path to target dir"
    )
    args = parser.parse_args()
    # processs = 8
    processs = cpu_count() - 2 if cpu_count() > 4 else 1
    pool = Pool(processes=processs)

    for speaker in os.listdir(args.in_dir):
        spk_dir = os.path.join(args.in_dir, speaker)
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
