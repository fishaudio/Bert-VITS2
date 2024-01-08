import argparse
from multiprocessing import Pool, cpu_count

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

import utils
from config import config
from clap_wrapper import get_clap_audio_feature
import librosa
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def process_line(line):
    device = config.emo_gen_config.device
    if config.emo_gen_config.use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")

    clap_path = wav_path.replace(".WAV", ".wav").replace(".wav", ".emo.pt")
    if os.path.isfile(clap_path):
        return

    audio = librosa.load(wav_path, 48000)[0]
    # audio = librosa.resample(audio, 44100, 48000)

    clap = get_clap_audio_feature(audio, device)
    torch.save(clap, clap_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default=config.emo_gen_config.config_path
    )
    parser.add_argument(
        "--num_processes", type=int, default=config.emo_gen_config.num_processes
    )
    args, _ = parser.parse_known_args()
    config_path = args.config
    hps = utils.get_hparams_from_file(config_path)
    lines = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())
    if len(lines) != 0:
        num_processes = min(args.num_processes, cpu_count())
        with Pool(processes=num_processes) as pool:
            for _ in tqdm(pool.imap_unordered(process_line, lines), total=len(lines)):
                pass

    print(f"clap生成完毕!, 共有{len(lines)}个emo.pt生成!")
