import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from style_bert_vits2.logging import logger


vad_model, utils = torch.hub.load(
    repo_or_dir="litagin02/silero-vad",
    model="silero_vad",
    onnx=True,
    trust_repo=True,
)

(get_speech_timestamps, _, read_audio, *_) = utils


def get_speech_ratio(audio_file):
    sampling_rate = 16000

    wav = read_audio(audio_file, sampling_rate=sampling_rate)
    speech_timestamps = get_speech_timestamps(
        wav, vad_model, sampling_rate=sampling_rate
    )

    speech_dur_ms = 0

    for ts in speech_timestamps:
        start_ms = ts["start"] / 16
        end_ms = ts["end"] / 16
        speech_dur_ms += end_ms - start_ms

    total_dur_ms = len(wav) / sampling_rate * 1000
    return speech_dur_ms / total_dur_ms


def process(file: Path):
    speech_ratio = get_speech_ratio(file)
    return file, speech_ratio


def main():
    parser = argparse.ArgumentParser(description="Calculate speech ratio.")
    parser.add_argument(
        "-i", "--input", help="Directory containing audio files", required=True
    )
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.input, "low_speech_ratio")):
        logger.info("Low speech ratio directory already exists, skipping...")
        exit(0)

    data_dir = Path(args.input)
    wav_files = list(data_dir.glob("*.wav"))
    wav_files.sort()

    if len(wav_files) < 100:
        logger.warning("Too few files, skipping...")
        exit(0)

    logger.info(f"Start VAD filtering for {data_dir}...")

    results = []

    for wav_file in tqdm(wav_files, file=sys.stdout):
        speech_ratio = get_speech_ratio(wav_file)
        results.append((wav_file, speech_ratio))

    results_df = pd.DataFrame(results, columns=["file", "speech_ratio"])
    results_df.to_csv(os.path.join(data_dir, "speech_ratio.csv"), index=False)

    logger.info(f"Speech ratio stats:\n{results_df['speech_ratio'].describe()}")
    threshold = 0.5

    low_speech_ratio_dir = os.path.join(data_dir, "low_speech_ratio")
    os.makedirs(low_speech_ratio_dir, exist_ok=True)

    low_speech_files = results_df[results_df["speech_ratio"] < threshold]["file"]
    logger.info(f"Moving {len(low_speech_files)} files to {low_speech_ratio_dir}...")
    for low_speech_file in low_speech_files:
        shutil.move(low_speech_file, low_speech_ratio_dir)
    logger.success("VAD filtering completed.")


if __name__ == "__main__":
    main()
