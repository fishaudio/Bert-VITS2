import argparse
import os
import sys

import yaml
from faster_whisper import WhisperModel
from tqdm import tqdm

from common.constants import Languages
from common.log import logger
from common.stdout_wrapper import SAFE_STDOUT


def transcribe(wav_path, initial_prompt=None, language="ja"):
    segments, _ = model.transcribe(
        wav_path, beam_size=5, language=language, initial_prompt=initial_prompt
    )
    texts = [segment.text for segment in segments]
    return "".join(texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--initial_prompt",
        type=str,
        default="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
    )
    parser.add_argument(
        "--language", type=str, default="ja", choices=["ja", "en", "zh"]
    )
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute_type", type=str, default="bfloat16")

    args = parser.parse_args()

    with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
        path_config: dict[str, str] = yaml.safe_load(f.read())
        dataset_root = path_config["dataset_root"]

    model_name = args.model_name

    input_dir = os.path.join(dataset_root, model_name, "raw")
    output_file = os.path.join(dataset_root, model_name, "esd.list")
    initial_prompt = args.initial_prompt
    language = args.language
    device = args.device
    compute_type = args.compute_type

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    logger.info(
        f"Loading Whisper model ({args.model}) with compute_type={compute_type}"
    )
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except ValueError as e:
        logger.warning(f"Failed to load model, so use `auto` compute_type: {e}")
        model = WhisperModel(args.model, device=device)

    wav_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".wav")
    ]
    if os.path.exists(output_file):
        logger.warning(f"{output_file} exists, backing up to {output_file}.bak")
        if os.path.exists(output_file + ".bak"):
            logger.warning(f"{output_file}.bak exists, deleting...")
            os.remove(output_file + ".bak")
        os.rename(output_file, output_file + ".bak")

    if language == "ja":
        language_id = Languages.JP.value
    elif language == "en":
        language_id = Languages.EN.value
    elif language == "zh":
        language_id = Languages.ZH.value
    else:
        raise ValueError(f"{language} is not supported.")
    with open(output_file, "w", encoding="utf-8") as f:
        for wav_file in tqdm(wav_files, file=SAFE_STDOUT):
            file_name = os.path.basename(wav_file)
            text = transcribe(
                wav_file, initial_prompt=initial_prompt, language=language
            )
            f.write(f"{file_name}|{model_name}|{language_id}|{text}\n")
            f.flush()
    sys.exit(0)
