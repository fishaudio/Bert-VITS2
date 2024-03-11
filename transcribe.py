import argparse
import os
import sys
from pathlib import Path

import yaml
from faster_whisper import WhisperModel
from tqdm import tqdm

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


def transcribe(wav_path: Path, initial_prompt=None, language="ja"):
    segments, _ = model.transcribe(
        str(wav_path), beam_size=5, language=language, initial_prompt=initial_prompt
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
        dataset_root = Path(path_config["dataset_root"])

    model_name = str(args.model_name)

    input_dir = dataset_root / model_name / "raw"
    output_file = dataset_root / model_name / "esd.list"
    initial_prompt = args.initial_prompt
    language = args.language
    device = args.device
    compute_type = args.compute_type

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Loading Whisper model ({args.model}) with compute_type={compute_type}"
    )
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except ValueError as e:
        logger.warning(f"Failed to load model, so use `auto` compute_type: {e}")
        model = WhisperModel(args.model, device=device)

    wav_files = [f for f in input_dir.rglob("*.wav") if f.is_file()]
    if output_file.exists():
        logger.warning(f"{output_file} exists, backing up to {output_file}.bak")
        backup_path = output_file.with_name(output_file.name + ".bak")
        if backup_path.exists():
            logger.warning(f"{output_file}.bak exists, deleting...")
            backup_path.unlink()
        output_file.rename(backup_path)

    if language == "ja":
        language_id = Languages.JP.value
    elif language == "en":
        language_id = Languages.EN.value
    elif language == "zh":
        language_id = Languages.ZH.value
    else:
        raise ValueError(f"{language} is not supported.")

    wav_files = sorted(wav_files, key=lambda x: x.name)

    for wav_file in tqdm(wav_files, file=SAFE_STDOUT):
        text = transcribe(wav_file, initial_prompt=initial_prompt, language=language)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{wav_file.name}|{model_name}|{language_id}|{text}\n")
    sys.exit(0)
