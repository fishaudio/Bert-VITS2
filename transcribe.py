import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from torch.utils.data import Dataset
from tqdm import tqdm

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


# faster-whisperは並列処理しても速度が向上しないので、単一モデルでループ処理する
def transcribe_with_faster_whisper(
    model: "WhisperModel",
    audio_file: Path,
    initial_prompt: Optional[str] = None,
    language: str = "ja",
    num_beams: int = 1,
    no_repeat_ngram_size: int = 10,
):
    segments, _ = model.transcribe(
        str(audio_file),
        beam_size=num_beams,
        language=language,
        initial_prompt=initial_prompt,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    texts = [segment.text for segment in segments]
    return "".join(texts)


# HF pipelineで進捗表示をするために必要なDatasetクラス
class StrListDataset(Dataset[str]):
    def __init__(self, original_list: list[str]) -> None:
        self.original_list = original_list

    def __len__(self) -> int:
        return len(self.original_list)

    def __getitem__(self, i: int) -> str:
        return self.original_list[i]


# HFのWhisperはファイルリストを与えるとバッチ処理ができて速い
def transcribe_files_with_hf_whisper(
    audio_files: list[Path],
    model_id: str,
    initial_prompt: Optional[str] = None,
    language: str = "ja",
    batch_size: int = 16,
    num_beams: int = 1,
    no_repeat_ngram_size: int = 10,
    device: str = "cuda",
    pbar: Optional[tqdm] = None,
) -> list[str]:
    import torch
    from transformers import WhisperProcessor, pipeline

    processor: WhisperProcessor = WhisperProcessor.from_pretrained(model_id)
    generate_kwargs: dict[str, Any] = {
        "language": language,
        "do_sample": False,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    logger.info(f"generate_kwargs: {generate_kwargs}")

    if initial_prompt is not None:
        prompt_ids: torch.Tensor = processor.get_prompt_ids(
            initial_prompt, return_tensors="pt"
        )
        prompt_ids = prompt_ids.to(device)
        generate_kwargs["prompt_ids"] = prompt_ids

    pipe = pipeline(
        model=model_id,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=batch_size,
        torch_dtype=torch.float16,
        device="cuda",
        generate_kwargs=generate_kwargs,
    )
    dataset = StrListDataset([str(f) for f in audio_files])

    results: list[str] = []
    for whisper_result in pipe(dataset):
        text: str = whisper_result["text"]
        # なぜかテキストの最初に" {initial_prompt}"が入るので、文字の最初からこれを削除する
        # cf. https://github.com/huggingface/transformers/issues/27594
        if text.startswith(f" {initial_prompt}"):
            text = text[len(f" {initial_prompt}") :]
        results.append(text)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return results


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
    parser.add_argument("--use_hf_whisper", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=10)
    args = parser.parse_args()

    with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
        path_config: dict[str, str] = yaml.safe_load(f.read())
        dataset_root = Path(path_config["dataset_root"])

    model_name = str(args.model_name)

    input_dir = dataset_root / model_name / "raw"
    output_file = dataset_root / model_name / "esd.list"
    initial_prompt: str = args.initial_prompt
    initial_prompt = initial_prompt.strip('"')
    language: str = args.language
    device: str = args.device
    compute_type: str = args.compute_type
    batch_size: int = args.batch_size
    num_beams: int = args.num_beams
    no_repeat_ngram_size: int = args.no_repeat_ngram_size

    output_file.parent.mkdir(parents=True, exist_ok=True)

    wav_files = [f for f in input_dir.rglob("*.wav") if f.is_file()]
    wav_files = sorted(wav_files, key=lambda x: x.name)

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

    if not args.use_hf_whisper:
        from faster_whisper import WhisperModel

        logger.info(
            f"Loading faster-whisper model ({args.model}) with compute_type={compute_type}"
        )
        try:
            model = WhisperModel(args.model, device=device, compute_type=compute_type)
        except ValueError as e:
            logger.warning(f"Failed to load model, so use `auto` compute_type: {e}")
            model = WhisperModel(args.model, device=device)
        for wav_file in tqdm(wav_files, file=SAFE_STDOUT):
            text = transcribe_with_faster_whisper(
                model=model,
                audio_file=wav_file,
                initial_prompt=initial_prompt,
                language=language,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{wav_file.name}|{model_name}|{language_id}|{text}\n")
    else:
        model_id = f"openai/whisper-{args.model}"
        logger.info(f"Loading HF Whisper model ({model_id})")
        pbar = tqdm(total=len(wav_files), file=SAFE_STDOUT)
        results = transcribe_files_with_hf_whisper(
            audio_files=wav_files,
            model_id=model_id,
            initial_prompt=initial_prompt,
            language=language,
            batch_size=batch_size,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=device,
            pbar=pbar,
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for wav_file, text in zip(wav_files, results):
                f.write(f"{wav_file.name}|{model_name}|{language_id}|{text}\n")

    sys.exit(0)
