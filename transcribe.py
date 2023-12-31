import argparse
import os
import sys

from faster_whisper import WhisperModel
from tqdm import tqdm

from common.stdout_wrapper import SAFE_STDOUT


def transcribe(wav_path, initial_prompt=None):
    segments, _ = model.transcribe(
        wav_path, beam_size=5, language="ja", initial_prompt=initial_prompt
    )
    texts = [segment.text for segment in segments]
    return "".join(texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="raw")
    parser.add_argument("--output_file", type=str, default="esd.list")
    parser.add_argument(
        "--initial_prompt", type=str, default="こんにちは。元気、ですかー？私は……ちゃんと元気だよ！"
    )
    parser.add_argument("--speaker_name", type=str, default=None, required=True)
    parser.add_argument("--model", type=str, default="large-v3")

    args = parser.parse_args()

    speaker_name = args.speaker_name

    input_dir = args.input_dir
    output_file = args.output_file
    initial_prompt = args.initial_prompt

    model = WhisperModel("large-v3", device="cuda", compute_type="bfloat16")

    wav_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".wav")
    ]
    if os.path.exists(output_file):
        print(f"{output_file}が存在するので、バックアップを{output_file}.bakに作成します。")
        if os.path.exists(output_file + ".bak"):
            print(f"{output_file}.bakも存在するので、削除します。")
            os.remove(output_file + ".bak")
        os.rename(output_file, output_file + ".bak")

    with open(output_file, "w", encoding="utf-8") as f:
        for wav_file in tqdm(wav_files, file=SAFE_STDOUT):
            file_name = os.path.basename(wav_file)
            text = transcribe(wav_file, initial_prompt=initial_prompt)
            f.write(f"{file_name}|{speaker_name}|JP|{text}\n")
