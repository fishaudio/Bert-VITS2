import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
import logging
import argparse
from pydub import AudioSegment
import concurrent.futures
from tqdm import tqdm

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)
os.environ["MODELSCOPE_CACHE"] = "./"


def transcribe_worker(file_path: str, inference_pipeline):
    """
    Worker function for transcribing a segment of an audio file.
    """
    rec_result = inference_pipeline(audio_in=file_path)
    text = str(rec_result.get("text", "")).strip()
    text_without_spaces = text.replace(" ", "")
    logger.critical(file_path)
    logger.critical("text: " + text_without_spaces)
    return text_without_spaces


def transcribe_folder_parallel(folder_path, language, max_workers=4):
    """
    Transcribe all .wav files in the given folder using ThreadPoolExecutor.
    """
    logger.critical(f"parallel transcribe: {folder_path}|{language}|{max_workers}")
    if language == "JP(日语)":
        workers = [
            pipeline(
                task=Tasks.auto_speech_recognition,
                model="damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline",
            )
            for _ in range(max_workers)
        ]

    else:
        workers = [
            pipeline(
                task=Tasks.auto_speech_recognition,
                model="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                model_revision="v1.2.4",
            )
            for _ in range(max_workers)
        ]
    max_duration = 60  # 最大持续时间（秒）
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                lab_file_path = os.path.splitext(file_path)[0] + ".lab"
                if os.path.exists(lab_file_path):
                    logger.info(lab_file_path + " 已存在")
                    continue
                audio = AudioSegment.from_wav(file_path)
                duration_in_seconds = len(audio) / 1000  # 将毫秒转换为秒
                if duration_in_seconds <= max_duration + 1:
                    file_paths.append(file_path)

    all_workers = (
        workers * (len(file_paths) // max_workers)
        + workers[: len(file_paths) % max_workers]
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(range(0, len(file_paths), max_workers), desc="转写进度: "):
            l, r = i, min(i + max_workers, len(file_paths))
            transcriptions = list(
                executor.map(transcribe_worker, file_paths[l:r], all_workers[l:r])
            )
            for file_path, transcription in zip(file_paths[l:r], transcriptions):
                if transcription:
                    lab_file_path = os.path.splitext(file_path)[0] + ".lab"
                    with open(lab_file_path, "w", encoding="utf-8") as lab_file:
                        lab_file.write(transcription)
    logger.critical("已经将wav文件转写为同名的.lab文件, 都在raw文件夹下")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filepath", default="./raw/lzy_zh", help="path of your model"
    )
    parser.add_argument("-l", "--language", default="ZH(中文)", help="language")
    parser.add_argument("-w", "--workers", default="4", help="trans workers")
    args = parser.parse_args()

    transcribe_folder_parallel(args.filepath, args.language, int(args.workers))
    print("转写结束！")
