import argparse
import concurrent.futures
import os

from loguru import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm

os.environ["MODELSCOPE_CACHE"] = "./"


def transcribe_worker(file_path: str, inference_pipeline, language):
    """
    Worker function for transcribing a segment of an audio file.
    """
    rec_result = inference_pipeline(audio_in=file_path)
    text = str(rec_result.get("text", "")).strip()
    text_without_spaces = text.replace(" ", "")
    logger.info(file_path)
    if language != "EN":
        logger.info("text: " + text_without_spaces)
        return text_without_spaces
    else:
        logger.info("text: " + text)
        return text


def transcribe_folder_parallel(folder_path, language, max_workers=4):
    """
    Transcribe all .wav files in the given folder using ThreadPoolExecutor.
    """
    logger.critical(f"parallel transcribe: {folder_path}|{language}|{max_workers}")
    if language == "JP":
        workers = [
            pipeline(
                task=Tasks.auto_speech_recognition,
                model="damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline",
            )
            for _ in range(max_workers)
        ]

    elif language == "ZH":
        workers = [
            pipeline(
                task=Tasks.auto_speech_recognition,
                model="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                model_revision="v1.2.4",
            )
            for _ in range(max_workers)
        ]
    else:
        workers = [
            pipeline(
                task=Tasks.auto_speech_recognition,
                model="damo/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-offline",
            )
            for _ in range(max_workers)
        ]

    file_paths = []
    langs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                lab_file_path = os.path.splitext(file_path)[0] + ".lab"
                file_paths.append(file_path)
                langs.append(language)

    all_workers = (
        workers * (len(file_paths) // max_workers)
        + workers[: len(file_paths) % max_workers]
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(range(0, len(file_paths), max_workers), desc="转写进度: "):
            l, r = i, min(i + max_workers, len(file_paths))
            transcriptions = list(
                executor.map(
                    transcribe_worker, file_paths[l:r], all_workers[l:r], langs[l:r]
                )
            )
            for file_path, transcription in zip(file_paths[l:r], transcriptions):
                if transcription:
                    lab_file_path = os.path.splitext(file_path)[0] + ".lab"
                    with open(lab_file_path, "w", encoding="utf-8") as lab_file:
                        lab_file.write(transcription)
    logger.critical("已经将wav文件转写为同名的.lab文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filepath", default="./raw/lzy_zh", help="path of your model"
    )
    parser.add_argument("-l", "--language", default="ZH", help="language")
    parser.add_argument("-w", "--workers", default="1", help="trans workers")
    args = parser.parse_args()

    transcribe_folder_parallel(args.filepath, args.language, int(args.workers))
    print("转写结束！")
