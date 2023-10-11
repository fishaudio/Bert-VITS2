import os
import soundfile
import multiprocessing
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
import logging
import argparse
from pydub import AudioSegment

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)

os.environ["MODELSCOPE_CACHE"] = "./"
print(os.environ["SELECT_LANGUAGE"])
if "ZH(中文)" in os.environ["SELECT_LANGUAGE"]:
    inference_pipeline_zh = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
    )

elif "JP(日语)" in os.environ["SELECT_LANGUAGE"]:
    inference_pipeline_jp = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline'
    )


def transcribe_worker(file_path:str):
    """
    Worker function for transcribing a segment of an audio file.
    """
    if file_path.rstrip(".wav").endswith("_zh"):
        rec_result = inference_pipeline_zh(audio_in=file_path)
    elif file_path.rstrip(".wav").endswith("_jp"):
        rec_result = inference_pipeline_jp(audio_in=file_path)
    else:
        rec_result = {"text": ""}
    print(file_path)
    print(rec_result)
    return str(rec_result.get('text','')).strip()


def transcribe_folder_parallel(folder_path):
    """
    Transcribe all .wav files in the given folder using multiple processes.
    """
    max_duration = 60 # 最大持续时间（秒）
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                audio = AudioSegment.from_wav(file_path)
                duration_in_seconds = len(audio) / 1000  # 将毫秒转换为秒
                if duration_in_seconds <= max_duration + 1:
                    file_paths.append(file_path)

    transcriptions = list()
    for path in file_paths:
        transcriptions.append(transcribe_worker(path))

    for file_path, transcription in zip(file_paths, transcriptions):
        if transcription:
            lab_file_path = os.path.splitext(file_path)[0] + ".lab"
            with open(lab_file_path, "w", encoding="utf-8") as lab_file:
                lab_file.write(transcription)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filepath", default='./raw/lzy_zh', help="path of your model"
    )
    args = parser.parse_args()

    transcribe_folder_parallel(args.filepath)
