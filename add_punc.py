import json
import os
import multiprocessing
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm
import logging

logging.getLogger("modelscope").setLevel(logging.ERROR)

metadata_path = "filelists/genshin.list"
output_path = "filelists/genshin_punc.list"

os.makedirs("tmp", exist_ok=True)


def split_list(lst, n):
    avg = len(lst) // n  # 每份的平均长度
    rem = len(lst) % n  # 剩余的长度

    result = []
    start = 0

    for i in range(n):
        end = start + avg + (1 if i < rem else 0)  # 计算当前份的结束位置
        result.append(lst[start:end])  # 切片并将结果添加到结果列表中
        start = end  # 更新下一份的起始位置

    return result


def process(audios, pid):
    print("loading model")
    inference_pipeline = pipeline(
        task=Tasks.punctuation,
        model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    )
    print("loaded model")

    total_cnt = 0
    skip_cnt = 0
    for audiofile in tqdm(audios):
        for k, sentence in enumerate(audiofile["segments"]):
            confidence = sentence["confidence"]
            if confidence >= 0.95:
                text = sentence["text"]
                rec_result = inference_pipeline(text_in=text)
                sentence["text"] = rec_result["text"]
                total_cnt += 1
            else:
                skip_cnt += 1
        print("preprocessed text count :", total_cnt, "skip count :", skip_cnt)
    with open(f"tmp/{pid}.json", "w") as f:
        json.dump(audios, f, ensure_ascii=False, indent=2)


def process_wrapper(args):
    audios, pid = args
    process(audios, pid)


if __name__ == "__main__":
    # Define the number of processes
    n_process = 8

    # Split the audios into chunks for multiprocessing
    with open(metadata_path, "r") as f:
        lines = f.readlines()
    audio_chunks = split_list(audios, n_process)

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=n_process)

    # Map the process function to each chunk of audios
    pool.map(process_wrapper, zip(audio_chunks, range(n_process)))

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Merge output files
    merged_data = []
