import argparse
import os
from loguru import logger


def extract_list(folder_path, language, name, transcript_txt_file):
    logger.info(f"extracting list: {folder_path}|{name}|{language}")
    current_dir = os.getcwd()
    relative_path = os.path.relpath(folder_path, current_dir)
    print(relative_path)
    os.makedirs(os.path.dirname(transcript_txt_file), exist_ok=True)
    with open(transcript_txt_file, "w", encoding="utf-8") as f:
        # 遍历 raw 文件夹下的所有子文件夹
        for root, _, files in os.walk(relative_path):
            for file in files:
                if file.endswith(".lab"):
                    lab_file_path = os.path.join(root, file)
                    # 读取转写文本
                    with open(lab_file_path, "r", encoding="utf-8") as lab_file:
                        transcription = lab_file.read().strip()
                    if len(transcription) == 0:
                        continue
                    # 获取对应的 WAV 文件路径
                    # ./Data/宵宫/audios/raw
                    # ./Data/宵宫/audios/wavs
                    wav_file_path = os.path.splitext(lab_file_path)[0] + ".wav"
                    if os.path.isfile(wav_file_path):
                        wav_file_path = wav_file_path.replace("\\", "/").replace(
                            "/raw", "/wavs"
                        )
                        # 写入数据到总的转写文本文件
                        line = f"{wav_file_path}|{name}|{language}|{transcription}\n"
                        f.write(line)
                    else:
                        print("not exists!")
    return f"转写文本 {transcript_txt_file} 生成完成"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filepath",
        required=True,
        help="path of your rawaudios, e.g. ./Data/xxx/audios/raw",
    )
    parser.add_argument("-l", "--language", default="ZH", help="language")
    parser.add_argument("-n", "--name", required=True, help="name of the character")
    parser.add_argument("-o", "--outfile", required=True, help="outfile")
    args = parser.parse_args()

    status_str = extract_list(args.filepath, args.language, args.name, args.outfile)
    logger.critical(status_str)
