from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os

"""
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='./Model/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
)

rec_result = inference_pipeline(audio_in='ge_1570_2.wav')
print(rec_result)
# {'text': '欢迎大家来体验达摩院推出的语音识别模型'}
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parent_dir", default="./raw/巴老师")
parser.add_argument(
    "-n",
    "--character_name",
    default="巴老师",
    help="人物名，对应到dataset的地址映射",
)
parser.add_argument(
    "-tp", "--target_path", default="./filelists/baTeacher/raw_barbara.list"
)
args = parser.parse_args()
parent_dir = args.parent_dir
character_name = args.character_name
target_path = args.target_path
local_dir_model = "./Model/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

dir_path = os.path.join(*target_path.split("/")[:-1])


if not os.path.exists(dir_path):
    print(f"{dir_path} 不存在, 已创建...")
    os.makedirs(dir_path)


complete_list = []
filelist = list(os.listdir(parent_dir))
# print(filelist)
if os.path.exists(target_path):
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            pt = line.strip().split("|")[0]
            complete_list.append(pt)
# print(complete_list)

inference_pipeline = pipeline(task=Tasks.auto_speech_recognition, model=local_dir_model)

from tqdm import tqdm

for file in tqdm(filelist):
    if file[-3:] != "wav":
        tqdm.write(f"{file} not supported, ignoring...\n")
        continue
    tqdm.write(f"transcribing {parent_dir +'/'+ file}...\n")
    if not character_name:
        character_name = file.rstrip(".wav").split("_")[0]
    savepth = "./raw/" + character_name + "/" + file

    if savepth in complete_list:
        tqdm.write(f"{file} is already done, skip!")
        continue

    rec_result = inference_pipeline(audio_in=os.path.join(parent_dir, file))

    if "text" not in rec_result:
        tqdm.write("Text is not recognized，ignoring...\n")
        continue

    annos_text = rec_result["text"]
    annos_text = "[ZH]" + annos_text.replace("\n", "") + "[ZH]"
    annos_text = annos_text + "\n"
    line1 = savepth + "|" + character_name + "|" + annos_text
    line2 = savepth + "|" + character_name + "|ZH|" + rec_result["text"] + "\n"
    # with open("./long_character_anno.txt", 'a', encoding='utf-8') as f:
    #     f.write(line1)
    with open(target_path, "w+", encoding="utf-8") as f:
        f.write(line2)
    tqdm.write(rec_result["text"])
print("Done!\n")
