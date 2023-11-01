# 根据清洗后的list转录文件，将dataset中的冗余语音数据删除,raw中的还在
# 如果不小心弄错，可以通过resame重采样得到dataset

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--list_path", default="./filelists/baTeacher/raw_baTeacher.list"
)
parser.add_argument(
    "-fp", "--final_list_path", default="./filelists/baTeacher/baTeacher.list"
)
args = parser.parse_args()
list_path = args.list_path

# 最终得到dataset版转录文件
final_list_path = args.final_list_path


if os.path.exists(list_path):
    with open(list_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
# 首先整理list文件，按照wav进行排序
lines = sorted(
    lines, key=lambda x: int(x.split("|")[0].split("/")[-1].split("_")[-1][:-4])
)


print("删除raw中冗余语音数据")
base_path = "/".join(lines[0].split("|")[0].split("/")[:-1])
base_path2 = base_path.replace("raw", "dataset")
# list文件中存储的file路径
files_path = [line.split("|")[0] for line in lines]
print(f"数据地址为:{base_path}")
from tqdm import tqdm

for f in tqdm(os.listdir(base_path)):
    remove_file = os.path.join(base_path, f)
    if remove_file not in files_path:
        tqdm.write(f"成功删除 : {remove_file}")
        os.remove(remove_file)


print("开始整合数据，填补空余的index位")
# 获取基准名称和位置
file = files_path[0].split("/")[-1][:-4]
base_name, base_index = file.split("_")
base_index = int(base_index)
index = base_index
f_list = []
f_list_res = []
for i, line in enumerate(tqdm(lines)):
    old_name = line.split("|")[0].split("/")[-1]
    if not os.path.exists(os.path.join(base_path, old_name)):
        continue
    new_name = f"{base_name}_{index }.wav"
    index += 1
    # 写入raw list文件
    f_list.append("|".join([os.path.join(base_path, new_name)] + line.split("|")[1:]))
    # 写入dataset list文件
    f_list_res.append(
        "|".join([os.path.join(base_path2, new_name)] + line.split("|")[1:])
    )
    # 重命名文件
    if not os.path.exists(os.path.join(base_path, new_name)):
        tqdm.write(f"{old_name} 重命名为：{new_name}")
        os.rename(os.path.join(base_path, old_name), os.path.join(base_path, new_name))


with open(list_path, "w+", encoding="utf-8") as f1, open(
    final_list_path, "w+", encoding="utf-8"
) as f2:
    f1.write("".join(f_list))
    f2.write("".join(f_list_res))
print("f{list_path} successfully!")
print("f{final_list_path} successfully!")
