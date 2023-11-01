import os
import re

# 对转录文件进行初步清洗，去除非中文字符，去除长度小于5的句子，然后下一步应该进行proprecess预处理

new_annos = []
cleaned_new_annos = []
name = "baTeacher/raw_baTeacher.list"
base = "./filelists"

path = os.path.join(base, name)
out_path = os.path.join(base, name + ".clean")
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        long_character_anno = f.readlines()
        new_annos += long_character_anno
else:
    print("文件 cannot be found, please confirm that the path is correct")
    exit()
chinese_re = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9，。、；‘’“”！？&#8203;``【oaicite:0】``&#8203;（）《》：\-_—\[\]{}()<>.,;:\'\"!?]"
)

for line in new_annos:
    try:
        results = line.split("|")
        path, name, lang = results[0], results[1], results[2]
        text = " ".join(results[3:])
    except Exception as e:
        print(f"Error : {e}")
        print(line)
        s
    text += "\n" if not text.endswith("\n") else ""
    if len(text) >= 5:
        chinese_chars = re.findall(chinese_re, text)
        cleaned_text = "".join(chinese_chars)
        if cleaned_text:
            cleaned_new_annos.append(
                path + "|" + name + "|" + lang + "|" + cleaned_text + "\n"
            )
        else:
            print(f"Skip non-kanji text : {cleaned_text}")

    else:
        print(f"skip too short wav : {text}")


with open(out_path, "w", encoding="utf-8") as f:
    for line in cleaned_new_annos:
        f.write(line)

print("完成! 保存为 clean_barbara.list")
