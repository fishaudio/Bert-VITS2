import os
import gradio as gr

lang_dict = {"EN(英文)": "_en", "ZH(中文)": "_zh", "JP(日语)": "_jp"}


def raw_dir_convert_to_path(target_dir: str, lang):
    res = target_dir.rstrip("/").rstrip("\\")
    if (not target_dir.startswith("raw")) and (not target_dir.startswith("./raw")):
        res = os.path.join("./raw", res)
    if (
        (not res.endswith("_zh"))
        and (not res.endswith("_jp"))
        and (not res.endswith("_en"))
    ):
        res += lang_dict[lang]
    return res


def update_g_files():
    g_files = []
    cnt = 0
    for root, dirs, files in os.walk(os.path.abspath("./logs")):
        for file in files:
            if file.startswith("G_") and file.endswith(".pth"):
                g_files.append(os.path.join(root, file))
                cnt += 1
    print(g_files)
    return f"更新模型列表完成, 共找到{cnt}个模型", gr.Dropdown.update(choices=g_files)


def update_c_files():
    c_files = []
    cnt = 0
    for root, dirs, files in os.walk(os.path.abspath("./logs")):
        for file in files:
            if file.startswith("config.json"):
                c_files.append(os.path.join(root, file))
                cnt += 1
    print(c_files)
    return f"更新模型列表完成, 共找到{cnt}个配置文件", gr.Dropdown.update(choices=c_files)


def update_model_folders():
    subdirs = []
    cnt = 0
    for root, dirs, files in os.walk(os.path.abspath("./logs")):
        for dir_name in dirs:
            if os.path.basename(dir_name) != "eval":
                subdirs.append(os.path.join(root, dir_name))
                cnt += 1
    print(subdirs)
    return f"更新模型文件夹列表完成, 共找到{cnt}个文件夹", gr.Dropdown.update(choices=subdirs)


def update_wav_lab_pairs():
    wav_count = tot_count = 0
    for root, _, files in os.walk("./raw"):
        for file in files:
            # print(file)
            file_path = os.path.join(root, file)
            if file.lower().endswith(".wav"):
                lab_file = os.path.splitext(file_path)[0] + ".lab"
                if os.path.exists(lab_file):
                    wav_count += 1
                tot_count += 1
    return f"{wav_count} / {tot_count}"


def update_raw_folders():
    subdirs = []
    cnt = 0
    script_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
    raw_path = os.path.join(script_path, "raw")
    print(raw_path)
    os.makedirs(raw_path, exist_ok=True)
    for root, dirs, files in os.walk(raw_path):
        for dir_name in dirs:
            relative_path = os.path.relpath(
                os.path.join(root, dir_name), script_path
            )  # 获取相对路径
            subdirs.append(relative_path)
            cnt += 1
    print(subdirs)
    return (
        f"更新raw音频文件夹列表完成, 共找到{cnt}个文件夹",
        gr.Dropdown.update(choices=subdirs),
        gr.Textbox.update(value=update_wav_lab_pairs()),
    )
