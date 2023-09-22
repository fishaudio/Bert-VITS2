import argparse
import shutil

import gradio as gr
import zipfile
import os
import soundfile as sf
import webbrowser
import tempfile

temp_folder = tempfile.gettempdir()

taboo_symbols = "{<>}[]"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--share", default=False, help="make link public", action="store_true"
)
parser.add_argument(
    "--target_path", default="./raw/gongzi", help="target path to store wavs", type=str
)
args = parser.parse_args()
lang_dict = {"EN(英文)": "_en", "ZH(中文)": "_zh", "JP(日语)": "_jp"}
inv_lang_dict = {"_en": "EN", "_zh": "ZH", "_jp": "JP"}


def update_transcript_status(target_path):
    parent_folder = os.path.dirname(target_path)
    os.makedirs(parent_folder, exist_ok=True)
    # 获取所有子文件夹名称
    sub_folders = [str(f.name) for f in os.scandir(parent_folder) if f.is_dir()]
    print("所有角色: ", sub_folders)
    sub_folders_str = ""
    for it in sub_folders:
        sub_folders_str += str(it) + "\n"
    print(sub_folders_str)
    return sub_folders_str


def process_uploaded_files(file, max_wav_len, taboo_symbols, target_path, lang):
    result_str = ""
    rm_files = []
    zip_files = []
    target_path += lang_dict[lang]
    print("target_path: ", target_path)
    try:
        os.makedirs(target_path, exist_ok=True)
    except:
        return "非法的目标文件路径，请重新输入", ""

    if not os.path.exists(target_path):
        return "非法的目标文件路径，请重新选择", ""
    try:
        file_name = file.name
        if file_name.endswith(".zip"):
            zip_files.append(os.path.abspath(file_name))

            with zipfile.ZipFile(file_name, "r") as z:
                z.extractall(target_path)
                for fname in os.listdir(target_path):
                    if fname.endswith(".wav"):
                        with sf.SoundFile(os.path.join(target_path, fname)) as sound:
                            if len(sound) / sound.samplerate > 10:
                                rm_files.append(os.path.join(target_path, fname))
                                wav_file_to_remove = fname.replace(".wav", ".lab")
                                if os.path.exists(
                                    os.path.join(target_path, wav_file_to_remove)
                                ):
                                    rm_files.append(
                                        os.path.join(target_path, wav_file_to_remove)
                                    )
                                    result_str += f"[Too long] Deleted {fname} because it's longer than {max_wav_len} seconds.\n"
                    elif fname.endswith(".lab"):
                        with open(
                            os.path.join(target_path, fname), "r", encoding="utf-8"
                        ) as f:
                            content = f.read()
                            if any(char in content for char in taboo_symbols):
                                rm_files.append(os.path.join(target_path, fname))
                                wav_file_to_remove = fname.replace(".lab", ".wav")
                                if os.path.exists(
                                    os.path.join(target_path, wav_file_to_remove)
                                ):
                                    rm_files.append(os.path.join(target_path, fname))
                                    result_str += f"[Invalid Chars] Deleted {wav_file_to_remove} and {fname} \n"

        if not result_str:
            return "No files were deleted.", ""
        else:
            for it in rm_files:
                if os.path.exists(it):
                    os.remove(it)
        sub_folders_str = update_transcript_status(target_path)
        return result_str, sub_folders_str

    except Exception as e:
        return str(e)


def clear_temp_files():
    try:
        _tmp_folder = os.path.join(temp_folder, "gradio")
        if os.path.exists(_tmp_folder):
            shutil.rmtree(_tmp_folder)
            return "Removed temp_folder: " + _tmp_folder
        else:
            return "already cleaned"
    except Exception as e:
        return str(e)


def fn_transcript(raw_folder):
    # 打开总的转写文本文件以写入数据
    raw_folder = os.path.dirname(raw_folder)
    os.makedirs("filelists", exist_ok=True)
    transcript_txt_file = os.path.join("filelists", "genshin.list")
    print(raw_folder, "\n", transcript_txt_file)
    with open(transcript_txt_file, "w", encoding="utf-8") as f:
        # 遍历 raw 文件夹下的所有子文件夹
        for root, _, files in os.walk(raw_folder):
            for file in files:
                if file.endswith(".lab"):
                    lab_file_path = os.path.join(root, file)
                    # 提取文件夹名
                    folder_name = os.path.basename(root)
                    folder_name_suffix = folder_name[-3:]
                    # 读取转写文本
                    with open(lab_file_path, "r", encoding="utf-8") as lab_file:
                        transcription = lab_file.read().strip()
                    # 获取对应的 WAV 文件路径
                    wav_file_path = os.path.splitext(lab_file_path)[0] + ".wav"
                    wav_file_path = wav_file_path.replace("\\", "/").replace(
                        "./raw", "./dataset"
                    )
                    print(wav_file_path)
                    # 写入数据到总的转写文本文件
                    line = f"{wav_file_path}|{folder_name}|{inv_lang_dict[folder_name_suffix]}|{transcription}\n"
                    f.write(line)
    return f"转写文本 {transcript_txt_file} 生成完成"


if __name__ == "__main__":
    with gr.Blocks(title="处理音频文件/压缩包") as app:
        with gr.Row():
            with gr.Column():
                file = gr.inputs.File(
                    label="上传 .zip or .rar",
                    type="file",
                    file_count="single",
                    keep_filename=False,
                    optional=True,
                )
                with gr.Row():
                    textbox_tar_path = gr.Textbox(
                        label="提取目的路径（可修改,精确到角色文件夹名)",
                        placeholder="输入目的文件夹路径",
                        value=args.target_path,
                        interactive=True,
                    )
                with gr.Row():
                    with gr.Column():
                        clear_btn = gr.Button(value="清除临时文件", variant="secondary")
                    with gr.Column():
                        submit_btn = gr.Button(value="1.整理音频", variant="primary")

                with gr.Row():
                    textbox_transcript = gr.Textbox(
                        label="可以提取的角色音频/状态",
                        value=update_transcript_status(textbox_tar_path.value),
                    )
                    dropdown_lang = gr.Dropdown(
                        label="选择语言", choices=list(lang_dict.keys()), value="ZH(中文)"
                    )
                with gr.Row():
                    transcript_btn = gr.Button(value="2.提取文本", variant="primary")
            with gr.Column():
                slider_max_wav_length = gr.Slider(
                    minimum=3, maximum=15, value=10, step=0.5, label="最大音频时长"
                )
                textbox_taboo_sym = gr.Textbox(
                    label="需要去掉的文本所包含的非法符号",
                    placeholder="输入所有需要屏蔽的符号",
                    value=taboo_symbols,
                    lines=3,
                    interactive=True,
                )
                textbox_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮")
        clear_btn.click(
            clear_temp_files,
            outputs=[
                textbox_output_text,
            ],
        )
        submit_btn.click(
            process_uploaded_files,
            inputs=[
                file,
                slider_max_wav_length,
                textbox_taboo_sym,
                textbox_tar_path,
                dropdown_lang,
            ],
            outputs=[textbox_output_text, textbox_transcript],
        )
        transcript_btn.click(
            fn_transcript, inputs=[textbox_tar_path], outputs=[textbox_transcript]
        )
    webbrowser.open("http://127.0.0.1:6660")
    app.launch(share=args.share, server_port=6660)
