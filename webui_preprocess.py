import gradio as gr
import webbrowser
import os
import json
import subprocess
import shutil


def get_path(data_dir):
    start_path = os.path.join("./data", data_dir)
    lbl_path = os.path.join(start_path, "esd.list")
    train_path = os.path.join(start_path, "train.list")
    val_path = os.path.join(start_path, "val.list")
    config_path = os.path.join(start_path, "configs", "config.json")
    return start_path, lbl_path, train_path, val_path, config_path


def generate_config(data_dir, batch_size):
    assert data_dir != "", "数据集名称不能为空"
    start_path, _, train_path, val_path, config_path = get_path(data_dir)
    if os.path.isfile(config_path):
        config = json.load(open(config_path, "r", encoding="utf-8"))
    else:
        config = json.load(open("configs/config.json", "r", encoding="utf-8"))
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["train"]["batch_size"] = batch_size
    out_path = os.path.join(start_path, "configs")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    model_path = os.path.join(start_path, "models")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    if not os.path.exists("config.yml"):
        shutil.copy(src="default_config.yml", dst="config.yml")
    return "配置文件生成完成"


def resample(data_dir):
    assert data_dir != "", "数据集名称不能为空"
    start_path, _, _, _, config_path = get_path(data_dir)
    in_dir = os.path.join(start_path, "raw")
    out_dir = os.path.join(start_path, "wavs")
    subprocess.run(
        f"python resample_legacy.py "
        f"--sr 44100 "
        f"--in_dir {in_dir} "
        f"--out_dir {out_dir} ",
        shell=True,
    )
    return "音频文件预处理完成"


def preprocess_text(data_dir):
    assert data_dir != "", "数据集名称不能为空"
    start_path, lbl_path, train_path, val_path, config_path = get_path(data_dir)
    lines = open(lbl_path, "r", encoding="utf-8").readlines()
    with open(lbl_path, "w", encoding="utf-8") as f:
        for line in lines:
            path, spk, language, text = line.strip().split("|")
            path = os.path.join(start_path, "wavs", os.path.basename(path)).replace(
                "\\", "/"
            )
            f.writelines(f"{path}|{spk}|{language}|{text}\n")
    subprocess.run(
        f"python preprocess_text.py "
        f"--transcription-path {lbl_path} "
        f"--train-path {train_path} "
        f"--val-path {val_path} "
        f"--config-path {config_path}",
        shell=True,
    )
    return "标签文件预处理完成"


def bert_gen(data_dir):
    assert data_dir != "", "数据集名称不能为空"
    _, _, _, _, config_path = get_path(data_dir)
    subprocess.run(
        f"python bert_gen.py " f"--config {config_path}",
        shell=True,
    )
    return "BERT 特征文件生成完成"


if __name__ == "__main__":
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                _ = gr.Markdown(
                    value="# Bert-VITS2 数据预处理\n"
                    "## 预先准备：\n"
                    "下载 BERT 和 WavLM 模型：\n"
                    "- [中文 RoBERTa](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)\n"
                    "- [日文 DeBERTa](https://huggingface.co/ku-nlp/deberta-v2-large-japanese-char-wwm)\n"
                    "- [英文 DeBERTa](https://huggingface.co/microsoft/deberta-v3-large)\n"
                    "- [WavLM](https://huggingface.co/microsoft/wavlm-base-plus)\n"
                    "\n"
                    "将 BERT 模型放置到 `bert` 文件夹下，WavLM 模型放置到 `slm` 文件夹下，覆盖同名文件夹。\n"
                    "\n"
                    "数据准备：\n"
                    "将数据放置在 data 文件夹下，按照如下结构组织：\n"
                    "\n"
                    "```\n"
                    "├── data\n"
                    "│   ├── {你的数据集名称}\n"
                    "│   │   ├── esd.list\n"
                    "│   │   ├── raw\n"
                    "│   │   │   ├── ****.wav\n"
                    "│   │   │   ├── ****.wav\n"
                    "│   │   │   ├── ...\n"
                    "```\n"
                    "\n"
                    "其中，`raw` 文件夹下保存所有的音频文件，`esd.list` 文件为标签文本，格式为\n"
                    "\n"
                    "```\n"
                    "****.wav|{说话人名}|{语言 ID}|{标签文本}\n"
                    "```\n"
                    "\n"
                    "例如：\n"
                    "```\n"
                    "vo_ABDLQ001_1_paimon_02.wav|派蒙|ZH|没什么没什么，只是平时他总是站在这里，有点奇怪而已。\n"
                    "noa_501_0001.wav|NOA|JP|そうだね、油断しないのはとても大事なことだと思う\n"
                    "Albedo_vo_ABDLQ002_4_albedo_01.wav|Albedo|EN|Who are you? Why did you alarm them?\n"
                    "...\n"
                    "```\n"
                )
                data_dir = gr.Textbox(
                    label="数据集名称",
                    placeholder="你放置在 data 文件夹下的数据集所在文件夹的名称，如 data/genshin 则填 genshin",
                )
                info = gr.Textbox(label="状态信息")
                _ = gr.Markdown(value="## 第一步：生成配置文件")
                with gr.Row():
                    batch_size = gr.Slider(
                        label="批大小（Batch size）：24 GB 显存可用 12",
                        value=8,
                        minimum=1,
                        maximum=64,
                        step=1,
                    )
                    generate_config_btn = gr.Button(value="执行", variant="primary")
                _ = gr.Markdown(value="## 第二步：预处理音频文件")
                resample_btn = gr.Button(value="执行", variant="primary")
                _ = gr.Markdown(value="## 第三步：预处理标签文件")
                preprocess_text_btn = gr.Button(value="执行", variant="primary")
                _ = gr.Markdown(value="## 第四步：生成 BERT 特征文件")
                bert_gen_btn = gr.Button(value="执行", variant="primary")
                _ = gr.Markdown(
                    value="## 训练模型及部署：\n"
                    "修改根目录下的 `config.yml` 中 `dataset_path` 一项为 `data/{你的数据集名称}`\n"
                    "- 训练：将[预训练模型文件](https://openi.pcl.ac.cn/Stardust_minus/Bert-VITS2/modelmanage/show_model)（`D_0.pth`、`DUR_0.pth`、`WD_0.pth` 和 `G_0.pth`）放到 `data/{你的数据集名称}/models` 文件夹下，执行 `torchrun --nproc_per_node=1 train_ms.py` 命令（多卡运行可参考 `run_MnodesAndMgpus.sh` 中的命令。\n"
                    "- 部署：修改根目录下的 `config.yml` 中 `webui` 下 `model` 一项为 `models/{权重文件名}.pth` （如 G_10000.pth），然后执行 `python webui.py`"
                )

        generate_config_btn.click(
            generate_config, inputs=[data_dir, batch_size], outputs=[info]
        )
        resample_btn.click(resample, inputs=[data_dir], outputs=[info])
        preprocess_text_btn.click(preprocess_text, inputs=[data_dir], outputs=[info])
        bert_gen_btn.click(bert_gen, inputs=[data_dir], outputs=[info])

    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=False, server_port=7860)
