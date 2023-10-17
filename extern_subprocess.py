import os
import subprocess
import re
import multiprocessing
import json
import platform
from update_status import raw_dir_convert_to_path
lang_dict = {"EN(英文)": "_en", "ZH(中文)": "_zh", "JP(日语)": "_jp"}

def update_json(batch_size:int, log_interval:int, eval_interval:int,
                epochs:int, lr:float, keep_ckpts:int):
    with open("configs/config.json", "r", encoding='utf-8') as json_file:
        hps = json.load(json_file)
    hps["train"]["batch_size"] = batch_size
    hps["train"]["log_interval"] = log_interval
    hps["train"]["eval_interval"] = eval_interval
    hps["train"]["epochs"] = epochs
    hps["train"]["learning_rate"] = lr
    hps["train"]["keep_ckpts"] = keep_ckpts
    print("现在的[BS,LI,EI,epochs,lr,keep]: ", [batch_size, log_interval, eval_interval, epochs, lr, keep_ckpts])
    with open("configs/config.json", "w", encoding='utf-8') as json_file:
        json.dump(hps, json_file, indent=4)
    print("config.json文件已更新")


class SubprocessManager:
    def __init__(self):
        self.process = None

    def worker(self, command):
        try:
            result = subprocess.check_output(
                command,
                universal_newlines=True
            )
            result = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', result)
            print(result)
        except subprocess.CalledProcessError as e:
            print(f"错误: {str(e)}")

    def start(self, command):
        if self.process:
            print("已有子进程正在运行，先终止它")
            self.terminate()

        if platform.system() == "Windows":
            cmd = ["cmd.exe", "/c"] + command
        else:
            cmd = command
        print(" ".join(cmd))
        self.process = multiprocessing.Process(target=self.worker, args=(cmd,))
        self.process.start()

    def terminate(self):
        if self.process:
            self.process.terminate()
            self.process.join()
            print("子进程已被终止")
            self.process = None


managers = [SubprocessManager() for _ in range(7)]

def do_transcribe(target_path, language, workers):
    target_path = raw_dir_convert_to_path(target_path, language)
    additional_args = ["-f", target_path, "-l", language, "-w", str(workers)]
    command = [r"python", "asr_transcript.py"]
    command.extend(additional_args)
    os.environ["SELECT_LANGUAGE"] = language
    managers[0].start(command)
    print("开始转写文本！请稍后~首先是验证下载模型，然后是转写~")

    return "开始转写文本！请稍后~首先是验证下载模型，然后是转写~"


def do_preprocess_text(target_path=""):
    os.makedirs("filelists", exist_ok=True)
    command = [r"python", "preprocess_text.py"]
    managers[1].start(command)
    print("开始生成训练集和验证集！请稍后~")
    return "开始生成训练集和验证集！请稍后~"

def do_resample(target_path=""):
    command = [r"python", "resample.py"]
    managers[2].start(command)
    print("开始音频重采样！请稍后~")
    return "开始音频重采样！请稍后~"

def do_bert_gen(num_processes, target_path=""):
    command = [r"python", "bert_gen.py"]
    command.extend(["--num_processes", str(num_processes)])
    managers[3].start(command)
    print("开始bert生成！请稍后~")
    return "开始bert生成！请稍后~"

def terminate_training():
    managers[4].terminate()
    print("终止训练！")
    return "终止训练！"

def do_training(model_folder:str, batch_size:int, log_interval:int, eval_interval:int,
                epochs:int, lr:float, keep_ckpts:int):
    update_json(batch_size, log_interval, eval_interval, epochs, lr, keep_ckpts)
    command = [r"python", "train_ms.py"]
    command.extend(["-m", model_folder,
                    "-c", './configs/config.json'])
    terminate_training()
    managers[4].start(command)
    print("开启训练成功！\n http://127.0.0.1:8000")
    return "开启训练成功！\n http://127.0.0.1:8000"

def terminate_webui():
    managers[5].terminate()
    print("关闭推理页面！")
    return "关闭推理页面！"

def do_inference_webui(model_path:str, config_path:str):
    command = [r"python", "webui.py"]
    command.extend(["-m", model_path, "-c", config_path])
    if not os.path.exists(model_path):
        return "找不到对应模型！请确保模型路径正确！"
    if not os.path.exists(config_path):
        return "找不到对应配置文件！请确保配置路径正确！"
    terminate_webui()
    managers[5].start(command)
    print("开启推理页面成功 \n http://127.0.0.1:7860")
    return "开启推理页面成功 \n http://127.0.0.1:7860"


def do_test(model_path:str=""):
    command = [r"python", "-m", "pip", "list"]
    managers[6].start(command)
    return "正在测试，请看控制台"
