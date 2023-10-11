import os
import subprocess
import re
import multiprocessing
import json
lang_dict = {"EN(英文)": "_en", "ZH(中文)": "_zh", "JP(日语)": "_jp"}

def update_batch(batch_size):
    with open("configs/config.json", "r", encoding='utf-8') as json_file:
        hps = json.load(json_file)
    hps["train"]["batch_size"] = batch_size
    print("现在的batch_size: ", batch_size)
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
        print(command)
        self.process = multiprocessing.Process(target=self.worker, args=(command,))
        self.process.start()

    def terminate(self):
        if self.process:
            self.process.terminate()
            self.process.join()
            print("子进程已被终止")
            self.process = None


managers = [SubprocessManager() for _ in range(7)]

def do_transcribe(target_path, language):
    additional_args = ["-f", target_path + lang_dict[language]]
    command = [r"python", "asr_transcript.py"]
    command.extend(additional_args)
    os.environ["SELECT_LANGUAGE"] = language
    managers[0].start(command)
    return "转写文本成功！"


def do_preprocess_text(target_path=""):
    os.makedirs("filelists", exist_ok=True)
    command = [r"python", "preprocess_text.py"]
    managers[1].start(command)
    return "生成训练集和验证集成功！"

def do_resample(target_path=""):
    command = [r"python", "resample.py"]
    managers[2].start(command)
    return "重采样完成！"

def do_bert_gen(num_processes, target_path=""):
    command = [r"python", "bert_gen.py"]
    command.extend(["--num_processes", str(num_processes)])
    managers[3].start(command)
    return "bert生成完成！"

def terminate_training():
    managers[4].terminate()
    return "终止训练！"

def do_training(model_folder:str, batch_size:int):
    update_batch(batch_size)

    model_dir = f'./logs/{model_folder}'
    os.makedirs(model_dir, exist_ok=True)
    command = [r"python", "train_ms.py"]
    command.extend(["-m", model_dir,
                    "-c", './configs/config.json'])
    terminate_training()
    managers[4].start(command)
    return "开启训练成功！\n http://127.0.0.1:8000"

def terminate_webui():
    managers[5].terminate()
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
    return "开启推理页面成功 \n http://127.0.0.1:7860"


def do_test(model_path:str=""):
    command = [r"python", "-m", "pip", "list"]
    managers[6].start(command)
    return "正在测试，请看控制台"