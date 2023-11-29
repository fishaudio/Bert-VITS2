import json
import os
import platform
import shutil
import signal
import subprocess
import webbrowser

import GPUtil
import gradio as gr
import psutil
import torch
import yaml

from config import yml_config
from tools.log import logger

bert_model_paths = [
    "./bert/bert-base-japanese-v3/pytorch_model.bin",
    "./bert/bert-large-japanese-v2/pytorch_model.bin",
    "./bert/chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "./bert/deberta-v2-large-japanese/pytorch_model.bin",
    "./bert/deberta-v3-large/pytorch_model.bin",
    "./bert/deberta-v3-large/pytorch_model.generator.bin",
    "./bert/deberta-v3-large/spm.model",
]

emo_model_paths = [
    "./emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim/pytorch_model.bin"
]

train_base_model_paths = [
    "D_0.pth",
    "G_0.pth",
    "DUR_0.pth"
]
default_yaml_path = "default_config.yml"
default_config_path = "configs/config.json"


def load_yaml_data_in_raw(yml_path=yml_config):
    with open(yml_path, 'r', encoding='utf-8') as file:
        # data = yaml.safe_load(file)
        data = file.read()
    return str(data)


def load_json_data_in_raw(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    formatted_json_data = json.dumps(json_data, ensure_ascii=False, indent=2)
    return formatted_json_data


def load_json_data_in_fact(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    return json_data


def load_yaml_data_in_fact(yml_path=yml_config):
    with open(yml_path, 'r', encoding='utf-8') as file:
        yml = yaml.safe_load(file)
        # data = file.read()
    return yml


def load_train_param(cfg_path):
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    train_json_path = os.path.join(data_path, cfg_path).replace("\\", "/")
    json_data = load_json_data_in_fact(train_json_path)
    bs = json_data['train']['batch_size']
    nc = json_data['train'].get('keep_ckpts', 5)
    li = json_data['train']['log_interval']
    ei = json_data['train']['eval_interval']
    ep = json_data['train']['epochs']
    lr = json_data['train']['learning_rate']
    ver = json_data['version']
    msg = f"加载训练配置文件: {train_json_path}"
    logger.info(msg)
    return gr.Textbox(value=msg), gr.Code(label=train_json_path, value=load_yaml_data_in_raw(train_json_path)), \
        gr.Slider(value=bs), gr.Slider(value=nc), gr.Slider(value=li), gr.Slider(value=ei), \
        gr.Slider(value=ep), gr.Slider(value=lr), gr.Dropdown(value=ver)


def write_yaml_data_in_fact(yml, yml_path=yml_config):
    with open(yml_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(yml, file, allow_unicode=True)
        # data = file.read()
    return yml


def write_json_data_in_fact(json_path, json_data):
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=2)


def check_if_exists_model(paths: list[str]):
    check_results = {path: os.path.exists(path) and os.path.isfile(path) for path in paths}
    val = [path for path, exists in check_results.items() if exists]
    return val


def check_bert_models():
    return gr.CheckboxGroup(value=check_if_exists_model(bert_model_paths))


def check_emo_models():
    return gr.CheckboxGroup(value=check_if_exists_model(emo_model_paths))


def check_base_models():
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    models_dir = yml['train_ms']['model']
    model_paths = [os.path.join(data_path, models_dir, p).replace('\\', '/') for p in train_base_model_paths]
    return gr.CheckboxGroup(
        label="检测底模状态",
        info="最好去下载底模进行训练",
        choices=model_paths,
        value=check_if_exists_model(model_paths),
        interactive=False
    )


def modify_data_path(data_path):
    yml = load_yaml_data_in_fact()
    yml['dataset_path'] = data_path
    write_yaml_data_in_fact(yml)
    txt_box = gr.Textbox(value=data_path)
    return gr.Dropdown(value=data_path), txt_box, txt_box, txt_box, \
        gr.Code(value=load_yaml_data_in_raw()), check_base_models()


def modify_preprocess_param(trans_path, cfg_path, val_per_spk, max_val_total):
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    yml['preprocess_text']['transcription_path'] = trans_path
    yml['preprocess_text']['config_path'] = cfg_path
    yml['preprocess_text']['val_per_spk'] = val_per_spk
    yml['preprocess_text']['max_val_total'] = max_val_total
    write_yaml_data_in_fact(yml)
    whole_path = os.path.join(data_path, cfg_path).replace("\\", "/")
    logger.info("预处理配置: ", whole_path)
    if not os.path.exists(whole_path):
        os.makedirs(os.path.dirname(whole_path), exist_ok=True)
        shutil.copy("configs/config.json", os.path.dirname(whole_path))
    return gr.Dropdown(value=trans_path), gr.Code(value=load_yaml_data_in_raw())


def modify_resample_path(in_dir, out_dir, sr):
    yml = load_yaml_data_in_fact()
    yml['resample']['in_dir'] = in_dir
    yml['resample']['out_dir'] = out_dir
    yml['resample']['sampling_rate'] = int(sr)
    write_yaml_data_in_fact(yml)
    msg = f"重采样参数已更改: [{in_dir}, {out_dir}, {sr}]\n"
    logger.info(msg)
    return gr.Textbox(value=in_dir), gr.Textbox(value=out_dir), gr.Textbox(value=msg), \
        gr.Dropdown(value=sr), gr.Code(value=load_yaml_data_in_raw())


def modify_bert_config(cfg_path, nps, dev, multi):
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    yml['bert_gen']['config_path'] = cfg_path
    yml['bert_gen']['num_processes'] = int(nps)
    yml['bert_gen']['device'] = dev
    yml['bert_gen']['use_multi_device'] = multi
    write_yaml_data_in_fact(yml)
    whole_path = os.path.join(data_path, cfg_path).replace("\\", "/")
    logger.info("bert配置路径: ", whole_path)
    if not os.path.exists(whole_path):
        os.makedirs(os.path.dirname(whole_path), exist_ok=True)
        shutil.copy("configs/config.json", os.path.dirname(whole_path))
    return gr.Textbox(value=cfg_path), gr.Slider(value=int(nps)), \
        gr.Dropdown(value=dev), gr.Radio(value=multi), gr.Code(value=load_yaml_data_in_raw())


def modify_train_path(model, cfg_path):
    yml = load_yaml_data_in_fact()
    yml['train_ms']['config_path'] = cfg_path
    yml['train_ms']['model'] = model
    write_yaml_data_in_fact(yml)
    logger.info(f"训练配置文件路径: {cfg_path}\n")
    logger.info(f"训练模型文件夹路径: {model}")
    return gr.Textbox(value=model), gr.Textbox(value=cfg_path), \
        gr.Code(value=load_yaml_data_in_raw()), check_base_models()


def modify_train_param(bs, nc, li, ei, ep, lr, ver):
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    cfg_path = yml['train_ms']['config_path']
    ok = False
    whole_path = os.path.join(data_path, cfg_path).replace("\\", "/")
    logger.info("config_path: ", whole_path)
    if not os.path.exists(whole_path):
        os.makedirs(os.path.dirname(whole_path), exist_ok=True)
        shutil.copy("configs/config.json", os.path.dirname(whole_path))
    if os.path.exists(whole_path) and os.path.isfile(whole_path):
        ok = True
        with open(whole_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        json_data['train']['batch_size'] = bs
        json_data['train']['keep_ckpts'] = nc
        json_data['train']['log_interval'] = li
        json_data['train']['eval_interval'] = ei
        json_data['train']['epochs'] = ep
        json_data['train']['learning_rate'] = lr
        json_data['version'] = ver
        with open(whole_path, 'w', encoding='utf-8') as file:
            json.dump(json_data, file, ensure_ascii=False, indent=2)
        msg = f"成功更改训练参数! [{bs},{nc},{li},{ei},{ep},{lr}]"
        logger.info(msg)
    else:
        msg = f"打开训练配置文件时出现错误: {whole_path}\n" \
              f"该文件不存在或损坏，现在打开默认配置文件"
        logger.error(msg)
    return gr.Textbox(value=msg), gr.Code(
        label=whole_path
        if ok else default_config_path,
        value=load_json_data_in_raw(whole_path)
        if ok else load_json_data_in_raw(default_config_path)
    )


def modify_infer_param(model_path, config_path, port, share, debug, ver):
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    yml['webui']['model'] = os.path.relpath(model_path, start=data_path)
    yml['webui']['config_path'] = os.path.relpath(config_path, start=data_path)
    port = int(port)
    port = port if 0 <= port <= 65535 else 10086
    yml['webui']['port'] = port
    yml['webui']['share'] = share
    yml['webui']['debug'] = debug
    write_yaml_data_in_fact(yml)
    json_data = load_json_data_in_fact(config_path)
    json_data['version'] = ver
    write_json_data_in_fact(config_path, json_data)
    msg = f"修改推理配置文件成功: [{model_path}, {config_path}, {port}, {ver}]"
    logger.info(msg)
    return gr.Textbox(value=msg), gr.Code(value=load_yaml_data_in_raw()), \
        gr.Code(label=config_path,
                value=load_json_data_in_raw(config_path)
                if os.path.exists(config_path) else load_json_data_in_raw(default_config_path)
                )


def get_status():
    """获取电脑运行状态"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_total = memory_info.total
    memory_available = memory_info.available
    memory_used = memory_info.used
    memory_percent = memory_info.percent
    gpuInfo = []
    devices = ["cpu"]
    for i in range(torch.cuda.device_count()):
        devices.append(f"cuda:{i}")
    if torch.cuda.device_count() > 0:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpuInfo.append(
                {
                    "GPU编号": gpu.id,
                    "GPU负载": f'{gpu.load} %',
                    "专用GPU内存": {
                        "总内存": f'{gpu.memoryTotal} MB',
                        "已使用": f'{gpu.memoryUsed} MB',
                        "空闲": f'{gpu.memoryFree} MB',
                    },
                }
            )
    status_data = {
        "devices": devices,
        "CPU占用率": f"{cpu_percent} %",
        "总内存": f"{memory_total // (1024 * 1024)} MB",
        "可用内存": f"{memory_available // (1024 * 1024)} MB",
        "已使用内存": f"{memory_used // (1024 * 1024)} MB",
        "百分数": f"{memory_percent} %",
        "gpu信息": gpuInfo,
    }
    formatted_json_data = json.dumps(status_data, ensure_ascii=False, indent=2)
    logger.info(formatted_json_data)
    return str(formatted_json_data)


def get_gpu_status():
    return gr.Code(value=get_status())


def list_infer_models():
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    inf_models, json_files = [], []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file).replace('\\', '/')
            if file.startswith("G_") and file.lower().endswith(".pth"):
                inf_models.append(filepath)
            elif file.lower().endswith(".json"):
                json_files.append(filepath)
    logger.info("找到推理模型文件: " + str(inf_models))
    logger.info("找到推理配置文件: " + str(json_files))
    return gr.Dropdown(choices=inf_models), gr.Dropdown(choices=json_files)


def do_resample(nps):
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    in_dir = yml['resample']['in_dir']
    comp_in_dir = os.path.join(os.path.abspath(data_path), in_dir).replace('\\', '/')
    logger.info(f'\n重采样路径: {comp_in_dir}')
    cmd = f'python resample.py --processes {nps}'
    logger.info(cmd)
    subprocess.run(cmd, shell=True)
    return gr.Textbox(value="重采样完成!")


def do_transcript(lang, workers):
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    in_dir = yml['resample']['in_dir']
    comp_in_dir = os.path.join(os.path.abspath(data_path), in_dir).replace('\\', '/')
    logger.info(f'\n转写文件夹路径: {comp_in_dir}')
    cmd = f'python asr_transcript.py -f \"{comp_in_dir}\" -l {lang} -w {workers}'
    logger.info(cmd)
    subprocess.run(cmd, shell=True)
    return gr.Textbox(value=f'\n转写文件夹路径: {comp_in_dir}\n转写到.lab完成!')


def do_extract(raw_path, lang, unclean, char_name):
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    lab_path = os.path.join(os.path.abspath(data_path), raw_path).replace('\\', '/')
    unclean_path = os.path.join(data_path, os.path.splitext(unclean)[0] + ".txt").replace('\\', '/')
    logger.info(f'\n提取转写文本路径: {lab_path}')
    lab_ok = False
    for root, _, files in os.walk(lab_path):
        for f_name in files:
            if str(f_name).lower().endswith('.lab'):
                lab_ok = True
                break
        if lab_ok:
            break

    if os.path.exists(lab_path) and os.path.isdir(lab_path):
        if lab_ok:
            cmd = f'python extract_list.py -f \"{lab_path}\" -l {lang} -n \"{char_name}\" -o \"{unclean_path}\"'
            logger.info(cmd)
            subprocess.run(cmd, shell=True)
            msg = f'提取完成!生成如下文件: {unclean_path}'
            logger.info(msg)
        else:
            msg = '未找到提取转写文本路径下的.lab文件!'
            logger.warning(msg)
    else:
        msg = '路径未选择正确!'
        logger.error(msg)
    return gr.Textbox(value=msg)


def do_clean_list(ban_chars, unclean, clean):
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    unclean_path = os.path.join(data_path, unclean)
    clean_path = os.path.join(data_path, clean)
    if os.path.exists(unclean_path) and os.path.isfile(unclean_path):
        cmd = f'python clean_list.py -c \"{ban_chars}\" -i \"{unclean_path}\" -o \"{clean_path}\"'
        logger.info(cmd)
        subprocess.run(cmd, shell=True)
        msg = "清洗标注文本完成!"
        logger.info(msg)
    else:
        msg = "未找到可清洗标注文本，请到2.2节重新生成!"
        logger.warning(msg)
    return gr.Textbox(value=msg)


def do_preprocess_text():
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    trans_path = yml['preprocess_text']['transcription_path']
    comp_trans_path = os.path.join(os.path.abspath(data_path), trans_path).replace('\\', '/')
    logger.info(f'\n清洗后标注文本文件路径: {comp_trans_path}')
    if os.path.exists(comp_trans_path) and os.path.isfile(comp_trans_path):
        cmd = f'python preprocess_text.py'
        logger.info(cmd)
        subprocess.run(cmd, shell=True)
        msg = "文本预处理完成!"
    else:
        msg = '\n清洗后标注文本文件不存在或失效！'
        logger.info(msg)
    return gr.Textbox(value=msg)


def do_bert_gen():
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    train_list_path = yml['preprocess_text']['train_path']
    val_list_path = yml['preprocess_text']['val_path']
    comp_t_path = os.path.join(os.path.abspath(data_path), train_list_path).replace('\\', '/')
    comp_v_path = os.path.join(os.path.abspath(data_path), val_list_path).replace('\\', '/')
    if os.path.exists(comp_t_path) and os.path.isfile(comp_t_path):
        subprocess.run('python bert_gen.py', shell=True)
        msg = "bert文件生成完成!"
        logger.info(msg)
    else:
        msg = f'未找到训练集和验证集文本!\ntrain: {comp_t_path}\nval:{comp_v_path}'
        logger.error(msg)
    return gr.Textbox(value=msg)


def modify_emo_gen(emo_cfg, emo_nps, emo_device):
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    yml['emo_gen']['config_path'] = emo_cfg
    yml['emo_gen']['num_processes'] = emo_nps
    yml['emo_gen']['device'] = emo_device
    write_yaml_data_in_fact(yml)
    comp_emo_cfg = os.path.join(os.path.abspath(data_path), emo_cfg).replace('\\', '/')
    msg = f"修改emo配置参数: [配置路径:{comp_emo_cfg}, 处理数:{emo_nps}, 设备:{emo_device}]"
    logger.info(msg)
    return gr.Textbox(value=msg), gr.Code(value=load_yaml_data_in_raw())


def do_emo_gen():
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    emo_config_path = yml['emo_gen']['config_path']
    comp_emo_path = os.path.join(os.path.abspath(data_path), emo_config_path).replace('\\', '/')
    if os.path.exists(comp_emo_path) and os.path.isfile(comp_emo_path):
        subprocess.run('python emo_gen.py', shell=True)
        msg = "emo.npy文件生成完成!"
        logger.info(msg)
    else:
        msg = f'选定路径下未找到配置文件!\n需要的config路径 : {comp_emo_path}'
        logger.error(msg)

    return gr.Textbox(value=msg)


def do_my_train():
    yml = load_yaml_data_in_fact()
    n_gpus = torch.cuda.device_count()
    # subprocess.run(f'python train_ms.py', shell=True)
    if os.path.exists(r'..\vits\python.exe') and os.path.isfile(r'..\vits\python.exe'):
        cmd = r'..\vits\python ..\vits\Scripts\torchrun.exe ' \
              f'--nproc_per_node={n_gpus} train_ms.py'
    else:
        cmd = f'torchrun --nproc_per_node={n_gpus} train_ms.py'

    subprocess.Popen(cmd, shell=True)
    train_port = yml['train_ms']['env']['MASTER_PORT']
    train_addr = yml['train_ms']['env']['MASTER_ADDR']
    url = f'env://{train_addr}:{train_port}'
    msg = f"训练开始!\nMASTER_URL: {url}\n使用gpu数:{n_gpus}\n推荐按下终止训练按钮来结束!"
    logger.info(msg)
    return gr.Textbox(value=msg)


def do_tensorboard():
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    train_model_dir = yml['train_ms']['model']
    whole_dir = os.path.join(data_path, train_model_dir).replace('\\', '/')
    if os.path.exists(r'..\vits\python.exe') and os.path.isfile(r'..\vits\python.exe'):
        first_cmd = r'..\vits\python ..\vits\Scripts\tensorboard.exe '
    else:
        first_cmd = 'tensorboard '
    tb_cmd = first_cmd + \
             f'--logdir={whole_dir} ' \
             f'--port={11451} ' \
             f'--window_title=\"训练情况一览\" ' \
             f'--reload_interval={120}'
    subprocess.Popen(tb_cmd, shell=True)
    url = f"http://localhost:{11451}"
    webbrowser.open(url=url)
    msg = tb_cmd + '\n' + url
    logger.info(msg)
    return gr.Textbox(value=msg)


def do_webui_infer():
    yml = load_yaml_data_in_fact()
    data_path = yml['dataset_path']
    model_path = yml['webui']['model']
    config_path = yml['webui']['config_path']
    comp_m_path = os.path.join(os.path.abspath(data_path), model_path)
    comp_c_path = os.path.join(os.path.abspath(data_path), config_path)
    if os.path.exists(comp_c_path) and os.path.exists(comp_m_path):
        webui_port = yml['webui']['port']
        subprocess.Popen('python webui.py', shell=True)
        url = f'http://localhost:{webui_port} | http://127.0.0.1:{webui_port}'
        msg = f"推理端已开启, 到控制台中复制网址打开页面\n{url}\n选择的模型:{model_path}"
        logger.info(msg)
    else:
        msg = f'未找到有效的模型或配置文件!\n模型路径:{comp_m_path}\n配置路径:{comp_c_path}'
        logger.error(msg)
    return gr.Textbox(value=msg)


def compress_model(cfg_path, in_path, out_path):
    subprocess.Popen("python compress_model.py"
                     f" -c {cfg_path}"
                     f" -i {in_path}", shell=True)
    msg = "到控制台中查看压缩结果"
    logger.info(msg)
    return gr.Textbox(value=msg)


def kill_specific_process_linux(cmd):
    try:
        output = subprocess.check_output(["pgrep", "-f", cmd], text=True)
        pids = output.strip().split('\n')

        for pid in pids:
            if pid:
                logger.critical(f"终止进程: {pid}")
                os.kill(int(pid), signal.SIGTERM)
                # os.kill(int(pid), signal.SIGKILL)
    except subprocess.CalledProcessError:
        logger.error("没有找到匹配的进程。")
    except Exception as e:
        logger.error(f"发生错误: {e}")


def kill_specific_process_windows(cmd):
    try:
        # 使用tasklist和findstr来找到匹配特定命令行模式的进程
        output = subprocess.check_output(f'tasklist /FO CSV /V | findstr /C:"{cmd}"', shell=True, text=True)
        lines = output.strip().split('\n')

        for line in lines:
            if line:
                pid = line.split(',')[1].strip('"')
                logger.critical(f"终止进程: {pid}")
                subprocess.run(["taskkill", "/PID", pid, "/F"], shell=True)  # 强制终止
    except subprocess.CalledProcessError:
        logger.error(f"没有找到匹配的{cmd}进程。")
    except Exception as e:
        logger.error(f"发生错误: {e}")


def stop_train_ms():
    yml = load_yaml_data_in_fact()
    train_port = yml['train_ms']['env']['MASTER_PORT']
    train_addr = yml['train_ms']['env']['MASTER_ADDR']
    if platform.system() == "Windows":
        kill_specific_process_windows('torchrun')
    else:
        kill_specific_process_linux('torchrun')
    url = f'env://{train_addr}:{train_port}'
    msg = f"训练结束!\nMASTER_URL: {url}"
    logger.critical(msg)
    return gr.Textbox(value=msg)


def stop_tensorboard():
    if platform.system() == "Windows":
        kill_specific_process_windows("tensorboard")
    else:
        kill_specific_process_linux("tensorboard")
    msg = f"关闭tensorboard!\n"
    logger.critical(msg)
    return gr.Textbox(value=msg)


def stop_webui_infer():
    yml = load_yaml_data_in_fact()
    webui_port = yml['webui']['port']
    if platform.system() == "Linux":
        kill_specific_process_linux('python webui.py')
    else:
        kill_specific_process_windows('python webui.py')
    msg = f"尝试终止推理进程，请到控制台查看情况\nport={webui_port}"
    logger.critical(msg)
    return gr.Textbox(value=msg)


if __name__ == '__main__':
    init_yml = load_yaml_data_in_fact()
    with gr.Blocks(title="Bert-VITS-2-v2.0-管理器",
                   theme=gr.themes.Soft(),
                   css=os.path.abspath("./css/custom.css")
                   ) as app:
        with gr.Row():
            with gr.Tabs():
                with gr.TabItem("首页"):
                    gr.Markdown("""
                        ## Bert-VITS2-v2.0 可视化界面
                        #### Copyright/Powered by 怕吃辣滴辣子酱 
                        #### 许可： [AGPL 3.0 Licence](https://github.com/AnyaCoder/Bert-VITS2/blob/master/LICENSE)
                        #### 请订阅我的频道: 
                        1. Bilibili： [spicysama](https://space.bilibili.com/47278440)
                        2. github： [AnyaCoder](https://github.com/AnyaCoder)

                        ### 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。
                        ### 严禁用于任何政治相关用途。
                        ## References
                        + [anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS)
                        + [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
                        + [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
                        + [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
                        + [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
                        ## 感谢所有贡献者作出的努力
                        <a href="https://github.com/AnyaCoder/Bert-VITS2/graphs/contributors">
                          <img src="https://contrib.rocks/image?repo=AnyaCoder/Bert-VITS2" />
                        </a>

                        Made with [contrib.rocks](https://contrib.rocks).

                    """)
                with gr.TabItem("模型检测"):
                    CheckboxGroup_bert_models = gr.CheckboxGroup(
                        label="检测bert模型状态",
                        info="对应文件夹下必须有对应的模型文件 (前2个是兼容旧模型的，也可以不下载)",
                        choices=bert_model_paths,
                        value=check_if_exists_model(bert_model_paths),
                        interactive=False
                    )
                    check_pth_btn1 = gr.Button(value="检查bert模型状态")
                    CheckboxGroup_emo_models = gr.CheckboxGroup(
                        label="检测emo模型状态",
                        info="对应文件夹下必须有对应的模型文件",
                        choices=emo_model_paths,
                        value=check_if_exists_model(emo_model_paths),
                        interactive=False
                    )
                    check_pth_btn2 = gr.Button(value="检查emo模型状态")
                with gr.TabItem("数据处理"):
                    with gr.Row():
                        dropdown_data_path = gr.Dropdown(
                            label="选择数据集存放路径 (右侧的dataset_path)",
                            info="详细说明可见右侧带注释的yaml文件",
                            interactive=True,
                            allow_custom_value=True,
                            choices=[init_yml['dataset_path']],
                            value=init_yml['dataset_path']
                        )
                    with gr.Row():
                        data_path_btn = gr.Button(value="确认更改存放路径", variant="primary")
                    with gr.Tabs():
                        with gr.TabItem("1. 音频重采样"):
                            with gr.Row():
                                resample_in_box = gr.Textbox(
                                    label="输入音频文件夹in_dir",
                                    value=init_yml['resample']['in_dir'],
                                    lines=1,
                                    interactive=True
                                )
                                resample_out_box = gr.Textbox(
                                    label="输出音频文件夹out_dir",
                                    lines=1,
                                    value=init_yml['resample']['out_dir'],
                                    interactive=True
                                )
                            with gr.Row():
                                dropdown_resample_sr = gr.Dropdown(
                                    label="输出采样率(Hz)",
                                    choices=['16000', '22050', '44100', '48000'],
                                    value='44100'
                                )
                                slider_resample_nps = gr.Slider(
                                    label="采样用的CPU核心数",
                                    minimum=1, maximum=64, step=1, value=2
                                )
                            with gr.Row():
                                resample_config_btn = gr.Button(
                                    value="确认重采样配置",
                                    variant="secondary",
                                )
                                resample_btn = gr.Button(
                                    value="1. 音频重采样",
                                    variant="primary",
                                )
                            with gr.Row():
                                resample_status = gr.Textbox(
                                    label="重采样结果",
                                    placeholder="执行重采样后可查看",
                                    lines=3,
                                    interactive=False
                                )
                        with gr.TabItem("2. 转写文本生成"):
                            with gr.Row():
                                dropdown_lang = gr.Dropdown(
                                    label="选择语言",
                                    info="ZH中文，JP日语，EN英语",
                                    choices=["ZH", "JP", "EN"],
                                    value="ZH"
                                )
                                slider_transcribe = gr.Slider(
                                    label="转写进程数",
                                    info="目的路径与前一节一致\n 重采样的输入路径",
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    value=1,
                                    interactive=True
                                )
                                clean_txt_box = gr.Textbox(
                                    label="非法字符集",
                                    info='在此文本框内出现的字符都会被整行删除',
                                    lines=1,
                                    value="{}<>",
                                    interactive=True
                                )
                            with gr.Row():
                                unclean_box = gr.Textbox(
                                    label="未清洗的文本",
                                    info="仅将.lab提取到这个文件里, 请保持txt格式",
                                    lines=1,
                                    value=os.path.splitext(init_yml['preprocess_text']['transcription_path'])[0]
                                          + '.txt',
                                    interactive=True
                                )
                                clean_box = gr.Textbox(
                                    label="已清洗的文本",
                                    info="将未清洗的文本做去除非法字符集处理后的文本",
                                    lines=1,
                                    value=init_yml['preprocess_text']['transcription_path'],
                                    interactive=True
                                )
                                char_name_box = gr.Textbox(
                                    label="输入角色名",
                                    info="区分说话人用",
                                    lines=1,
                                    placeholder="填入一个名称",
                                    interactive=True
                                )
                            with gr.Row():
                                transcribe_btn = gr.Button(
                                    value="2.1 转写文本",
                                    interactive=True
                                )
                                extract_list_btn = gr.Button(
                                    value="2.2 合成filelist",
                                )
                                clean_trans_btn = gr.Button(
                                    value="2.3 清洗标注"
                                )
                            with gr.Row():
                                preprocess_status_box = gr.Textbox(
                                    label="标注状态"
                                )
                        with gr.TabItem("3. 文本预处理"):
                            with gr.Row():
                                slider_val_per_spk = gr.Slider(
                                    label="每个speaker的验证集条数",
                                    info="TensorBoard里的eval音频展示条目",
                                    minimum=1, maximum=20,
                                    step=1, value=init_yml['preprocess_text']['val_per_spk']
                                )
                                slider_max_val_total = gr.Slider(
                                    label="验证集最大条数",
                                    info="多于此项的会被截断并放到训练集中",
                                    minimum=8, maximum=160,
                                    step=8, value=init_yml['preprocess_text']['max_val_total']
                                )
                            with gr.Row():
                                dropdown_filelist_path = gr.Dropdown(
                                    interactive=True,
                                    label="输入filelist路径",
                                    allow_custom_value=True,
                                    choices=[init_yml['preprocess_text']['transcription_path']],
                                    value=init_yml['preprocess_text']['transcription_path']
                                )
                                preprocess_config_box = gr.Textbox(
                                    label="预处理配置文件路径",
                                    value=init_yml['preprocess_text']['config_path']
                                )
                            with gr.Row():
                                preprocess_config_btn = gr.Button(
                                    value="更新预处理配置文件"
                                )
                                preprocess_text_btn = gr.Button(
                                    value="标注文本预处理",
                                    variant="primary"
                                )
                            with gr.Row():
                                label_status = gr.Textbox(
                                    label="转写状态"
                                )
                        with gr.TabItem("4. bert_gen"):
                            with gr.Row():
                                bert_dataset_box = gr.Textbox(
                                    label="数据集存放路径",
                                    text_align="right",
                                    value=str(init_yml['dataset_path']).rstrip('/'),
                                    lines=1,
                                    interactive=False,
                                    scale=10
                                )
                                gr.Markdown("""
                                    <br></br>
                                    ## +
                                """)
                                bert_config_box = gr.Textbox(
                                    label="bert_gen配置文件路径",
                                    text_align="left",
                                    value=init_yml['bert_gen']['config_path'],
                                    lines=1,
                                    interactive=True,
                                    scale=10
                                )
                            with gr.Row():
                                slider_bert_nps = gr.Slider(
                                    label="bert_gen并行处理数",
                                    minimum=1,
                                    maximum=12,
                                    step=1,
                                    value=init_yml['bert_gen']['num_processes']
                                )
                                dropdown_bert_dev = gr.Dropdown(
                                    label="bert_gen处理设备",
                                    choices=['cuda', 'cpu'],
                                    value=init_yml['bert_gen']['device']
                                )
                                radio_bert_multi = gr.Radio(
                                    label="使用多卡推理",
                                    choices=[True, False],
                                    value=False
                                )
                            with gr.Row():
                                bert_config_btn = gr.Button(
                                    value="确认更改bert配置项"
                                )
                                bert_gen_btn = gr.Button(
                                    value="Go! Bert Gen!",
                                    variant="primary"
                                )
                            with gr.Row():
                                bert_status = gr.Textbox(
                                    label="状态信息"
                                )
                        with gr.TabItem("5. emo_gen"):
                            with gr.Row():
                                emo_config_box = gr.Textbox(
                                    label="emo_gen配置文件路径",
                                    info="找一找你的config.json路径,相对于数据集路径",
                                    value=init_yml['emo_gen']['config_path'],
                                    lines=1,
                                    interactive=True,
                                    scale=10
                                )
                            with gr.Row():
                                slider_emo_nps = gr.Slider(
                                    label="emo_gen并行处理数",
                                    info="最好预留2个以上的核数空闲，防卡死",
                                    minimum=1,
                                    maximum=32,
                                    step=1,
                                    value=init_yml['emo_gen']['num_processes']
                                )
                                dropdown_emo_device = gr.Dropdown(
                                    label="emo_gen使用设备",
                                    info="可选cpu或cuda",
                                    choices=['cpu', 'cuda'],
                                    value='cuda'
                                )
                            with gr.Row():
                                emo_config_btn = gr.Button(
                                    value="更新emo配置"
                                )
                                emo_gen_btn = gr.Button(
                                    value="Emo Gen!",
                                    variant="primary"
                                )
                            with gr.Row():
                                emo_status = gr.Textbox(
                                    label="状态信息"
                                )

                with gr.TabItem("训练界面"):
                    with gr.Tabs():
                        with gr.TabItem("训练配置文件路径"):
                            with gr.Row():
                                train_dataset_box_1 = gr.Textbox(
                                    label="数据集存放路径",
                                    text_align="right",
                                    value=str(init_yml['dataset_path']).rstrip('/'),
                                    lines=1,
                                    interactive=False,
                                    scale=20
                                )
                                gr.Markdown("""
                                    <br></br>
                                    ## +
                                """)
                                train_config_box = gr.Textbox(
                                    label="train_ms配置文件路径",
                                    text_align="left",
                                    value=init_yml['train_ms']['config_path'],
                                    lines=1,
                                    interactive=True,
                                    scale=20
                                )
                            with gr.Row():
                                train_dataset_box_2 = gr.Textbox(
                                    label="数据集存放路径",
                                    text_align="right",
                                    value=str(init_yml['dataset_path']).rstrip('/'),
                                    lines=1,
                                    interactive=False,
                                    scale=20
                                )
                                gr.Markdown("""
                                    <br></br>
                                    ## +
                                """)
                                train_model_box = gr.Textbox(
                                    label="train_ms模型文件夹路径",
                                    value=init_yml['train_ms']['model'],
                                    lines=1,
                                    interactive=True,
                                    scale=20
                                )
                            with gr.Row():
                                train_ms_path_btn = gr.Button(
                                    value="更改训练路径配置"
                                )
                            CheckboxGroup_train_models = check_base_models()
                            check_pth_btn3 = gr.Button(value="检查训练底模状态")
                        with gr.TabItem("训练参数设置"):
                            with gr.Row():
                                slider_batch_size = gr.Slider(minimum=1, maximum=40, value=4, step=1,
                                                              label="batch_size 批处理大小")
                                slider_keep_ckpts = gr.Slider(minimum=1, maximum=20, value=5, step=1,
                                                              label="最多保存n个最新模型，超过则删除最早的")
                            with gr.Row():
                                slider_log_interval = gr.Slider(minimum=50, maximum=3000, value=200, step=50,
                                                                label="log_interval 打印日志步数间隔")
                                slider_eval_interval = gr.Slider(minimum=100, maximum=5000, value=1000, step=50,
                                                                 label="eval_interval 保存模型步数间隔")
                            with gr.Row():
                                slider_epochs = gr.Slider(minimum=50, maximum=2000, value=100, step=50,
                                                          label="epochs 训练轮数")
                                slider_lr = gr.Slider(minimum=0.0001, maximum=0.0010, value=0.0003, step=0.0001,
                                                      label="learning_rate 初始学习率")
                            with gr.Row():
                                dropdown_version = gr.Dropdown(
                                    label="模型版本选择",
                                    info="推荐使用最新版底模和版本训练",
                                    choices=['2.0', '1.1.1', '1.1.0', '1.0.1'],
                                    value='2.0'
                                )
                            with gr.Row():
                                train_ms_load_btn = gr.Button(
                                    value="加载训练参数配置",
                                    variant="primary"
                                )
                                train_ms_param_btn = gr.Button(
                                    value="更改训练参数配置",
                                    variant="primary"
                                )
                            with gr.Row():
                                train_btn = gr.Button(value="3.1 点击开始训练", variant="primary")
                                train_btn_2 = gr.Button(value="3.2 继续训练", variant="primary")
                                stop_train_btn = gr.Button(value="终止训练", variant="secondary")
                            with gr.Row():
                                train_output_box = gr.Textbox(
                                    label="状态信息",
                                    lines=1,
                                    autoscroll=True
                                )
                        with gr.TabItem("TensorBoard"):
                            with gr.Row():
                                gr.Markdown("""
                                    ### Tensorboard的logdir 默认为训练的models路径
                                    ### 请在前一节 `训练配置文件路径` 查看
                                """)
                            with gr.Row():
                                open_tb_btn = gr.Button("开启Tensorboard")
                                stop_tb_btn = gr.Button("关闭Tensorboard")
                            with gr.Row():
                                tb_output_box = gr.Textbox(
                                    label="状态信息",
                                    lines=1,
                                    autoscroll=True
                                )
                with gr.TabItem("推理界面"):
                    with gr.Tabs():
                        with gr.TabItem("模型选择"):
                            with gr.Row():
                                dropdown_infer_model = gr.Dropdown(
                                    label="选择推理模型",
                                    info="默认选择预处理阶段配置的文件夹内容; 也可以自己输入路径。",
                                    interactive=True,
                                    allow_custom_value=True,
                                )
                                dropdown_infer_config = gr.Dropdown(
                                    label="选择配置文件",
                                    info="默认选择预处理阶段配置的文件夹内容; 也可以自己输入路径。",
                                    interactive=True,
                                    allow_custom_value=True,
                                )
                            with gr.Row():
                                dropdown_model_fresh_btn = gr.Button(
                                    value="刷新推理模型列表"
                                )
                            with gr.Row():
                                webui_port_box = gr.Textbox(
                                    label="WebUI推理的端口号",
                                    placeholder="范围:[0, 65535]",
                                    max_lines=1,
                                    lines=1,
                                    value=init_yml['webui']['port'],
                                    interactive=True
                                )
                                infer_ver_box = gr.Dropdown(
                                    label="更改推理版本",
                                    info="已经实现兼容推理，请选择合适的版本",
                                    choices=['2.0', '1.1.1', '1.1.0', '1.0.1'],
                                    value='2.0'
                                )
                            with gr.Row():
                                radio_webui_share = gr.Radio(
                                    label="公开",
                                    info="是否公开部署，对外网开放",
                                    choices=[True, False],
                                    value=init_yml['webui']['share']
                                )
                                radio_webui_debug = gr.Radio(
                                    label="调试模式",
                                    info="是否开启debug模式",
                                    choices=[True, False],
                                    value=init_yml['webui']['debug']
                                )
                            with gr.Row():
                                infer_config_btn = gr.Button(
                                    value="更新推理配置文件"
                                )
                                stop_infer_btn = gr.Button(
                                    value="结束WebUI推理"
                                )
                            with gr.Row():
                                infer_webui_btn = gr.Button(
                                    value="开启WebUI推理",
                                    variant="primary"
                                )
                            with gr.Row():
                                infer_webui_box = gr.Textbox(
                                    label="提示信息",
                                    interactive=False
                                )

                        with gr.TabItem("模型压缩"):
                            with gr.Row():
                                compress_config = gr.Textbox(
                                    label="压缩配置文件",
                                    info="模型对应的config.json"
                                )
                            with gr.Row():
                                compress_input_path = gr.Textbox(
                                    label="待压缩模型路径",
                                    info="所谓的模型是：G_{步数}.pth"
                                )
                            with gr.Row():
                                compress_output_path = gr.Textbox(
                                    label="输出模型路径",
                                    info="输出为：G_{步数}_release.pth",
                                    value="在待压缩模型路径的同一文件夹下",
                                    interactive=False
                                )
                            with gr.Row():
                                compress_btn = gr.Button(
                                    value="压缩模型",
                                    variant="primary"
                                )
                            with gr.Row():
                                compress_status = gr.Textbox(
                                    label="状态信息"
                                )
            with gr.Tabs():
                with gr.TabItem("yaml配置文件状态"):
                    code_config_yml = gr.Code(
                        interactive=False,
                        label=yml_config,
                        value=load_yaml_data_in_raw(),
                        language="yaml",
                        elem_id="yml_code"
                    )
                with gr.TabItem("带注释的yaml配置文件"):
                    code_default_yml = gr.Code(
                        interactive=False,
                        label=default_yaml_path,
                        value=load_yaml_data_in_raw(default_yaml_path),
                        language="yaml",
                        elem_id="yml_code"
                    )
                with gr.TabItem("训练的json配置文件"):
                    code_train_config_json = gr.Code(
                        interactive=False,
                        label=default_config_path,
                        value=load_json_data_in_raw(default_config_path),
                        language="json",
                        elem_id="json_code"
                    )
                with gr.TabItem("推理的json配置文件"):
                    code_infer_config_json = gr.Code(
                        interactive=False,
                        label=default_config_path,
                        value=load_json_data_in_raw(default_config_path),
                        language="json",
                        elem_id="json_code"
                    )
                with gr.TabItem("其他状态"):
                    code_gpu_json = gr.Code(label="本机资源使用情况",
                                            interactive=False,
                                            value=get_status(),
                                            language="json",
                                            elem_id="gpu_code")
                    gpu_json_btn = gr.Button(
                        value="刷新本机状态"
                    )

        check_pth_btn1.click(fn=check_bert_models,
                             inputs=[],
                             outputs=[CheckboxGroup_bert_models])
        check_pth_btn2.click(fn=check_emo_models,
                             inputs=[],
                             outputs=[CheckboxGroup_emo_models])
        check_pth_btn3.click(fn=check_base_models,
                             inputs=[],
                             outputs=[CheckboxGroup_train_models])
        data_path_btn.click(fn=modify_data_path,
                            inputs=[dropdown_data_path],
                            outputs=[dropdown_data_path, bert_dataset_box,
                                     train_dataset_box_1, train_dataset_box_2,
                                     code_config_yml, CheckboxGroup_train_models])
        preprocess_config_btn.click(fn=modify_preprocess_param,
                                    inputs=[dropdown_filelist_path, preprocess_config_box,
                                            slider_val_per_spk, slider_max_val_total],
                                    outputs=[dropdown_filelist_path, code_config_yml])
        preprocess_text_btn.click(fn=do_preprocess_text,
                                  inputs=[],
                                  outputs=[label_status])
        resample_config_btn.click(fn=modify_resample_path,
                                  inputs=[resample_in_box, resample_out_box, dropdown_resample_sr],
                                  outputs=[resample_in_box, resample_out_box, resample_status, dropdown_resample_sr,
                                           code_config_yml])
        resample_btn.click(fn=do_resample,
                           inputs=[slider_resample_nps],
                           outputs=[resample_status])
        transcribe_btn.click(fn=do_transcript,
                             inputs=[dropdown_lang, slider_transcribe],
                             outputs=[preprocess_status_box])
        extract_list_btn.click(fn=do_extract,
                               inputs=[resample_in_box, dropdown_lang, unclean_box, char_name_box],
                               outputs=[preprocess_status_box])
        clean_trans_btn.click(fn=do_clean_list,
                              inputs=[clean_txt_box, unclean_box, clean_box],
                              outputs=[preprocess_status_box])
        bert_config_btn.click(fn=modify_bert_config,
                              inputs=[bert_config_box, slider_bert_nps, dropdown_bert_dev, radio_bert_multi],
                              outputs=[bert_config_box, slider_bert_nps, dropdown_bert_dev, radio_bert_multi,
                                       code_config_yml])
        bert_gen_btn.click(fn=do_bert_gen,
                           inputs=[],
                           outputs=[bert_status])
        emo_config_btn.click(fn=modify_emo_gen,
                             inputs=[emo_config_box, slider_emo_nps, dropdown_emo_device],
                             outputs=[emo_status, code_config_yml])
        emo_gen_btn.click(fn=do_emo_gen,
                          inputs=[],
                          outputs=[emo_status])
        train_ms_load_btn.click(fn=load_train_param,
                                inputs=[train_config_box],
                                outputs=[train_output_box, code_train_config_json,
                                         slider_batch_size, slider_keep_ckpts,
                                         slider_log_interval, slider_eval_interval,
                                         slider_epochs, slider_lr, dropdown_version,
                                         ])
        train_ms_path_btn.click(fn=modify_train_path,
                                inputs=[train_model_box, train_config_box],
                                outputs=[train_model_box, train_config_box, code_config_yml,
                                         CheckboxGroup_train_models])
        train_ms_param_btn.click(fn=modify_train_param,
                                 inputs=[slider_batch_size, slider_keep_ckpts,
                                         slider_log_interval, slider_eval_interval,
                                         slider_epochs, slider_lr, dropdown_version],
                                 outputs=[train_output_box, code_train_config_json])
        train_btn.click(fn=do_my_train,
                        inputs=[],
                        outputs=[train_output_box])
        train_btn_2.click(fn=do_my_train,
                          inputs=[],
                          outputs=[train_output_box])
        stop_train_btn.click(fn=stop_train_ms,
                             inputs=[],
                             outputs=[train_output_box])
        open_tb_btn.click(fn=do_tensorboard,
                          inputs=[],
                          outputs=[tb_output_box])
        stop_tb_btn.click(fn=stop_tensorboard,
                          inputs=[],
                          outputs=[tb_output_box])
        dropdown_model_fresh_btn.click(fn=list_infer_models,
                                       inputs=[],
                                       outputs=[dropdown_infer_model, dropdown_infer_config])
        infer_config_btn.click(
            fn=modify_infer_param,
            inputs=[dropdown_infer_model, dropdown_infer_config,
                    webui_port_box, radio_webui_share, radio_webui_debug, infer_ver_box],
            outputs=[infer_webui_box, code_config_yml, code_infer_config_json]
        )
        infer_webui_btn.click(
            fn=do_webui_infer,
            inputs=[],
            outputs=[infer_webui_box]
        )
        compress_btn.click(
            fn=compress_model,
            inputs=[compress_config, compress_input_path, compress_output_path],
            outputs=[compress_status]
        )
        stop_infer_btn.click(
            fn=stop_webui_infer,
            inputs=[],
            outputs=[infer_webui_box]
        )
        gpu_json_btn.click(
            fn=get_gpu_status,
            inputs=[],
            outputs=[code_gpu_json]
        )
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    webbrowser.open("http://127.0.0.1:6006")
    app.launch(share=False, server_port=6006)
