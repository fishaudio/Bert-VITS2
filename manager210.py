import shutil
import gradio as gr
import os
import webbrowser
import subprocess
#import datetime
import json
#import requests
#import soundfile as sf
import numpy as np
import yaml
#from config import config


current_directory = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists('./Data'):
    os.mkdir('./Data')
 

try:
   with open('file_structure.md', mode="r", encoding="utf-8") as f:
       file_structure_md=f.read()
except:
    file_structure_md='读取错误'



current_yml=None
def get_status():
    global current_yml
    try:
        cfg = yaml.load(open('config.yml'),Loader=yaml.FullLoader)
        current_yml='当前的训练： '+os.path.basename(cfg["dataset_path"])+"\n\n以下是配置文件内容：\n\n"
        with open('config.yml', mode="r", encoding="utf-8", errors='ignore') as f:
            current_y=f.read()
            current_yml+=current_y
    except Exception as error:
        current_yml=error

get_status()

def p0_write_yml(name,val_per_spk,max_val_total,bert_num_processes,emo_num_processes,num_workers):
    if name=='null'or name=='':
        return '请选择！'
    config_path=os.path.join('Data',name,'config.yml')
    config_yml = yaml.load(open(config_path),Loader=yaml.FullLoader)
    config_yml["preprocess_text"]["val_per_spk"] = int(val_per_spk)
    config_yml["preprocess_text"]["max_val_total"] = int(max_val_total)
    config_yml["bert_gen"]["num_processes"] = int(bert_num_processes)
    config_yml["emo_gen"]["num_processes"] = int(emo_num_processes)
    config_yml["train_ms"]["num_workers"]=int(num_workers)
    with open(config_path, 'w', encoding='utf-8') as f:
          yaml.dump(config_yml, f) 
    return 'Success'


list_project = []  
def refresh_project_list():
    global list_project
    list_project = []  
    for item in os.listdir('Data'):
       item_path = os.path.join('Data', item)
       if os.path.isdir(item_path):
          list_project.append(item)
    return (project_name.update(choices=list_project),project_name2.update(choices=list_project),project_name3.update(choices=list_project),'已刷新下拉列表')

for item in os.listdir('Data'):
    item_path = os.path.join('Data', item)
    if os.path.isdir(item_path):
        list_project.append(item)



def p0_mkdir(name):
    if name!='':
       try:   
         path='Data'
         path=os.path.join('Data',name)  
         os.mkdir(path)#path=data/xxx/
         os.mkdir(os.path.join(path,'custom_character_voice'))
         os.mkdir(os.path.join(path,'filelists'))
         os.mkdir(os.path.join(path,'models'))       
         shutil.copy("./configs/config.json",os.path.join(path,"config.json"))
         with open('./configs/default_config.yml', mode="r", encoding="utf-8") as f:
            cfg_yml=yaml.load(f,Loader=yaml.FullLoader)
         cfg_yml["dataset_path"]=path
         cfg_yml["resample"]["in_dir"]="custom_character_voice"
         cfg_yml["resample"]["out_dir"]="custom_character_voice"
         cfg_yml["preprocess_text"]["cleaned_path"]='filelists/cleaned.list'
         cfg_yml["preprocess_text"]["transcription_path"]='filelists/short_character_anno.list'
         cfg_yml["preprocess_text"]["train_path"]='filelists/train.list'
         cfg_yml["preprocess_text"]["val_path"]='filelists/val.list'
         cfg_yml["preprocess_text"]["config_path"]='config.json'
         cfg_yml["bert_gen"]["config_path"]='config.json'
         cfg_yml["emo_gen"]["config_path"]='config.json'
         cfg_yml["train_ms"]["config_path"]='config.json'
         with open(os.path.join(path,"config.yml"), 'w', encoding='utf-8') as f:
            yaml.dump(cfg_yml, f) 
         os.startfile(path)
         refresh_project_list()
         return project_name.update(choices=list_project,value=name),'Success. 已经自动打开了创建好的文件夹。请将音频按说话人分文件夹放入custom_character_voice内。然后进行下一步操作。'
       except Exception as error:
         return error   
    else:
       return '请输入名称！'    

def p0_load_cfg(projectname):
    if projectname=='null'or projectname=='':
        return p0_status.update(value=current_yml),'请选择！'
    try:
        shutil.copy(os.path.join('Data',projectname,'config.yml'),'config.yml')
        get_status()
        return p0_status.update(value=current_yml) ,'Success'
    except Exception as error:
        return p0_status.update(value=current_yml),error

        


def a1a_transcribe(size,lang):
     command = f'venv\python.exe short_audio_transcribe.py --languages {lang} --whisper_size {size}'
     print(command+'\n\n')
     subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
     return '新的命令行窗口已经打开，请关注输出信息。完成后无报错即可关闭进行下一步！'

def a1b_transcribe_genshin():
     command = r"venv\python.exe transcribe_genshin.py"
     print(command+'\n\n')
     subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
     return '新的命令行窗口已经打开，请关注输出信息。完成后无报错即可关闭进行下一步！'

def a2_preprocess_text():
     command = r"venv\python.exe preprocess_text.py"
     print(command+'\n\n')
     subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
     return '新的命令行窗口已经打开，请关注输出信息。完成后无报错即可关闭进行下一步！'

def a3_bert_gen():
     command = r"venv\python.exe bert_gen.py"
     print(command+'\n\n')
     subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
     return '新的命令行窗口已经打开，请关注输出信息。完成后无报错即可关闭进行下一步！'

def a3_emo_gen():
    command=r"venv\python.exe emo_gen.py"
    print(command+'\n\n')
    subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
    return '新的命令行窗口已经打开，请关注输出信息。完成后无报错即可关闭进行下一步！'

def a35_json(bs,lr,interval):
    try:
        with open('config.yml', mode="r", encoding="utf-8") as f:
            cfg_yml=yaml.load(f,Loader=yaml.FullLoader)
        config_path=os.path.join(cfg_yml["dataset_path"],'config.json')
        configjson = json.load(open(config_path))
        configjson["train"]["batch_size"] = int(bs)
        configjson["train"]["learning_rate"] = lr
        configjson["train"]["log_interval"] = int(interval)
        configjson["train"]['eval_interval'] = int(interval)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(configjson, f, indent=2, ensure_ascii=False) 
        return 'success'
    except Exception as error:
        return error 
    
        
def a4a_train():
     command = r"venv\python.exe train_ms.py"
     with open('config.yml', mode="r", encoding="utf-8") as f:
          configyml=yaml.load(f,Loader=yaml.FullLoader)
     cfg_path=os.path.join(configyml["dataset_path"],'config.json')        
     configjson = json.load(open(cfg_path))
     if not configjson["train"]["skip_optimizer"]:
         configjson["train"]["skip_optimizer"]=True
         with open(cfg_path, 'w', encoding='utf-8') as f:
             json.dump(configjson, f, indent=2, ensure_ascii=False)
         print("已经修改配置文件！\n")
     shutil.copy('./pretrained_models/D_0.pth',os.path.join(os.path.join(configyml["dataset_path"],configyml["train_ms"]["model"]),'D_0.pth'))
     shutil.copy('./pretrained_models/G_0.pth',os.path.join(os.path.join(configyml["dataset_path"],configyml["train_ms"]["model"]),'G_0.pth'))
     shutil.copy('./pretrained_models/DUR_0.pth',os.path.join(os.path.join(configyml["dataset_path"],configyml["train_ms"]["model"]),'DUR_0.pth'))
     print("已经复制了底模\n")
     subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
     print(command+'\n\n')
     return '新的命令行窗口已经打开，请关注输出信息。关闭窗口或Ctrl+C终止训练'

def a4b_train_cont():
     command = r"venv\python.exe train_ms.py"
     with open('config.yml', mode="r", encoding="utf-8") as f:
            configyml=yaml.load(f,Loader=yaml.FullLoader)
     cfg_path=os.path.join(configyml["dataset_path"],'config.json')   
     configjson = json.load(open(cfg_path))
     if configjson["train"]["skip_optimizer"]:
         configjson["train"]["skip_optimizer"]=False
         with open(cfg_path, 'w', encoding='utf-8') as f:
             json.dump(configjson, f, indent=2, ensure_ascii=False)
         print("已经修改配置文件！\n")
     subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
     print(command+'\n\n')
     return '新的命令行窗口已经打开，请关注输出信息。关闭窗口或Ctrl+C终止训练'

def start_tb():
     with open('config.yml', mode="r", encoding="utf-8") as f:
            configyml=yaml.load(f,Loader=yaml.FullLoader)
     command = r"venv\python.exe -m tensorboard.main --logdir="+os.path.join(configyml["dataset_path"],configyml["train_ms"]["model"])
     print(command+'\n\n')
     subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
     return '新的命令行窗口已经打开，请关注输出信息。'
ckpt_list = ['null']
'''

try:
   file_list = os.listdir(f'{current_directory}/logs/OUTPUT_MODEL')
   for ck in file_list:
      if os.path.splitext(ck)[-1] == ".pth"and ck[:2] != "D_" and ck[:4] !="DUR_":
         ckpt_list.append(ck)
except Exception as error:
    print("Attention. An error occurred in reading {./logs/OUTPUT_MODEL}.Check if the directory exists.")
    print(error)

def refresh_models_in_logs():
   try:
      file_list = os.listdir(os.path.join(config.dataset_path,config.train_ms_config.model))
      global ckpt_list
      ckpt_list = ['null']
      for ck in file_list:
         if os.path.splitext(ck)[-1] == ".pth"and ck[:2] == "G_":
            ckpt_list.append(ck)
      return (models_logs.update(choices=ckpt_list),"已刷新下拉列表")
   except Exception as error:
      return(models_logs.update(choices=['null']),f"读取失败 {error}")


def c1_infer(file_name):
    if file_name=='null':
        return "请选择模型！"    
    command = f'venv\python.exe webui.py -c ./logs/OUTPUT_MODEL/config.json -m ./logs/OUTPUT_MODEL/{file_name}'
    print(command+'\n\n')
    subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
    return '新的命令行窗口已经打开，请关注输出信息。关闭窗口结束推理服务。'

#backup_ckpt_list=['null']
'''

def c2_refresh_sub_opt(name):  
   try:
       global ckpt_list
       ckpt_list=['null']
       file_list = os.listdir(os.path.join("Data",name,"models"))
       for ck in file_list:
         if os.path.splitext(ck)[-1] == ".pth"and ck[:2] != "D_" and ck[:4] !="DUR_":
            ckpt_list.append(ck)
       return models_in_project.update(choices=ckpt_list,value=ckpt_list[-1])
   except :
       return models_in_project.update(choices=['null'],value='null')

def c2_infer(proj_name,model_name):
    if proj_name=='null' or model_name=='null':
        return '请选择模型！'
   
    path=f'./Data/{proj_name}'
    command = f'venv\python.exe webui.py -c {path}/config.json -m {path}/models/{model_name}'
    print(command+'\n\n')
    subprocess.Popen(['start', 'cmd', '/k', command],cwd=current_directory,shell=True)
    return '新的命令行窗口已经打开，请关注输出信息。关闭窗口结束推理服务。'

def c2_infer_2(proj_name,model_name):
    y=yaml.load(open('config.yml'),Loader=yaml.FullLoader)
    if proj_name=='null' or model_name=='null':
        y["server"]["models"]=[]
    else:
        y["server"]["models"]=[]
        y["server"]["models"].append({"config":os.path.join('Data',proj_name,'config.json'),"device":'cuda',"language": 'ZH',"model":os.path.join('Data',proj_name,'models',model_name),"speakers":[]})
    with open("config.yml", 'w', encoding='utf-8') as f:
        yaml.dump(y, f) 
    subprocess.Popen(['start', 'cmd', '/k','venv\python.exe server_fastapi.py'],cwd=current_directory,shell=True)
    return '已经修改了全局配置文件。新的命令行窗口已经打开，请关注输出信息。关闭窗口结束推理服务。'

def write_version(name,version,cont):
    if name=='null':
        return opt_continue.update(value=False),'请选择！'
    path=os.path.join('Data',name,'config.json')
    try:
       configjson = json.load(open(path))
       if "version" in configjson:
          if not cont:
            return opt_continue.update(value=False),'版本信息已经存在，是不是手滑了？'
          configjson["version"] = version
       else:
           configjson["version"] = version
       with open(path, 'w', encoding='utf-8') as f:
            json.dump(configjson, f, indent=2, ensure_ascii=False) 
       return opt_continue.update(value=False),f'Success. {version}'
    except Exception as e:
       return opt_continue.update(value=False),e





if __name__ == "__main__":
    with gr.Blocks(title="Bert-VITS-2-Manager-WebUI-202") as app:
        gr.Markdown(value="""
        Bert-VITS2训练管理器
                    
        严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。由使用本整合包产生的问题和作者、原作者无关！！！
        
        作者：bilibili@数列解析几何一生之敌
        
        适用于整合包版本V2.0.2,不兼容之前的版本。
                    
        WebUI更新日期：2023.11.23
        """) 
        with gr.Tabs():
           with gr.TabItem("1.创建实验文件夹和加载全局配置"):
               with gr.Row(): 
                   with gr.Column():
                       p0_mkdir_name=gr.Textbox(label="这将创建实验文件夹，请输入实验名称,不要包含中文、特殊字符和保留字符。",
                       placeholder="请输入实验名称,最好不要包含中文、特殊字符和保留字符。",
                       value="",
                       lines=1,
                       interactive=True)                       
                       p0_mkdir_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)
                       p0_mkdir_btn=gr.Button(value="创建！", variant="primary")

                       gr.Markdown(value="<br>")
                                                     
                       project_name = gr.Dropdown(label="实验文件夹", choices=list_project, value='null'if not list_project else list_project[-1],interactive=True) 
                       with gr.Row():
                            p0_val_ps = gr.Number(label="每个说话人的验证集数", value="4",interactive=True)
                            p0_val_tt = gr.Number(label="总的验证集数", value="8",interactive=True)
                            p0_bg_t = gr.Number(label="bert_gen线程数", value="2",interactive=True)
                            p0_emo_t = gr.Number(label="emo_gen线程数", value="2",interactive=True)
                            p0_dataloader = gr.Number(label="data_loader数量", value="4",interactive=True)
                       p0_load_cfg_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)
                       with gr.Row():
                          p0_write_cfg_btn=gr.Button(value="保存更改(但不会自动加载)", variant="primary")
                          p0_load_cfg_btn = gr.Button(value="加载训练配置", variant="primary")
                          p0_load_cfg_refresh_btn=gr.Button(value="刷新选项", variant="secondary")


                   with gr.Column():
                       #p0_current_proj=gr.Textbox(label="当前生效的训练",value="",interactive=False)
                       p0_status=gr.TextArea(label="训练前请确认当前的全局配置信息", value=current_yml,interactive=False)

           with gr.TabItem("2.训练"):
               with gr.Column():                                 
                   whisper_size = gr.Radio(label="选择whisper大小，large需要12G显存", choices=['large','medium','small'], value="medium")
                   language = gr.Radio(label="选择语言(默认中日英，其他不支持的语言会被跳过)", choices=['C','CJ','CJE'], value="CJE") 
               with gr.Column():
                       a1_textbox_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮，二选一",interactive=False)
                       a1a_btn = gr.Button(value="1.a.数据集重采样和标注(使用whisper)", variant="primary")
                       a1b_btn = gr.Button(value="1.b.处理下载的已标注的原神数据集", variant="primary")              
               gr.Markdown(value="<br>")      
               with gr.Row():
                    a2_btn = gr.Button(value="2.文本预处理", variant="primary")
                    a2_textbox_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)

               with gr.Row():
                  a3_btn = gr.Button(value="3-1.生成bert文件", variant="primary")
                  a3_btn_2 = gr.Button(value="3-2.生成emo文件", variant="primary")
                  a3_textbox_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)                                  

               with gr.Row():
                    with gr.Row():
                        a35_textbox_bs = gr.Number(label="批大小", value="8",interactive=True)
                        a35_textbox_lr = gr.Number(label="学习率", value="0.0001",interactive=True)
                        a35_textbox_save = gr.Number(label="保存间隔", value="100",interactive=True)
               with gr.Row():
                   a35_btn = gr.Button(value="写入配置文件", variant="primary")
                   a35_textbox_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)
   
               with gr.Row():              
                    with gr.Row():
                       a4a_btn = gr.Button(value="4a.首次训练", variant="primary")
                       a4b_btn = gr.Button(value="4b.继续训练", variant="primary")                       
                    with gr.Column():
                       a4_textbox_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)
        
               with gr.Row():              
                       tb_btn = gr.Button(value="启动TensorBoard", variant="primary")            
                       tb_textbox_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)                   
           with gr.TabItem("3.启动推理"):
                gr.Markdown(value='工作区模型推理(Data内各实验目录下的模型)') 
                with gr.Row():
                    project_name2 = gr.Dropdown(label="选择实验名", choices=list_project, value='null',interactive=True)
                    models_in_project = gr.Dropdown(label="选择模型", choices=ckpt_list, value='null'if not ckpt_list else ckpt_list[0],interactive=True)
                    with gr.Column():
                       c2_btn = gr.Button(value="启动推理(Gr WebUI)", variant="primary")
                       c2_btn2 = gr.Button(value="启动推理(Hiyori UI)", variant="primary")
                       c2_btn_refresh=gr.Button(value="刷新选项", variant="secondary")
                with gr.Column():
                       c2_textbox_output_text = gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False) 
                project_name2.change(c2_refresh_sub_opt,[project_name2],[models_in_project]) 
           with gr.TabItem("辅助功能"):
                with gr.TabItem("配置文件添加版本号"):
                    gr.Markdown(value='旧版本模型的配置文件添加版本号后方可在2.0版本下使用兼容推理')
                    gr.Markdown(value='按文件结构把配置文件和模型放到对应位置，然后开始操作。')
                    gr.Markdown(value='使用1.1和1.1.1版兼容推理需要安装上一个版本使用的日语bert。')
                    gr.Markdown(value='可选版本为1.0.1,1.1,1.1.1和2.0。旧整合包版本为1.0.1或1.1.1或2.0。')
                    project_name3 = gr.Dropdown(label="选择实验名", choices=list_project, value='null'if not list_project else list_project[0],interactive=True)
                    with gr.Row():
                      with gr.Column():
                          choose_version=gr.Dropdown(label="选择版本", choices=['1.0.1','1.1','1.1.1','2.0','2.1'], value='1.0.1',interactive=True)
                          opt_continue = gr.Checkbox(label="我没手滑")
                          write_ver_btn=gr.Button(value="写入",variant="primary")
                          write_ver_refresh_btn=gr.Button(value="刷新",variant="secondary")
                      with gr.Column():
                          write_ver_textbox=gr.Textbox(label="输出信息", placeholder="点击处理按钮",interactive=False)
                with gr.TabItem("帮助"):
                    with gr.TabItem("文件结构参考"):
                         help=gr.Markdown(value=file_structure_md)

        p0_write_cfg_btn.click(p0_write_yml,
                           inputs=[project_name,p0_val_ps,p0_val_tt,p0_bg_t,p0_emo_t,p0_dataloader],
                           outputs=[
                p0_load_cfg_output_text,
            ],)                
        p0_mkdir_btn.click(p0_mkdir,
                           inputs=[p0_mkdir_name],
                           outputs=[
                               project_name,
                p0_mkdir_output_text,
            ],)
        p0_load_cfg_btn.click(p0_load_cfg,
                           inputs=[project_name],
                           outputs=[p0_status,
                p0_load_cfg_output_text,
            ],)
        p0_load_cfg_refresh_btn.click(refresh_project_list,
                           inputs=[],
                           outputs=[project_name,
                                    project_name2,
                                    project_name3,
                p0_load_cfg_output_text,
            ],)
        a1a_btn.click(
            a1a_transcribe,
            inputs=[whisper_size,language],
            outputs=[
                a1_textbox_output_text,
            ],
        )
 
        a1b_btn.click(
            a1b_transcribe_genshin,
            outputs=[
                a1_textbox_output_text,
            ],
        )
        
        a2_btn.click(
            a2_preprocess_text,
            outputs=[
                a2_textbox_output_text,
            ],
        )
        
        a3_btn.click(
            a3_bert_gen,
            outputs=[
                a3_textbox_output_text,
            ],
        )
        a3_btn_2.click(
            a3_emo_gen,
            outputs=[
                a3_textbox_output_text,
            ],
        )        
        a35_btn.click(
            a35_json,
            inputs=[a35_textbox_bs,a35_textbox_lr,a35_textbox_save],
            outputs=[
                a35_textbox_output_text,
            ],
        )
        a4a_btn.click(
            a4a_train,
            outputs=[
                a4_textbox_output_text,
            ],
        )
        a4b_btn.click(
            a4b_train_cont,
            outputs=[
                a4_textbox_output_text,
            ],
        ) 
        tb_btn.click(
            start_tb,
            outputs=[
                tb_textbox_output_text,
            ],
        )
        '''
        b1_btn.click(
            b1_move_in,
            inputs=[textbox_backup_name],
            outputs=[
                b1_textbox_output_text,
            ],
        )            
        b2_btn_load.click(
            b2_move_out,
            inputs=[models],
            outputs=[
                b2_textbox_output_text,
            ],
        )        
        b2_btn_refresh.click(refresh_backup_list,[],[models,b2_textbox_output_text])
        '''
        '''
        c1_btn.click(
            c1_infer,
            inputs=[models_logs],
            outputs=[speaker,
                c1_textbox_output_text
            ],
        )
        c1_btn_refresh.click(refresh_models_in_logs,[],[models_logs,c1_textbox_output_text])
        '''
        c2_btn.click(
            c2_infer,
            inputs=[project_name2,models_in_project],
            outputs=[
                c2_textbox_output_text,
            ],
        )
        c2_btn2.click(
            c2_infer_2,
            inputs=[project_name2,models_in_project],
            outputs=[
                c2_textbox_output_text,
            ],
        )            
        c2_btn_refresh.click(refresh_project_list,[],[project_name,project_name2,project_name3,c2_textbox_output_text])
        write_ver_refresh_btn.click(refresh_project_list,[],[project_name,project_name2,project_name3,write_ver_textbox])
        write_ver_btn.click(
            write_version,
            inputs=[project_name3,choose_version,opt_continue],
            outputs=[opt_continue,
                write_ver_textbox,
            ],
        ) 
        
webbrowser.open("http://127.0.0.1:6660")
app.launch(server_port=6660)
