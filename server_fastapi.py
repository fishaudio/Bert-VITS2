"""
api服务 多版本多模型 fastapi实现
"""
import logging
import gc
import random

from pydantic import BaseModel
import gradio
import numpy as np
import utils
from fastapi import FastAPI, Query, Request
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from scipy.io import wavfile
import uvicorn
import torch
import webbrowser
import psutil
import GPUtil
from typing import Dict, Optional, List, Set
import os
from tools.log import logger
from urllib.parse import unquote

from infer import infer, get_net_g, latest_version
import tools.translate as trans
from re_matching import cut_sent


from config import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Model:
    """模型封装类"""

    def __init__(self, config_path: str, model_path: str, device: str, language: str):
        self.config_path: str = os.path.normpath(config_path)
        self.model_path: str = os.path.normpath(model_path)
        self.device: str = device
        self.language: str = language
        self.hps = utils.get_hparams_from_file(config_path)
        self.spk2id: Dict[str, int] = self.hps.data.spk2id  # spk - id 映射字典
        self.id2spk: Dict[int, str] = dict()  # id - spk 映射字典
        for speaker, speaker_id in self.hps.data.spk2id.items():
            self.id2spk[speaker_id] = speaker
        self.version: str = (
            self.hps.version if hasattr(self.hps, "version") else latest_version
        )
        self.net_g = get_net_g(
            model_path=model_path,
            version=self.version,
            device=device,
            hps=self.hps,
        )

    def to_dict(self) -> Dict[str, any]:
        return {
            "config_path": self.config_path,
            "model_path": self.model_path,
            "device": self.device,
            "language": self.language,
            "spk2id": self.spk2id,
            "id2spk": self.id2spk,
            "version": self.version,
        }


class Models:
    def __init__(self):
        self.models: Dict[int, Model] = dict()
        self.num = 0
        # spkInfo[角色名][模型id] = 角色id
        self.spk_info: Dict[str, Dict[int, int]] = dict()
        self.path2ids: Dict[str, Set[int]] = dict()  # 路径指向的model的id

    def init_model(
        self, config_path: str, model_path: str, device: str, language: str
    ) -> int:
        """
        初始化并添加一个模型

        :param config_path: 模型config.json路径
        :param model_path: 模型路径
        :param device: 模型推理使用设备
        :param language: 模型推理默认语言
        """
        # 若路径中的模型已存在，则不添加模型，若不存在，则进行初始化。
        model_path = os.path.realpath(model_path)
        if model_path not in self.path2ids.keys():
            self.path2ids[model_path] = {self.num}
            self.models[self.num] = Model(
                config_path=config_path,
                model_path=model_path,
                device=device,
                language=language,
            )
            logger.success(f"添加模型{model_path}，使用配置文件{os.path.realpath(config_path)}")
        else:
            # 获取一个指向id
            m_id = next(iter(self.path2ids[model_path]))
            self.models[self.num] = self.models[m_id]
            self.path2ids[model_path].add(self.num)
            logger.success("模型已存在，添加模型引用。")
        # 添加角色信息
        for speaker, speaker_id in self.models[self.num].spk2id.items():
            if speaker not in self.spk_info.keys():
                self.spk_info[speaker] = {self.num: speaker_id}
            else:
                self.spk_info[speaker][self.num] = speaker_id
        # 修改计数
        self.num += 1
        return self.num - 1

    def del_model(self, index: int) -> Optional[int]:
        """删除对应序号的模型，若不存在则返回None"""
        if index not in self.models.keys():
            return None
        # 删除角色信息
        for speaker, speaker_id in self.models[index].spk2id.items():
            self.spk_info[speaker].pop(index)
            if len(self.spk_info[speaker]) == 0:
                # 若对应角色的所有模型都被删除，则清除该角色信息
                self.spk_info.pop(speaker)
        # 删除路径信息
        model_path = os.path.realpath(self.models[index].model_path)
        self.path2ids[model_path].remove(index)
        if len(self.path2ids[model_path]) == 0:
            self.path2ids.pop(model_path)
            logger.success(f"删除模型{model_path}, id = {index}")
        else:
            logger.success(f"删除模型引用{model_path}, id = {index}")
        # 删除模型
        self.models.pop(index)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return index

    def get_models(self):
        """获取所有模型"""
        return self.models


if __name__ == "__main__":
    app = FastAPI()
    app.logger = logger
    # 挂载静态文件
    StaticDir: str = "./Web"
    dirs = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
    files = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
    for dirName in dirs:
        app.mount(
            f"/{dirName}",
            StaticFiles(directory=f"./{StaticDir}/{dirName}"),
            name=dirName,
        )
    loaded_models = Models()
    # 加载模型
    models_info = config.server_config.models
    for model_info in models_info:
        loaded_models.init_model(
            config_path=model_info["config"],
            model_path=model_info["model"],
            device=model_info["device"],
            language=model_info["language"],
        )

    @app.get("/")
    async def index():
        return FileResponse("./Web/index.html")

    class Text(BaseModel):
        text: str

    @app.post("/voice")
    def voice(
        request: Request,  # fastapi自动注入
        text: Text,
        model_id: int = Query(..., description="模型ID"),  # 模型序号
        speaker_name: str = Query(
            None, description="说话人名"
        ),  # speaker_name与 speaker_id二者选其一
        speaker_id: int = Query(None, description="说话人id，与speaker_name二选一"),
        sdp_ratio: float = Query(0.2, description="SDP/DP混合比"),
        noise: float = Query(0.2, description="感情"),
        noisew: float = Query(0.9, description="音素长度"),
        length: float = Query(1, description="语速"),
        language: str = Query(None, description="语言"),  # 若不指定使用语言则使用默认值
        auto_translate: bool = Query(False, description="自动翻译"),
        auto_split: bool = Query(False, description="自动切分"),
    ):
        """语音接口"""
        text = text.text
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )} text={text}"
        )
        # 检查模型是否存在
        if model_id not in loaded_models.models.keys():
            return {"status": 10, "detail": f"模型model_id={model_id}未加载"}
        # 检查是否提供speaker
        if speaker_name is None and speaker_id is None:
            return {"status": 11, "detail": "请提供speaker_name或speaker_id"}
        elif speaker_name is None:
            # 检查speaker_id是否存在
            if speaker_id not in loaded_models.models[model_id].id2spk.keys():
                return {"status": 12, "detail": f"角色speaker_id={speaker_id}不存在"}
            speaker_name = loaded_models.models[model_id].id2spk[speaker_id]
        # 检查speaker_name是否存在
        if speaker_name not in loaded_models.models[model_id].spk2id.keys():
            return {"status": 13, "detail": f"角色speaker_name={speaker_name}不存在"}
        if language is None:
            language = loaded_models.models[model_id].language
        if auto_translate:
            text = trans.translate(Sentence=text, to_Language=language.lower())
        if not auto_split:
            with torch.no_grad():
                audio = infer(
                    text=text,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise,
                    noise_scale_w=noisew,
                    length_scale=length,
                    sid=speaker_name,
                    language=language,
                    hps=loaded_models.models[model_id].hps,
                    net_g=loaded_models.models[model_id].net_g,
                    device=loaded_models.models[model_id].device,
                )
        else:
            texts = cut_sent(text)
            audios = []
            with torch.no_grad():
                for t in texts:
                    audios.append(
                        infer(
                            text=t,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise,
                            noise_scale_w=noisew,
                            length_scale=length,
                            sid=speaker_name,
                            language=language,
                            hps=loaded_models.models[model_id].hps,
                            net_g=loaded_models.models[model_id].net_g,
                            device=loaded_models.models[model_id].device,
                        )
                    )
                audios.append(np.zeros((int)(44100 * 0.3)))
                audio = np.concatenate(audios)
                audio = gradio.processing_utils.convert_to_16_bit_wav(audio)
        wavContent = BytesIO()
        wavfile.write(
            wavContent, loaded_models.models[model_id].hps.data.sampling_rate, audio
        )
        response = Response(content=wavContent.getvalue(), media_type="audio/wav")
        return response

    @app.get("/voice")
    def voice(
        request: Request,  # fastapi自动注入
        text: str = Query(..., description="输入文字"),
        model_id: int = Query(..., description="模型ID"),  # 模型序号
        speaker_name: str = Query(
            None, description="说话人名"
        ),  # speaker_name与 speaker_id二者选其一
        speaker_id: int = Query(None, description="说话人id，与speaker_name二选一"),
        sdp_ratio: float = Query(0.2, description="SDP/DP混合比"),
        noise: float = Query(0.2, description="感情"),
        noisew: float = Query(0.9, description="音素长度"),
        length: float = Query(1, description="语速"),
        language: str = Query(None, description="语言"),  # 若不指定使用语言则使用默认值
        auto_translate: bool = Query(False, description="自动翻译"),
        auto_split: bool = Query(False, description="自动切分"),
    ):
        """语音接口"""
        logger.info(
            f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}"
        )
        # 检查模型是否存在
        if model_id not in loaded_models.models.keys():
            return {"status": 10, "detail": f"模型model_id={model_id}未加载"}
        # 检查是否提供speaker
        if speaker_name is None and speaker_id is None:
            return {"status": 11, "detail": "请提供speaker_name或speaker_id"}
        elif speaker_name is None:
            # 检查speaker_id是否存在
            if speaker_id not in loaded_models.models[model_id].id2spk.keys():
                return {"status": 12, "detail": f"角色speaker_id={speaker_id}不存在"}
            speaker_name = loaded_models.models[model_id].id2spk[speaker_id]
        # 检查speaker_name是否存在
        if speaker_name not in loaded_models.models[model_id].spk2id.keys():
            return {"status": 13, "detail": f"角色speaker_name={speaker_name}不存在"}
        if language is None:
            language = loaded_models.models[model_id].language
        if auto_translate:
            text = trans.translate(Sentence=text, to_Language=language.lower())
        if not auto_split:
            with torch.no_grad():
                audio = infer(
                    text=text,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise,
                    noise_scale_w=noisew,
                    length_scale=length,
                    sid=speaker_name,
                    language=language,
                    hps=loaded_models.models[model_id].hps,
                    net_g=loaded_models.models[model_id].net_g,
                    device=loaded_models.models[model_id].device,
                )
        else:
            texts = cut_sent(text)
            audios = []
            with torch.no_grad():
                for t in texts:
                    audios.append(
                        infer(
                            text=t,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise,
                            noise_scale_w=noisew,
                            length_scale=length,
                            sid=speaker_name,
                            language=language,
                            hps=loaded_models.models[model_id].hps,
                            net_g=loaded_models.models[model_id].net_g,
                            device=loaded_models.models[model_id].device,
                        )
                    )
                audios.append(np.zeros((int)(44100 * 0.3)))
                audio = np.concatenate(audios)
                audio = gradio.processing_utils.convert_to_16_bit_wav(audio)
        wavContent = BytesIO()
        wavfile.write(
            wavContent, loaded_models.models[model_id].hps.data.sampling_rate, audio
        )
        response = Response(content=wavContent.getvalue(), media_type="audio/wav")
        return response

    @app.get("/models/info")
    def get_loaded_models_info(request: Request):
        """获取已加载模型信息"""

        result: Dict[str, Dict] = dict()
        for key, model in loaded_models.models.items():
            result[str(key)] = model.to_dict()
        return result

    @app.get("/models/delete")
    def delete_model(
        request: Request, model_id: int = Query(..., description="删除模型id")
    ):
        """删除指定模型"""
        logger.info(
            f"{request.client.host}:{request.client.port}/models/delete  { unquote(str(request.query_params) )}"
        )
        result = loaded_models.del_model(model_id)
        if result is None:
            return {"status": 14, "detail": f"模型{model_id}不存在，删除失败"}
        return {"status": 0, "detail": "删除成功"}

    @app.get("/models/add")
    def add_model(
        request: Request,
        model_path: str = Query(..., description="添加模型路径"),
        config_path: str = Query(
            None, description="添加模型配置文件路径，不填则使用./config.json或../config.json"
        ),
        device: str = Query("cuda", description="推理使用设备"),
        language: str = Query("ZH", description="模型默认语言"),
    ):
        """添加指定模型：允许重复添加相同路径模型，且不重复占用内存"""
        logger.info(
            f"{request.client.host}:{request.client.port}/models/add  { unquote(str(request.query_params) )}"
        )
        if config_path is None:
            model_dir = os.path.dirname(model_path)
            if os.path.isfile(os.path.join(model_dir, "config.json")):
                config_path = os.path.join(model_dir, "config.json")
            elif os.path.isfile(os.path.join(model_dir, "../config.json")):
                config_path = os.path.join(model_dir, "../config.json")
            else:
                return {
                    "status": 15,
                    "detail": "查询未传入配置文件路径，同时默认路径./与../中不存在配置文件config.json。",
                }
        try:
            model_id = loaded_models.init_model(
                config_path=config_path,
                model_path=model_path,
                device=device,
                language=language,
            )
        except Exception:
            logging.exception("模型加载出错")
            return {
                "status": 16,
                "detail": "模型加载出错，详细查看日志",
            }
        return {
            "status": 0,
            "detail": "模型添加成功",
            "Data": {
                "model_id": model_id,
                "model_info": loaded_models.models[model_id].to_dict(),
            },
        }

    def _get_all_models(root_dir: str = "Data", only_unloaded: bool = False):
        """从root_dir搜索获取所有可用模型"""
        result: Dict[str, List[str]] = dict()
        files = os.listdir(root_dir) + ["."]
        for file in files:
            if os.path.isdir(os.path.join(root_dir, file)):
                sub_dir = os.path.join(root_dir, file)
                # 搜索 "sub_dir" 、 "sub_dir/models" 两个路径
                result[file] = list()
                sub_files = os.listdir(sub_dir)
                model_files = []
                for sub_file in sub_files:
                    relpath = os.path.realpath(os.path.join(sub_dir, sub_file))
                    if only_unloaded and relpath in loaded_models.path2ids.keys():
                        continue
                    if sub_file.endswith(".pth") and sub_file.startswith("G_"):
                        if os.path.isfile(relpath):
                            model_files.append(sub_file)
                # 对模型文件按步数排序
                model_files = sorted(
                    model_files,
                    key=lambda pth: int(pth.lstrip("G_").rstrip(".pth"))
                    if pth.lstrip("G_").rstrip(".pth").isdigit()
                    else 10**10,
                )
                result[file] = model_files
                models_dir = os.path.join(sub_dir, "models")
                model_files = []
                if os.path.isdir(models_dir):
                    sub_files = os.listdir(models_dir)
                    for sub_file in sub_files:
                        relpath = os.path.realpath(os.path.join(models_dir, sub_file))
                        if only_unloaded and relpath in loaded_models.path2ids.keys():
                            continue
                        if sub_file.endswith(".pth") and sub_file.startswith("G_"):
                            if os.path.isfile(os.path.join(models_dir, sub_file)):
                                model_files.append(f"models/{sub_file}")
                    # 对模型文件按步数排序
                    model_files = sorted(
                        model_files,
                        key=lambda pth: int(pth.lstrip("models/G_").rstrip(".pth"))
                        if pth.lstrip("models/G_").rstrip(".pth").isdigit()
                        else 10**10,
                    )
                    result[file] += model_files
                if len(result[file]) == 0:
                    result.pop(file)

        return result

    @app.get("/models/get_unloaded")
    def get_unloaded_models_info(
        request: Request, root_dir: str = Query("Data", description="搜索根目录")
    ):
        """获取未加载模型"""
        logger.info(
            f"{request.client.host}:{request.client.port}/models/get_unloaded  { unquote(str(request.query_params) )}"
        )
        return _get_all_models(root_dir, only_unloaded=True)

    @app.get("/models/get_local")
    def get_local_models_info(
        request: Request, root_dir: str = Query("Data", description="搜索根目录")
    ):
        """获取全部本地模型"""
        logger.info(
            f"{request.client.host}:{request.client.port}/models/get_local  { unquote(str(request.query_params) )}"
        )
        return _get_all_models(root_dir, only_unloaded=False)

    @app.get("/status")
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
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpuInfo.append(
                {
                    "gpu_id": gpu.id,
                    "gpu_load": gpu.load,
                    "gpu_memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                    },
                }
            )
        return {
            "devices": devices,
            "cpu_percent": cpu_percent,
            "memory_total": memory_total,
            "memory_available": memory_available,
            "memory_used": memory_used,
            "memory_percent": memory_percent,
            "gpu": gpuInfo,
        }

    @app.get("/tools/translate")
    def translate(
        request: Request,
        texts: str = Query(..., description="待翻译文本"),
        to_language: str = Query(..., description="翻译目标语言"),
    ):
        """翻译"""
        logger.info(
            f"{request.client.host}:{request.client.port}/tools/translate  { unquote(str(request.query_params) )}"
        )
        return {"texts": trans.translate(Sentence=texts, to_Language=to_language)}

    all_examples: Dict[str, Dict[str, List]] = dict()  # 存放示例

    @app.get("/tools/random_example")
    def random_example(
        request: Request,
        language: str = Query(None, description="指定语言，未指定则随机返回"),
        root_dir: str = Query("Data", description="搜索根目录"),
    ):
        """
        获取一个随机音频+文本，用于对比，音频会从本地目录随机选择。
        """
        logger.info(
            f"{request.client.host}:{request.client.port}/tools/random_example  { unquote(str(request.query_params) )}"
        )
        global all_examples
        # 数据初始化
        if root_dir not in all_examples.keys():
            all_examples[root_dir] = {"ZH": [], "JP": [], "EN": []}

            examples = all_examples[root_dir]

            # 从项目Data目录中搜索train/val.list
            for root, directories, _files in os.walk(root_dir):
                for file in _files:
                    if file in ["train.list", "val.list"]:
                        with open(
                            os.path.join(root, file), mode="r", encoding="utf-8"
                        ) as f:
                            lines = f.readlines()
                            for line in lines:
                                data = line.split("|")
                                if len(data) != 7:
                                    continue
                                # 音频存在 且语言为ZH/EN/JP
                                if os.path.isfile(data[0]) and data[2] in [
                                    "ZH",
                                    "JP",
                                    "EN",
                                ]:
                                    examples[data[2]].append(
                                        {
                                            "text": data[3],
                                            "audio": data[0],
                                            "speaker": data[1],
                                        }
                                    )

        examples = all_examples[root_dir]
        if language is None:
            if len(examples["ZH"]) + len(examples["JP"]) + len(examples["EN"]) == 0:
                return {"status": 17, "detail": "没有加载任何示例数据"}
            else:
                # 随机选一个
                rand_num = random.randint(
                    0,
                    len(examples["ZH"]) + len(examples["JP"]) + len(examples["EN"]) - 1,
                )
                # ZH
                if rand_num < len(examples["ZH"]):
                    return {"status": 0, "Data": examples["ZH"][rand_num]}
                # JP
                if rand_num < len(examples["ZH"]) + len(examples["JP"]):
                    return {
                        "status": 0,
                        "Data": examples["JP"][rand_num - len(examples["ZH"])],
                    }
                # EN
                return {
                    "status": 0,
                    "Data": examples["EN"][
                        rand_num - len(examples["ZH"]) - len(examples["JP"])
                    ],
                }

        else:
            if len(examples[language]) == 0:
                return {"status": 17, "detail": f"没有加载任何{language}数据"}
            return {
                "status": 0,
                "Data": examples[language][
                    random.randint(0, len(examples[language]) - 1)
                ],
            }

    @app.get("/tools/get_audio")
    def get_audio(request: Request, path: str = Query(..., description="本地音频路径")):
        logger.info(
            f"{request.client.host}:{request.client.port}/tools/get_audio  { unquote(str(request.query_params) )}"
        )
        if not os.path.isfile(path):
            return {"status": 18, "detail": "指定音频不存在"}
        if not path.endswith(".wav"):
            return {"status": 19, "detail": "非wav格式文件"}
        return FileResponse(path=path)

    logger.warning("本地服务，请勿将服务端口暴露于外网")
    logger.info(f"api文档地址 http://127.0.0.1:{config.server_config.port}/docs")
    webbrowser.open(f"http://127.0.0.1:{config.server_config.port}")
    uvicorn.run(
        app, port=config.server_config.port, host="0.0.0.0", log_level="warning"
    )
