"""
@Desc: 全局配置文件读取
"""

import os
import shutil
from typing import Dict, List

import torch
import yaml

from style_bert_vits2.logging import logger


# If not cuda available, set possible devices to cpu
cuda_available = torch.cuda.is_available()


class Resample_config:
    """重采样配置"""

    def __init__(self, in_dir: str, out_dir: str, sampling_rate: int = 44100):
        self.sampling_rate: int = sampling_rate  # 目标采样率
        self.in_dir: str = in_dir  # 待处理音频目录路径
        self.out_dir: str = out_dir  # 重采样输出路径

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        """从字典中生成实例"""

        # 不检查路径是否有效，此逻辑在resample.py中处理
        data["in_dir"] = os.path.join(dataset_path, data["in_dir"])
        data["out_dir"] = os.path.join(dataset_path, data["out_dir"])

        return cls(**data)


class Preprocess_text_config:
    """数据预处理配置"""

    def __init__(
        self,
        transcription_path: str,
        cleaned_path: str,
        train_path: str,
        val_path: str,
        config_path: str,
        val_per_lang: int = 5,
        max_val_total: int = 10000,
        clean: bool = True,
    ):
        self.transcription_path: str = (
            transcription_path  # 原始文本文件路径，文本格式应为{wav_path}|{speaker_name}|{language}|{text}。
        )
        self.cleaned_path: str = (
            cleaned_path  # 数据清洗后文本路径，可以不填。不填则将在原始文本目录生成
        )
        self.train_path: str = (
            train_path  # 训练集路径，可以不填。不填则将在原始文本目录生成
        )
        self.val_path: str = (
            val_path  # 验证集路径，可以不填。不填则将在原始文本目录生成
        )
        self.config_path: str = config_path  # 配置文件路径
        self.val_per_lang: int = val_per_lang  # 每个speaker的验证集条数
        self.max_val_total: int = (
            max_val_total  # 验证集最大条数，多于的会被截断并放到训练集中
        )
        self.clean: bool = clean  # 是否进行数据清洗

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        """从字典中生成实例"""

        data["transcription_path"] = os.path.join(
            dataset_path, data["transcription_path"]
        )
        if data["cleaned_path"] == "" or data["cleaned_path"] is None:
            data["cleaned_path"] = None
        else:
            data["cleaned_path"] = os.path.join(dataset_path, data["cleaned_path"])
        data["train_path"] = os.path.join(dataset_path, data["train_path"])
        data["val_path"] = os.path.join(dataset_path, data["val_path"])
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class Bert_gen_config:
    """bert_gen 配置"""

    def __init__(
        self,
        config_path: str,
        num_processes: int = 1,
        device: str = "cuda",
        use_multi_device: bool = False,
    ):
        self.config_path = config_path
        self.num_processes = num_processes
        if not cuda_available:
            device = "cpu"
        self.device = device
        self.use_multi_device = use_multi_device

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class Style_gen_config:
    """style_gen 配置"""

    def __init__(
        self,
        config_path: str,
        num_processes: int = 4,
        device: str = "cuda",
    ):
        self.config_path = config_path
        self.num_processes = num_processes
        if not cuda_available:
            device = "cpu"
        self.device = device

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class Train_ms_config:
    """训练配置"""

    def __init__(
        self,
        config_path: str,
        env: Dict[str, any],
        # base: Dict[str, any],
        model_dir: str,
        num_workers: int,
        spec_cache: bool,
        keep_ckpts: int,
    ):
        self.env = env  # 需要加载的环境变量
        # self.base = base  # 底模配置
        self.model_dir = model_dir  # 训练模型存储目录，该路径为相对于dataset_path的路径，而非项目根目录
        self.config_path = config_path  # 配置文件路径
        self.num_workers = num_workers  # worker数量
        self.spec_cache = spec_cache  # 是否启用spec缓存
        self.keep_ckpts = keep_ckpts  # ckpt数量

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        # data["model"] = os.path.join(dataset_path, data["model"])
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class Webui_config:
    """webui 配置 (for webui.py, not supported now)"""

    def __init__(
        self,
        device: str,
        model: str,
        config_path: str,
        language_identification_library: str,
        port: int = 7860,
        share: bool = False,
        debug: bool = False,
    ):
        if not cuda_available:
            device = "cpu"
        self.device: str = device
        self.model: str = model  # 端口号
        self.config_path: str = config_path  # 是否公开部署，对外网开放
        self.port: int = port  # 是否开启debug模式
        self.share: bool = share  # 模型路径
        self.debug: bool = debug  # 配置文件路径
        self.language_identification_library: str = (
            language_identification_library  # 语种识别库
        )

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["config_path"] = os.path.join(dataset_path, data["config_path"])
        data["model"] = os.path.join(dataset_path, data["model"])
        return cls(**data)


class Server_config:
    def __init__(
        self,
        port: int = 5000,
        device: str = "cuda",
        limit: int = 100,
        language: str = "JP",
        origins: List[str] = None,
    ):
        self.port: int = port
        if not cuda_available:
            device = "cpu"
        self.device: str = device
        self.language: str = language
        self.limit: int = limit
        self.origins: List[str] = origins

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


class Translate_config:
    """翻译api配置"""

    def __init__(self, app_key: str, secret_key: str):
        self.app_key = app_key
        self.secret_key = secret_key

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)


class Config:
    def __init__(self, config_path: str, path_config: dict[str, str]):
        if not os.path.isfile(config_path) and os.path.isfile("default_config.yml"):
            shutil.copy(src="default_config.yml", dst=config_path)
            logger.info(
                f"A configuration file {config_path} has been generated based on the default configuration file default_config.yml."
            )
            logger.info(
                "If you have no special needs, please do not modify default_config.yml."
            )
            # sys.exit(0)
        with open(config_path, "r", encoding="utf-8") as file:
            yaml_config: Dict[str, any] = yaml.safe_load(file.read())
            model_name: str = yaml_config["model_name"]
            self.model_name: str = model_name
            if "dataset_path" in yaml_config:
                dataset_path = yaml_config["dataset_path"]
            else:
                dataset_path = os.path.join(path_config["dataset_root"], model_name)
            self.dataset_path: str = dataset_path
            self.assets_root: str = path_config["assets_root"]
            self.out_dir = os.path.join(self.assets_root, model_name)
            self.resample_config: Resample_config = Resample_config.from_dict(
                dataset_path, yaml_config["resample"]
            )
            self.preprocess_text_config: Preprocess_text_config = (
                Preprocess_text_config.from_dict(
                    dataset_path, yaml_config["preprocess_text"]
                )
            )
            self.bert_gen_config: Bert_gen_config = Bert_gen_config.from_dict(
                dataset_path, yaml_config["bert_gen"]
            )
            self.style_gen_config: Style_gen_config = Style_gen_config.from_dict(
                dataset_path, yaml_config["style_gen"]
            )
            self.train_ms_config: Train_ms_config = Train_ms_config.from_dict(
                dataset_path, yaml_config["train_ms"]
            )
            self.webui_config: Webui_config = Webui_config.from_dict(
                dataset_path, yaml_config["webui"]
            )
            self.server_config: Server_config = Server_config.from_dict(
                yaml_config["server"]
            )
            # self.translate_config: Translate_config = Translate_config.from_dict(
            #     yaml_config["translate"]
            # )


with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    # Should contain the following keys:
    # - dataset_root: the root directory of the dataset, default to "Data"
    # - assets_root: the root directory of the assets, default to "model_assets"


try:
    config = Config("config.yml", path_config)
except (TypeError, KeyError):
    logger.warning("Old config.yml found. Replace it with default_config.yml.")
    shutil.copy(src="default_config.yml", dst="config.yml")
    config = Config("config.yml", path_config)
