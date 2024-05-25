"""
@Desc: 全局配置文件读取
"""

import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from style_bert_vits2.logging import logger


class PathConfig:
    def __init__(self, dataset_root: str, assets_root: str):
        self.dataset_root = Path(dataset_root)
        self.assets_root = Path(assets_root)


# If not cuda available, set possible devices to cpu
cuda_available = torch.cuda.is_available()


class Resample_config:
    """重采样配置"""

    def __init__(self, in_dir: str, out_dir: str, sampling_rate: int = 44100):
        self.sampling_rate = sampling_rate  # 目标采样率
        self.in_dir = Path(in_dir)  # 待处理音频目录路径
        self.out_dir = Path(out_dir)  # 重采样输出路径

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        """从字典中生成实例"""

        # 不检查路径是否有效，此逻辑在resample.py中处理
        data["in_dir"] = dataset_path / data["in_dir"]
        data["out_dir"] = dataset_path / data["out_dir"]

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
        self.transcription_path = Path(transcription_path)
        self.train_path = Path(train_path)
        if cleaned_path == "" or cleaned_path is None:
            self.cleaned_path = self.transcription_path.with_name(
                self.transcription_path.name + ".cleaned"
            )
        else:
            self.cleaned_path = Path(cleaned_path)
        self.val_path = Path(val_path)
        self.config_path = Path(config_path)
        self.val_per_lang = val_per_lang
        self.max_val_total = max_val_total
        self.clean = clean

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        """从字典中生成实例"""

        data["transcription_path"] = dataset_path / data["transcription_path"]
        if data["cleaned_path"] == "" or data["cleaned_path"] is None:
            data["cleaned_path"] = ""
        else:
            data["cleaned_path"] = dataset_path / data["cleaned_path"]
        data["train_path"] = dataset_path / data["train_path"]
        data["val_path"] = dataset_path / data["val_path"]
        data["config_path"] = dataset_path / data["config_path"]

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
        self.config_path = Path(config_path)
        self.num_processes = num_processes
        if not cuda_available:
            device = "cpu"
        self.device = device
        self.use_multi_device = use_multi_device

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        data["config_path"] = dataset_path / data["config_path"]

        return cls(**data)


class Style_gen_config:
    """style_gen 配置"""

    def __init__(
        self,
        config_path: str,
        num_processes: int = 4,
        device: str = "cuda",
    ):
        self.config_path = Path(config_path)
        self.num_processes = num_processes
        if not cuda_available:
            device = "cpu"
        self.device = device

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        data["config_path"] = dataset_path / data["config_path"]

        return cls(**data)


class Train_ms_config:
    """训练配置"""

    def __init__(
        self,
        config_path: str,
        env: dict[str, Any],
        # base: Dict[str, any],
        model_dir: str,
        num_workers: int,
        spec_cache: bool,
        keep_ckpts: int,
    ):
        self.env = env  # 需要加载的环境变量
        # self.base = base  # 底模配置
        self.model_dir = Path(
            model_dir
        )  # 训练模型存储目录，该路径为相对于dataset_path的路径，而非项目根目录
        self.config_path = Path(config_path)  # 配置文件路径
        self.num_workers = num_workers  # worker数量
        self.spec_cache = spec_cache  # 是否启用spec缓存
        self.keep_ckpts = keep_ckpts  # ckpt数量

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        # data["model"] = os.path.join(dataset_path, data["model"])
        data["config_path"] = dataset_path / data["config_path"]

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
        self.device = device
        self.model = Path(model)
        self.config_path = Path(config_path)
        self.port: int = port
        self.share: bool = share
        self.debug: bool = debug
        self.language_identification_library: str = language_identification_library

    @classmethod
    def from_dict(cls, dataset_path: Path, data: dict[str, Any]):
        data["config_path"] = dataset_path / data["config_path"]
        data["model"] = dataset_path / data["model"]
        return cls(**data)


class Server_config:
    def __init__(
        self,
        port: int = 5000,
        device: str = "cuda",
        limit: int = 100,
        language: str = "JP",
        origins: list[str] = ["*"],
    ):
        self.port: int = port
        if not cuda_available:
            device = "cpu"
        self.device: str = device
        self.language: str = language
        self.limit: int = limit
        self.origins: list[str] = origins

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


class Translate_config:
    """翻译api配置"""

    def __init__(self, app_key: str, secret_key: str):
        self.app_key = app_key
        self.secret_key = secret_key

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)


class Config:
    def __init__(self, config_path: str, path_config: PathConfig):
        if not Path(config_path).exists():
            shutil.copy(src="default_config.yml", dst=config_path)
            logger.info(
                f"A configuration file {config_path} has been generated based on the default configuration file default_config.yml."
            )
            logger.info(
                "Please do not modify default_config.yml. Instead, modify config.yml."
            )
            # sys.exit(0)
        with open(config_path, encoding="utf-8") as file:
            yaml_config: dict[str, Any] = yaml.safe_load(file.read())
            model_name: str = yaml_config["model_name"]
            self.model_name: str = model_name
            if "dataset_path" in yaml_config:
                dataset_path = Path(yaml_config["dataset_path"])
            else:
                dataset_path = path_config.dataset_root / model_name
            self.dataset_path = dataset_path
            self.dataset_root = path_config.dataset_root
            self.assets_root = path_config.assets_root
            self.out_dir = self.assets_root / model_name
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


# Load and initialize the configuration


def get_path_config() -> PathConfig:
    path_config_path = Path("configs/paths.yml")
    if not path_config_path.exists():
        shutil.copy(src="configs/default_paths.yml", dst=path_config_path)
        logger.info(
            f"A configuration file {path_config_path} has been generated based on the default configuration file default_paths.yml."
        )
        logger.info(
            "Please do not modify configs/default_paths.yml. Instead, modify configs/paths.yml."
        )
    with open(path_config_path, encoding="utf-8") as file:
        path_config_dict: dict[str, str] = yaml.safe_load(file.read())
    return PathConfig(**path_config_dict)


def get_config() -> Config:
    path_config = get_path_config()
    try:
        config = Config("config.yml", path_config)
    except (TypeError, KeyError):
        logger.warning("Old config.yml found. Replace it with default_config.yml.")
        shutil.copy(src="default_config.yml", dst="config.yml")
        config = Config("config.yml", path_config)

    return config
