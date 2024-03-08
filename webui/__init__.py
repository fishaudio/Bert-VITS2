from .dataset import create_dataset_app
from .inference import create_inference_app
from .merge import create_merge_app
from .style_vectors import create_style_vectors_app
from .train import create_train_app


class TrainSettings:
    def __init__(self, setting_json):
        self.setting_json = setting_json

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass
