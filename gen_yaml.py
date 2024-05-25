import argparse
import os
import shutil

import yaml


parser = argparse.ArgumentParser(
    description="config.ymlの生成。あらかじめ前準備をしたデータをバッチファイルなどで連続で学習する時にtrain_ms.pyより前に使用する。"
)
# そうしないと最後の前準備したデータで学習してしまう
parser.add_argument("--model_name", type=str, help="Model name", required=True)
parser.add_argument(
    "--dataset_path",
    type=str,
    help="Dataset path(example: Data\\your_model_name)",
    required=True,
)
args = parser.parse_args()


def gen_yaml(model_name, dataset_path):
    if not os.path.exists("config.yml"):
        shutil.copy(src="default_config.yml", dst="config.yml")
    with open("config.yml", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = dataset_path
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)


if __name__ == "__main__":
    gen_yaml(args.model_name, args.dataset_path)
