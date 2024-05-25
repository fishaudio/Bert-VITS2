import shutil

import gradio as gr
from huggingface_hub import snapshot_download

from config import get_path_config
from style_bert_vits2.logging import logger


assets_root = get_path_config().assets_root

how_to_md = """
## 使い方

学習済みモデルの共有サイト Hugging Face 🤗 に公開されているモデルをダウンロードして音声合成で使えるようにします。

例:

- `https://huggingface.co/username/my_sbv2_model`を指定すると、`model_assets/username-my_sbv2_model`に全体がダウンロードされます。
- `https://huggingface.co/username/my_sbv2_models/tree/main/model1`を指定すると、`model_assets/username-my_sbv2_models-model1`に`model1`フォルダがダウンロードされます。

**注意**

- **必ずモデルの利用には（掲載があれば）利用規約を確認してください。** ダウンロード後にREADMEファイルが下記に表示されます。
- 音声合成で使うには、`model_assets/{model_name}`の**直下**に`*.safetensors`ファイルと`config.json`ファイルと`style_vectors.npy`ファイルが必要です。特にリポジトリの構成は確認しないので、ダウンロード後に確認し、必要ならば再配置を行ってください。
- 内容はチェックしませんので、**ダウンロードする前にURLにアクセスして中身を必ず確認**してください。怪しいURLは入力しないでください。
"""


def download_model(url: str):
    # Parse url like: https://huggingface.co/username/myrepo/tree/main/jvnv-F1-jp
    # or like: https://huggingface.co/username/myrepo

    # repo_id = "username/myrepo"
    repo_id = url.split("https://huggingface.co/")[1].split("/tree/main")[0]
    if len(repo_id.split("/")) != 2:
        logger.error(f"Invalid URL: {url}")
        return "Error: URLが不正です。"
    # repo_folder = "jvnv-F1-jp"
    repo_folder = url.split("/tree/main/")[-1] if "/tree/main/" in url else ""
    # remove last / if exists
    if repo_folder.endswith("/"):
        repo_folder = repo_folder[:-1]
    if repo_folder == "":
        model_name = repo_id.replace("/", "-")
        local_dir = assets_root / model_name
        logger.info(f"Downloading {repo_id} to {local_dir}")
        result = snapshot_download(repo_id, local_dir=local_dir)
    else:
        model_name = repo_id.replace("/", "-") + "-" + repo_folder.split("/")[-1]
        local_dir = assets_root / model_name
        logger.info(f"Downloading {repo_id}/{repo_folder} to {local_dir}")
        result = snapshot_download(
            repo_id,
            local_dir=local_dir,
            allow_patterns=[repo_folder + "/*"],
        )
        # Move the downloaded folder to the correct path
        shutil.copytree(
            assets_root / model_name / repo_folder, local_dir, dirs_exist_ok=True
        )
        shutil.rmtree(assets_root / model_name / repo_folder.split("/")[0])
    # try to download README.md
    try:
        snapshot_download(
            repo_id,
            local_dir=local_dir,
            allow_patterns=["README.md"],
        )
        # README.mdの中身を表示
        with open(local_dir / "README.md", encoding="utf-8") as f:
            readme = f.read()
    except Exception as e:
        logger.warning(f"README.md not found: {e}")
        readme = "README.mdが見つかりませんでした。"

    # Remove local_dir/.huggingface
    hf_dir = local_dir / ".huggingface"
    if hf_dir.exists():
        shutil.rmtree(local_dir / ".huggingface")
    return f"保存完了。フォルダ:\n{result}", readme


def create_download_app() -> gr.Blocks:
    with gr.Blocks() as app:
        gr.Markdown(how_to_md)
        url = gr.Textbox(
            label="URL", placeholder="https://huggingface.co/username/myrepo"
        )
        btn = gr.Button("ダウンロード")
        info = gr.Markdown("ダウンロード結果")
        md = gr.Markdown(
            label="README.mdファイル", value="ここにREADME.mdがあれば表示されます。"
        )
        btn.click(download_model, inputs=[url], outputs=[info, md])

    return app


if __name__ == "__main__":
    app = create_download_app()
    app.launch()
