import gradio as gr
from huggingface_hub import hf_hub_download, snapshot_download

from config import get_path_config


assets_root = get_path_config().assets_root

how_to_md = """
## 使い方

Hugging Face 🤗 に公開されているモデルをダウンロードして音声合成で使えるようにします。

例:

- `https://huggingface.co/username/my_sbv2_model`を指定すると、`model_assets/username-my_sbv2_model`に全体がダウンロードされます。
- `https://huggingface.co/username/my_sbv2_models/tree/main/model1`を指定すると、`model_assets/username-my_sbv2_models/model1`に`model1`フォルダのみがダウンロードされます。

**注意**

- 音声合成で使うには、`model_assets/{model_name}`の**直下**に`*.safetensors`ファイルと`config.json`ファイルと`style_vectors.npy`ファイルが必要です。特にリポジトリの構成は確認しないので、ダウンロード後に必要ならば再配置等を行ってください。
- リポジトリの内容はチェックしませんので、ダウンロードする前にURLにアクセスしてリポジトリの内容を確認してください。怪しいURLは入力しないでください。
"""


def download_model(url: str):
    # Parse url like: https://huggingface.co/username/myrepo/tree/main/jvnv-F1-jp
    # or like: https://huggingface.co/username/myrepo

    # repo_id = "username/myrepo"
    repo_id = url.split("https://huggingface.co/")[1].split("/tree/main")[0]
    if len(repo_id.split("/")) != 2:
        return "Error: URLが不正です。"
    # repo_folder = "jvnv-F1-jp"
    repo_folder = url.split("/tree/main/")[-1] if "/tree/main/" in url else ""
    # remove last / if exists
    if repo_folder.endswith("/"):
        repo_folder = repo_folder[:-1]
    if repo_folder == "":
        model_name = repo_id.replace("/", "-")
        result = snapshot_download(repo_id, local_dir=assets_root / model_name)
    else:
        model_name = repo_id.replace("/", "-")
        result = snapshot_download(
            repo_id,
            local_dir=assets_root / model_name,
            allow_patterns=[repo_folder + "/*"],
        )
    return f"ダウンロード完了: {result}"


def create_download_app() -> gr.Blocks:
    with gr.Blocks() as app:
        gr.Markdown(how_to_md)
        url = gr.Textbox(
            label="URL", placeholder="https://huggingface.co/username/myrepo"
        )
        btn = gr.Button("ダウンロード")
        info = gr.Textbox(label="情報", value="")
        btn.click(download_model, inputs=[url], outputs=[info])

    return app


if __name__ == "__main__":
    app = create_download_app()
    app.launch()
