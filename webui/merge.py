import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from style_bert_vits2.constants import DEFAULT_STYLE, GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder


voice_keys = ["dec"]
voice_pitch_keys = ["flow"]
speech_style_keys = ["enc_p"]
tempo_keys = ["sdp", "dp"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get path settings
with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    # dataset_root = path_config["dataset_root"]
    assets_root = path_config["assets_root"]


def merge_style(model_name_a, model_name_b, weight, output_name, style_triple_list):
    """
    style_triple_list: list[(model_aでのスタイル名, model_bでのスタイル名, 出力するスタイル名)]
    """
    # 新スタイル名リストにNeutralが含まれているか確認し、Neutralを先頭に持ってくる
    if any(triple[2] == DEFAULT_STYLE for triple in style_triple_list):
        # 存在する場合、リストをソート
        sorted_list = sorted(style_triple_list, key=lambda x: x[2] != DEFAULT_STYLE)
    else:
        # 存在しない場合、エラーを発生
        raise ValueError(f"No element with {DEFAULT_STYLE} output style name found.")

    style_vectors_a = np.load(
        os.path.join(assets_root, model_name_a, "style_vectors.npy")
    )  # (style_num_a, 256)
    style_vectors_b = np.load(
        os.path.join(assets_root, model_name_b, "style_vectors.npy")
    )  # (style_num_b, 256)
    with open(
        os.path.join(assets_root, model_name_a, "config.json"), "r", encoding="utf-8"
    ) as f:
        config_a = json.load(f)
    with open(
        os.path.join(assets_root, model_name_b, "config.json"), "r", encoding="utf-8"
    ) as f:
        config_b = json.load(f)
    style2id_a = config_a["data"]["style2id"]
    style2id_b = config_b["data"]["style2id"]
    new_style_vecs = []
    new_style2id = {}
    for style_a, style_b, style_out in sorted_list:
        if style_a not in style2id_a:
            logger.error(f"{style_a} is not in {model_name_a}.")
            raise ValueError(f"{style_a} は {model_name_a} にありません。")
        if style_b not in style2id_b:
            logger.error(f"{style_b} is not in {model_name_b}.")
            raise ValueError(f"{style_b} は {model_name_b} にありません。")
        new_style = (
            style_vectors_a[style2id_a[style_a]] * (1 - weight)
            + style_vectors_b[style2id_b[style_b]] * weight
        )
        new_style_vecs.append(new_style)
        new_style2id[style_out] = len(new_style_vecs) - 1
    new_style_vecs = np.array(new_style_vecs)

    output_style_path = os.path.join(assets_root, output_name, "style_vectors.npy")
    np.save(output_style_path, new_style_vecs)

    new_config = config_a.copy()
    new_config["data"]["num_styles"] = len(new_style2id)
    new_config["data"]["style2id"] = new_style2id
    new_config["model_name"] = output_name
    with open(
        os.path.join(assets_root, output_name, "config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)

    # recipe.jsonを読み込んで、style_triple_listを追記
    info_path = os.path.join(assets_root, output_name, "recipe.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}
    info["style_triple_list"] = style_triple_list
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return output_style_path, list(new_style2id.keys())


def lerp_tensors(t, v0, v1):
    return v0 * (1 - t) + v1 * t


def slerp_tensors(t, v0, v1, dot_thres=0.998):
    device = v0.device
    v0c = v0.cpu().numpy()
    v1c = v1.cpu().numpy()

    dot = np.sum(v0c * v1c / (np.linalg.norm(v0c) * np.linalg.norm(v1c)))

    if abs(dot) > dot_thres:
        return lerp_tensors(t, v0, v1)

    th0 = np.arccos(dot)
    sin_th0 = np.sin(th0)
    th_t = th0 * t

    return torch.from_numpy(
        v0c * np.sin(th0 - th_t) / sin_th0 + v1c * np.sin(th_t) / sin_th0
    ).to(device)


def merge_models(
    model_path_a,
    model_path_b,
    voice_weight,
    voice_pitch_weight,
    speech_style_weight,
    tempo_weight,
    output_name,
    use_slerp_instead_of_lerp,
):
    """model Aを起点に、model Bの各要素を重み付けしてマージする。
    safetensors形式を前提とする。"""
    model_a_weight = {}
    with safe_open(model_path_a, framework="pt", device="cpu") as f:
        for k in f.keys():
            model_a_weight[k] = f.get_tensor(k)

    model_b_weight = {}
    with safe_open(model_path_b, framework="pt", device="cpu") as f:
        for k in f.keys():
            model_b_weight[k] = f.get_tensor(k)

    merged_model_weight = model_a_weight.copy()

    for key in model_a_weight.keys():
        if any([key.startswith(prefix) for prefix in voice_keys]):
            weight = voice_weight
        elif any([key.startswith(prefix) for prefix in voice_pitch_keys]):
            weight = voice_pitch_weight
        elif any([key.startswith(prefix) for prefix in speech_style_keys]):
            weight = speech_style_weight
        elif any([key.startswith(prefix) for prefix in tempo_keys]):
            weight = tempo_weight
        else:
            continue
        merged_model_weight[key] = (
            slerp_tensors if use_slerp_instead_of_lerp else lerp_tensors
        )(weight, model_a_weight[key], model_b_weight[key])

    merged_model_path = os.path.join(
        assets_root, output_name, f"{output_name}.safetensors"
    )
    os.makedirs(os.path.dirname(merged_model_path), exist_ok=True)
    save_file(merged_model_weight, merged_model_path)

    info = {
        "model_a": model_path_a,
        "model_b": model_path_b,
        "voice_weight": voice_weight,
        "voice_pitch_weight": voice_pitch_weight,
        "speech_style_weight": speech_style_weight,
        "tempo_weight": tempo_weight,
    }
    with open(
        os.path.join(assets_root, output_name, "recipe.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return merged_model_path


def merge_models_gr(
    model_name_a,
    model_path_a,
    model_name_b,
    model_path_b,
    output_name,
    voice_weight,
    voice_pitch_weight,
    speech_style_weight,
    tempo_weight,
    use_slerp_instead_of_lerp,
):
    if output_name == "":
        return "Error: 新しいモデル名を入力してください。"
    merged_model_path = merge_models(
        model_path_a,
        model_path_b,
        voice_weight,
        voice_pitch_weight,
        speech_style_weight,
        tempo_weight,
        output_name,
        use_slerp_instead_of_lerp,
    )
    return f"Success: モデルを{merged_model_path}に保存しました。"


def merge_style_gr(
    model_name_a,
    model_name_b,
    weight,
    output_name,
    style_triple_list_str: str,
):
    if output_name == "":
        return "Error: 新しいモデル名を入力してください。", None
    style_triple_list = []
    for line in style_triple_list_str.split("\n"):
        if not line:
            continue
        style_triple = line.split(",")
        if len(style_triple) != 3:
            logger.error(f"Invalid style triple: {line}")
            return (
                f"Error: スタイルを3つのカンマ区切りで入力してください:\n{line}",
                None,
            )
        style_a, style_b, style_out = style_triple
        style_a = style_a.strip()
        style_b = style_b.strip()
        style_out = style_out.strip()
        style_triple_list.append((style_a, style_b, style_out))
    try:
        new_style_path, new_styles = merge_style(
            model_name_a, model_name_b, weight, output_name, style_triple_list
        )
    except ValueError as e:
        return f"Error: {e}"
    return f"Success: スタイルを{new_style_path}に保存しました。", gr.Dropdown(
        choices=new_styles, value=new_styles[0]
    )


def simple_tts(model_name, text, style=DEFAULT_STYLE, style_weight=1.0):
    model_path = os.path.join(assets_root, model_name, f"{model_name}.safetensors")
    config_path = os.path.join(assets_root, model_name, "config.json")
    style_vec_path = os.path.join(assets_root, model_name, "style_vectors.npy")

    model = TTSModel(Path(model_path), Path(config_path), Path(style_vec_path), device)
    return model.infer(text, style=style, style_weight=style_weight)


def update_two_model_names_dropdown(model_holder: TTSModelHolder):
    new_names, new_files, _ = model_holder.update_model_names_for_gradio()
    return new_names, new_files, new_names, new_files


def load_styles_gr(model_name_a, model_name_b):
    config_path_a = os.path.join(assets_root, model_name_a, "config.json")
    with open(config_path_a, "r", encoding="utf-8") as f:
        config_a = json.load(f)
    styles_a = list(config_a["data"]["style2id"].keys())

    config_path_b = os.path.join(assets_root, model_name_b, "config.json")
    with open(config_path_b, "r", encoding="utf-8") as f:
        config_b = json.load(f)
    styles_b = list(config_b["data"]["style2id"].keys())

    return (
        gr.Textbox(value=", ".join(styles_a)),
        gr.Textbox(value=", ".join(styles_b)),
        gr.TextArea(
            label="スタイルのマージリスト",
            placeholder=f"{DEFAULT_STYLE}, {DEFAULT_STYLE},{DEFAULT_STYLE}\nAngry, Angry, Angry",
            value="\n".join(
                f"{sty_a}, {sty_b}, {sty_a if sty_a != sty_b else ''}{sty_b}"
                for sty_a in styles_a
                for sty_b in styles_b
            ),
        ),
    )


initial_md = """
## 使い方

1. マージしたい2つのモデルを選択してください（`model_assets`フォルダの中から選ばれます）。
2. マージ後のモデルの名前を入力してください。
3. マージ後のモデルの声質・話し方・話す速さを調整してください。
4. 「モデルファイルのマージ」ボタンを押してください（safetensorsファイルがマージされる）。
5. スタイルベクトルファイルも生成する必要があるので、指示に従ってマージ方法を入力後、「スタイルのマージ」ボタンを押してください。

以上でマージは完了で、`model_assets/マージ後のモデル名`にマージ後のモデルが保存され、音声合成のときに使えます。

また`model_asses/マージ後のモデル名/recipe.json`には、マージの配合レシピが記録されます（推論にはいらないので配合メモ用です）。

一番下にマージしたモデルによる簡易的な音声合成機能もつけています。

## 注意
1.x系と2.x-JP-Extraのモデルマージは失敗するようです。
"""

style_merge_md = f"""
## スタイルベクトルのマージ

1行に「モデルAのスタイル名, モデルBのスタイル名, 左の2つを混ぜて出力するスタイル名」
という形式で入力してください。例えば、
```
{DEFAULT_STYLE}, {DEFAULT_STYLE}, {DEFAULT_STYLE}
Happy, Surprise, HappySurprise
```
と入力すると、マージ後のスタイルベクトルは、
- `{DEFAULT_STYLE}`: モデルAの`{DEFAULT_STYLE}`とモデルBの`{DEFAULT_STYLE}`を混ぜたもの
- `HappySurprise`: モデルAの`Happy`とモデルBの`Surprise`を混ぜたもの
の2つになります。

### 注意
- 必ず「{DEFAULT_STYLE}」という名前のスタイルを作ってください。これは、マージ後のモデルの平均スタイルになります。
- 構造上の相性の関係で、スタイルベクトルを混ぜる重みは、上の「話し方」と同じ比率で混ぜられます。例えば「話し方」が0のときはモデルAのみしか使われません。
"""


def create_merge_app(model_holder: TTSModelHolder) -> gr.Blocks:
    model_names = model_holder.model_names
    if len(model_names) == 0:
        logger.error(
            f"モデルが見つかりませんでした。{assets_root}にモデルを置いてください。"
        )
        with gr.Blocks() as app:
            gr.Markdown(
                f"Error: モデルが見つかりませんでした。{assets_root}にモデルを置いてください。"
            )
        return app
    initial_id = 0
    initial_model_files = model_holder.model_files_dict[model_names[initial_id]]

    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(
            "2つのStyle-Bert-VITS2モデルから、声質・話し方・話す速さを取り替えたり混ぜたりできます。"
        )
        with gr.Accordion(label="使い方", open=False):
            gr.Markdown(initial_md)
        with gr.Row():
            with gr.Column(scale=3):
                model_name_a = gr.Dropdown(
                    label="モデルA",
                    choices=model_names,
                    value=model_names[initial_id],
                )
                model_path_a = gr.Dropdown(
                    label="モデルファイル",
                    choices=initial_model_files,
                    value=initial_model_files[0],
                )
            with gr.Column(scale=3):
                model_name_b = gr.Dropdown(
                    label="モデルB",
                    choices=model_names,
                    value=model_names[initial_id],
                )
                model_path_b = gr.Dropdown(
                    label="モデルファイル",
                    choices=initial_model_files,
                    value=initial_model_files[0],
                )
            refresh_button = gr.Button("更新", scale=1, visible=True)
        with gr.Column(variant="panel"):
            new_name = gr.Textbox(label="新しいモデル名", placeholder="new_model")
            with gr.Row():
                voice_slider = gr.Slider(
                    label="声質",
                    value=0,
                    minimum=0,
                    maximum=1,
                    step=0.1,
                )
                voice_pitch_slider = gr.Slider(
                    label="声の高さ",
                    value=0,
                    minimum=0,
                    maximum=1,
                    step=0.1,
                )
                speech_style_slider = gr.Slider(
                    label="話し方（抑揚・感情表現等）",
                    value=0,
                    minimum=0,
                    maximum=1,
                    step=0.1,
                )
                tempo_slider = gr.Slider(
                    label="話す速さ・リズム・テンポ",
                    value=0,
                    minimum=0,
                    maximum=1,
                    step=0.1,
                )
                use_slerp_instead_of_lerp = gr.Checkbox(
                    label="線形補完のかわりに球面線形補完を使う",
                    value=False,
                )
            with gr.Column(variant="panel"):
                gr.Markdown("## モデルファイル（safetensors）のマージ")
                model_merge_button = gr.Button(
                    "モデルファイルのマージ", variant="primary"
                )
                info_model_merge = gr.Textbox(label="情報")
            with gr.Column(variant="panel"):
                gr.Markdown(style_merge_md)
                with gr.Row():
                    load_style_button = gr.Button("スタイル一覧をロード", scale=1)
                    styles_a = gr.Textbox(label="モデルAのスタイル一覧")
                    styles_b = gr.Textbox(label="モデルBのスタイル一覧")
                style_triple_list = gr.TextArea(
                    label="スタイルのマージリスト",
                    placeholder=f"{DEFAULT_STYLE}, {DEFAULT_STYLE},{DEFAULT_STYLE}\nAngry, Angry, Angry",
                    value=f"{DEFAULT_STYLE}, {DEFAULT_STYLE}, {DEFAULT_STYLE}",
                )
                style_merge_button = gr.Button("スタイルのマージ", variant="primary")
                info_style_merge = gr.Textbox(label="情報")

        text_input = gr.TextArea(
            label="テキスト", value="これはテストです。聞こえていますか？"
        )
        style = gr.Dropdown(
            label="スタイル",
            choices=["スタイルをマージしてください"],
            value="スタイルをマージしてください",
        )
        emotion_weight = gr.Slider(
            minimum=0,
            maximum=50,
            value=1,
            step=0.1,
            label="スタイルの強さ",
        )
        tts_button = gr.Button("音声合成", variant="primary")
        audio_output = gr.Audio(label="結果")

        model_name_a.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name_a],
            outputs=[model_path_a],
        )
        model_name_b.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name_b],
            outputs=[model_path_b],
        )

        refresh_button.click(
            lambda: update_two_model_names_dropdown(model_holder),
            outputs=[model_name_a, model_path_a, model_name_b, model_path_b],
        )

        load_style_button.click(
            load_styles_gr,
            inputs=[model_name_a, model_name_b],
            outputs=[styles_a, styles_b, style_triple_list],
        )

        model_merge_button.click(
            merge_models_gr,
            inputs=[
                model_name_a,
                model_path_a,
                model_name_b,
                model_path_b,
                new_name,
                voice_slider,
                voice_pitch_slider,
                speech_style_slider,
                tempo_slider,
                use_slerp_instead_of_lerp,
            ],
            outputs=[info_model_merge],
        )

        style_merge_button.click(
            merge_style_gr,
            inputs=[
                model_name_a,
                model_name_b,
                speech_style_slider,
                new_name,
                style_triple_list,
            ],
            outputs=[info_style_merge, style],
        )

        tts_button.click(
            simple_tts,
            inputs=[new_name, text_input, style, emotion_weight],
            outputs=[audio_output],
        )

    return app
