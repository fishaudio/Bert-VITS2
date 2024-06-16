import json
from pathlib import Path
from typing import Any, Union

import gradio as gr
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from config import get_path_config
from style_bert_vits2.constants import DEFAULT_STYLE, GRADIO_THEME
from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder


voice_keys = ["dec"]
voice_pitch_keys = ["flow"]
speech_style_keys = ["enc_p"]
tempo_keys = ["sdp", "dp"]

device = "cuda" if torch.cuda.is_available() else "cpu"
path_config = get_path_config()
assets_root = path_config.assets_root


def load_safetensors(model_path: Union[str, Path]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result


def load_config(model_name: str) -> dict[str, Any]:
    with open(assets_root / model_name / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    return config


def save_config(config: dict[str, Any], model_name: str):
    with open(assets_root / model_name / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_recipe(model_name: str) -> dict[str, Any]:
    receipe_path = assets_root / model_name / "recipe.json"
    if receipe_path.exists():
        with open(receipe_path, encoding="utf-8") as f:
            recipe = json.load(f)
    else:
        recipe = {}
    return recipe


def save_recipe(recipe: dict[str, Any], model_name: str):
    with open(assets_root / model_name / "recipe.json", "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)


def load_style_vectors(model_name: str) -> np.ndarray:
    return np.load(assets_root / model_name / "style_vectors.npy")


def save_style_vectors(style_vectors: np.ndarray, model_name: str):
    np.save(assets_root / model_name / "style_vectors.npy", style_vectors)


def merge_style_usual(
    model_name_a: str,
    model_name_b: str,
    weight: float,
    output_name: str,
    style_tuple_list: list[tuple[str, ...]],
) -> list[str]:
    """
    new = (1 - weight) * A + weight * B
    style_triple_list: list[(model_aでのスタイル名, model_bでのスタイル名, 出力するスタイル名)]
    """
    style_vectors_a = load_style_vectors(model_name_a)
    style_vectors_b = load_style_vectors(model_name_b)
    config_a = load_config(model_name_a)
    config_b = load_config(model_name_b)
    style2id_a = config_a["data"]["style2id"]
    style2id_b = config_b["data"]["style2id"]
    new_style_vecs = []
    new_style2id = {}
    for style_a, style_b, style_out in style_tuple_list:
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
    save_style_vectors(new_style_vecs, output_name)

    new_config = config_a.copy()
    new_config["data"]["num_styles"] = len(new_style2id)
    new_config["data"]["style2id"] = new_style2id
    new_config["model_name"] = output_name
    save_config(new_config, output_name)

    receipe = load_recipe(output_name)
    receipe["style_tuple_list"] = style_tuple_list
    save_recipe(receipe, output_name)

    return list(new_style2id.keys())


def merge_style_add_diff(
    model_name_a: str,
    model_name_b: str,
    model_name_c: str,
    weight: float,
    output_name: str,
    style_tuple_list: list[tuple[str, ...]],
) -> list[str]:
    """
    new = A + weight * (B - C)
    style_tuple_list: list[(model_aでのスタイル名, model_bでのスタイル名, model_cでのスタイル名, 出力するスタイル名)]
    """
    style_vectors_a = load_style_vectors(model_name_a)
    style_vectors_b = load_style_vectors(model_name_b)
    style_vectors_c = load_style_vectors(model_name_c)
    config_a = load_config(model_name_a)
    config_b = load_config(model_name_b)
    config_c = load_config(model_name_c)
    style2id_a = config_a["data"]["style2id"]
    style2id_b = config_b["data"]["style2id"]
    style2id_c = config_c["data"]["style2id"]
    new_style_vecs = []
    new_style2id = {}
    for style_a, style_b, style_c, style_out in style_tuple_list:
        if style_a not in style2id_a:
            logger.error(f"{style_a} is not in {model_name_a}.")
            raise ValueError(f"{style_a} は {model_name_a} にありません。")
        if style_b not in style2id_b:
            logger.error(f"{style_b} is not in {model_name_b}.")
            raise ValueError(f"{style_b} は {model_name_b} にありません。")
        if style_c not in style2id_c:
            logger.error(f"{style_c} is not in {model_name_c}.")
            raise ValueError(f"{style_c} は {model_name_c} にありません。")
        new_style = style_vectors_a[style2id_a[style_a]] + weight * (
            style_vectors_b[style2id_b[style_b]] - style_vectors_c[style2id_c[style_c]]
        )
        new_style_vecs.append(new_style)
        new_style2id[style_out] = len(new_style_vecs) - 1
    new_style_vecs = np.array(new_style_vecs)

    save_style_vectors(new_style_vecs, output_name)

    new_config = config_a.copy()
    new_config["data"]["num_styles"] = len(new_style2id)
    new_config["data"]["style2id"] = new_style2id
    new_config["model_name"] = output_name
    save_config(new_config, output_name)

    receipe = load_recipe(output_name)
    receipe["style_tuple_list"] = style_tuple_list
    save_recipe(receipe, output_name)

    return list(new_style2id.keys())


def merge_style_weighted_sum(
    model_name_a: str,
    model_name_b: str,
    model_name_c: str,
    model_a_coeff: float,
    model_b_coeff: float,
    model_c_coeff: float,
    output_name: str,
    style_tuple_list: list[tuple[str, ...]],
) -> list[str]:
    """
    new = A * model_a_coeff + B * model_b_coeff + C * model_c_coeff
    style_tuple_list: list[(model_aでのスタイル名, model_bでのスタイル名, model_cでのスタイル名, 出力するスタイル名)]
    """
    style_vectors_a = load_style_vectors(model_name_a)
    style_vectors_b = load_style_vectors(model_name_b)
    style_vectors_c = load_style_vectors(model_name_c)
    config_a = load_config(model_name_a)
    config_b = load_config(model_name_b)
    config_c = load_config(model_name_c)
    style2id_a = config_a["data"]["style2id"]
    style2id_b = config_b["data"]["style2id"]
    style2id_c = config_c["data"]["style2id"]
    new_style_vecs = []
    new_style2id = {}
    for style_a, style_b, style_c, style_out in style_tuple_list:
        if style_a not in style2id_a:
            logger.error(f"{style_a} is not in {model_name_a}.")
            raise ValueError(f"{style_a} は {model_name_a} にありません。")
        if style_b not in style2id_b:
            logger.error(f"{style_b} is not in {model_name_b}.")
            raise ValueError(f"{style_b} は {model_name_b} にありません。")
        if style_c not in style2id_c:
            logger.error(f"{style_c} is not in {model_name_c}.")
            raise ValueError(f"{style_c} は {model_name_c} にありません。")
        new_style = (
            style_vectors_a[style2id_a[style_a]] * model_a_coeff
            + style_vectors_b[style2id_b[style_b]] * model_b_coeff
            + style_vectors_c[style2id_c[style_c]] * model_c_coeff
        )
        new_style_vecs.append(new_style)
        new_style2id[style_out] = len(new_style_vecs) - 1
    new_style_vecs = np.array(new_style_vecs)

    save_style_vectors(new_style_vecs, output_name)

    new_config = config_a.copy()
    new_config["data"]["num_styles"] = len(new_style2id)
    new_config["data"]["style2id"] = new_style2id
    new_config["model_name"] = output_name
    save_config(new_config, output_name)

    receipe = load_recipe(output_name)
    receipe["style_tuple_list"] = style_tuple_list
    save_recipe(receipe, output_name)

    return list(new_style2id.keys())


def merge_style_add_null(
    model_name_a: str,
    model_name_b: str,
    weight: float,
    output_name: str,
    style_tuple_list: list[tuple[str, ...]],
) -> list[str]:
    """
    new = A + weight * B
    style_tuple_list: list[(model_aでのスタイル名, model_bでのスタイル名, 出力するスタイル名)]
    """
    style_vectors_a = load_style_vectors(model_name_a)
    style_vectors_b = load_style_vectors(model_name_b)
    config_a = load_config(model_name_a)
    config_b = load_config(model_name_b)
    style2id_a = config_a["data"]["style2id"]
    style2id_b = config_b["data"]["style2id"]
    new_style_vecs = []
    new_style2id = {}
    for style_a, style_b, style_out in style_tuple_list:
        if style_a not in style2id_a:
            logger.error(f"{style_a} is not in {model_name_a}.")
            raise ValueError(f"{style_a} は {model_name_a} にありません。")
        if style_b not in style2id_b:
            logger.error(f"{style_b} is not in {model_name_b}.")
            raise ValueError(f"{style_b} は {model_name_b} にありません。")
        new_style = (
            style_vectors_a[style2id_a[style_a]]
            + weight * style_vectors_b[style2id_b[style_b]]
        )
        new_style_vecs.append(new_style)
        new_style2id[style_out] = len(new_style_vecs) - 1
    new_style_vecs = np.array(new_style_vecs)

    save_style_vectors(new_style_vecs, output_name)

    new_config = config_a.copy()
    new_config["data"]["num_styles"] = len(new_style2id)
    new_config["data"]["style2id"] = new_style2id
    new_config["model_name"] = output_name
    save_config(new_config, output_name)

    receipe = load_recipe(output_name)
    receipe["style_tuple_list"] = style_tuple_list
    save_recipe(receipe, output_name)

    return list(new_style2id.keys())


def lerp_tensors(t: float, v0: torch.Tensor, v1: torch.Tensor):
    return v0 * (1 - t) + v1 * t


def slerp_tensors(
    t: float, v0: torch.Tensor, v1: torch.Tensor, dot_thres: float = 0.998
):
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


def merge_models_usual(
    model_path_a: str,
    model_path_b: str,
    voice_weight: float,
    voice_pitch_weight: float,
    speech_style_weight: float,
    tempo_weight: float,
    output_name: str,
    use_slerp_instead_of_lerp: bool,
):
    """
    new = (1 - weight) * A + weight * B
    """
    model_a_weight = load_safetensors(model_path_a)
    model_b_weight = load_safetensors(model_path_b)

    merged_model_weight = model_a_weight.copy()

    for key in model_a_weight:
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

    merged_model_path = assets_root / output_name / f"{output_name}.safetensors"
    merged_model_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged_model_weight, merged_model_path)

    receipe = {
        "method": "usual",
        "model_a": model_path_a,
        "model_b": model_path_b,
        "voice_weight": voice_weight,
        "voice_pitch_weight": voice_pitch_weight,
        "speech_style_weight": speech_style_weight,
        "tempo_weight": tempo_weight,
        "use_slerp_instead_of_lerp": use_slerp_instead_of_lerp,
    }
    save_recipe(receipe, output_name)

    # Merge default Neutral style vectors and save
    model_name_a = Path(model_path_a).parent.name
    model_name_b = Path(model_path_b).parent.name
    style_vectors_a = load_style_vectors(model_name_a)
    style_vectors_b = load_style_vectors(model_name_b)

    new_config = load_config(model_name_a)
    new_config["model_name"] = output_name
    new_config["data"]["num_styles"] = 1
    new_config["data"]["style2id"] = {DEFAULT_STYLE: 0}
    save_config(new_config, output_name)

    neutral_vector_a = style_vectors_a[0]
    neutral_vector_b = style_vectors_b[0]
    weight = speech_style_weight
    new_neutral_vector = (1 - weight) * neutral_vector_a + weight * neutral_vector_b
    new_style_vectors = np.array([new_neutral_vector])
    save_style_vectors(new_style_vectors, output_name)
    return merged_model_path


def merge_models_add_diff(
    model_path_a: str,
    model_path_b: str,
    model_path_c: str,
    voice_weight: float,
    voice_pitch_weight: float,
    speech_style_weight: float,
    tempo_weight: float,
    output_name: str,
):
    """
    new = A + weight * (B - C)
    """
    model_a_weight = load_safetensors(model_path_a)
    model_b_weight = load_safetensors(model_path_b)
    model_c_weight = load_safetensors(model_path_c)

    merged_model_weight = model_a_weight.copy()

    for key in model_a_weight:
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
        merged_model_weight[key] = model_a_weight[key] + weight * (
            model_b_weight[key] - model_c_weight[key]
        )

    merged_model_path = assets_root / output_name / f"{output_name}.safetensors"
    merged_model_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged_model_weight, merged_model_path)

    info = {
        "method": "add_diff",
        "model_a": model_path_a,
        "model_b": model_path_b,
        "model_c": model_path_c,
        "voice_weight": voice_weight,
        "voice_pitch_weight": voice_pitch_weight,
        "speech_style_weight": speech_style_weight,
        "tempo_weight": tempo_weight,
    }
    with open(assets_root / output_name / "recipe.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Default style merge only using Neutral style
    model_name_a = Path(model_path_a).parent.name
    model_name_b = Path(model_path_b).parent.name
    model_name_c = Path(model_path_c).parent.name

    style_vectors_a = np.load(
        assets_root / model_name_a / "style_vectors.npy"
    )  # (style_num_a, 256)
    style_vectors_b = np.load(
        assets_root / model_name_b / "style_vectors.npy"
    )  # (style_num_b, 256)
    style_vectors_c = np.load(
        assets_root / model_name_c / "style_vectors.npy"
    )  # (style_num_c, 256)
    with open(assets_root / model_name_a / "config.json", encoding="utf-8") as f:
        new_config = json.load(f)

    new_config["model_name"] = output_name
    new_config["data"]["num_styles"] = 1
    new_config["data"]["style2id"] = {DEFAULT_STYLE: 0}
    with open(assets_root / output_name / "config.json", "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)

    neutral_vector_a = style_vectors_a[0]
    neutral_vector_b = style_vectors_b[0]
    neutral_vector_c = style_vectors_c[0]
    weight = speech_style_weight
    new_neutral_vector = neutral_vector_a + weight * (
        neutral_vector_b - neutral_vector_c
    )
    new_style_vectors = np.array([new_neutral_vector])
    new_style_path = assets_root / output_name / "style_vectors.npy"
    np.save(new_style_path, new_style_vectors)
    return merged_model_path


def merge_models_weighted_sum(
    model_path_a: str,
    model_path_b: str,
    model_path_c: str,
    model_a_coeff: float,
    model_b_coeff: float,
    model_c_coeff: float,
    output_name: str,
):
    model_a_weight = load_safetensors(model_path_a)
    model_b_weight = load_safetensors(model_path_b)
    model_c_weight = load_safetensors(model_path_c)

    merged_model_weight = model_a_weight.copy()

    for key in model_a_weight:
        merged_model_weight[key] = (
            model_a_coeff * model_a_weight[key]
            + model_b_coeff * model_b_weight[key]
            + model_c_coeff * model_c_weight[key]
        )

    merged_model_path = assets_root / output_name / f"{output_name}.safetensors"
    merged_model_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged_model_weight, merged_model_path)

    info = {
        "method": "weighted_sum",
        "model_a": model_path_a,
        "model_b": model_path_b,
        "model_c": model_path_c,
        "model_a_coeff": model_a_coeff,
        "model_b_coeff": model_b_coeff,
        "model_c_coeff": model_c_coeff,
    }
    with open(assets_root / output_name / "recipe.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Default style merge only using Neutral style
    model_name_a = Path(model_path_a).parent.name
    model_name_b = Path(model_path_b).parent.name
    model_name_c = Path(model_path_c).parent.name

    style_vectors_a = np.load(
        assets_root / model_name_a / "style_vectors.npy"
    )  # (style_num_a, 256)
    style_vectors_b = np.load(
        assets_root / model_name_b / "style_vectors.npy"
    )  # (style_num_b, 256)
    style_vectors_c = np.load(
        assets_root / model_name_c / "style_vectors.npy"
    )  # (style_num_c, 256)

    with open(assets_root / model_name_a / "config.json", encoding="utf-8") as f:
        new_config = json.load(f)

    new_config["model_name"] = output_name
    new_config["data"]["num_styles"] = 1
    new_config["data"]["style2id"] = {DEFAULT_STYLE: 0}
    with open(assets_root / output_name / "config.json", "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)

    neutral_vector_a = style_vectors_a[0]
    neutral_vector_b = style_vectors_b[0]
    neutral_vector_c = style_vectors_c[0]
    new_neutral_vector = (
        model_a_coeff * neutral_vector_a
        + model_b_coeff * neutral_vector_b
        + model_c_coeff * neutral_vector_c
    )
    new_style_vectors = np.array([new_neutral_vector])
    new_style_path = assets_root / output_name / "style_vectors.npy"
    np.save(new_style_path, new_style_vectors)
    return merged_model_path


def merge_models_add_null(
    model_path_a: str,
    model_path_b: str,
    voice_weight: float,
    voice_pitch_weight: float,
    speech_style_weight: float,
    tempo_weight: float,
    output_name: str,
):
    model_a_weight = load_safetensors(model_path_a)
    model_b_weight = load_safetensors(model_path_b)

    merged_model_weight = model_a_weight.copy()

    for key in model_a_weight:
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
        merged_model_weight[key] = model_a_weight[key] + weight * model_b_weight[key]

    merged_model_path = assets_root / output_name / f"{output_name}.safetensors"
    merged_model_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged_model_weight, merged_model_path)

    info = {
        "method": "add_null",
        "model_a": model_path_a,
        "model_b": model_path_b,
        "voice_weight": voice_weight,
        "voice_pitch_weight": voice_pitch_weight,
        "speech_style_weight": speech_style_weight,
        "tempo_weight": tempo_weight,
    }
    with open(assets_root / output_name / "recipe.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Default style merge only using Neutral style
    model_name_a = Path(model_path_a).parent.name
    model_name_b = Path(model_path_b).parent.name

    style_vectors_a = np.load(
        assets_root / model_name_a / "style_vectors.npy"
    )  # (style_num_a, 256)
    style_vectors_b = np.load(
        assets_root / model_name_b / "style_vectors.npy"
    )  # (style_num_b, 256)
    with open(assets_root / model_name_a / "config.json", encoding="utf-8") as f:
        new_config = json.load(f)

    new_config["model_name"] = output_name
    new_config["data"]["num_styles"] = 1
    new_config["data"]["style2id"] = {DEFAULT_STYLE: 0}
    with open(assets_root / output_name / "config.json", "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)

    neutral_vector_a = style_vectors_a[0]
    neutral_vector_b = style_vectors_b[0]
    weight = speech_style_weight
    new_neutral_vector = neutral_vector_a + weight * neutral_vector_b
    new_style_vectors = np.array([new_neutral_vector])
    new_style_path = assets_root / output_name / "style_vectors.npy"
    np.save(new_style_path, new_style_vectors)
    return merged_model_path


def merge_models_gr(
    model_path_a: str,
    model_path_b: str,
    model_path_c: str,
    model_a_coeff: float,
    model_b_coeff: float,
    model_c_coeff: float,
    method: str,
    output_name: str,
    voice_weight: float,
    voice_pitch_weight: float,
    speech_style_weight: float,
    tempo_weight: float,
    use_slerp_instead_of_lerp: bool,
):
    if output_name == "":
        return "Error: 新しいモデル名を入力してください。"
    assert method in [
        "usual",
        "add_diff",
        "weighted_sum",
        "add_null",
    ], f"Invalid method: {method}"
    model_a_name = Path(model_path_a).parent.name
    model_b_name = Path(model_path_b).parent.name
    model_c_name = Path(model_path_c).parent.name
    if method == "usual":
        if output_name in [model_a_name, model_b_name]:
            return "Error: マージ元のモデル名と同じ名前は使用できません。", None
        merged_model_path = merge_models_usual(
            model_path_a,
            model_path_b,
            voice_weight,
            voice_pitch_weight,
            speech_style_weight,
            tempo_weight,
            output_name,
            use_slerp_instead_of_lerp,
        )
    elif method == "add_diff":
        if output_name in [model_a_name, model_b_name, model_c_name]:
            return "Error: マージ元のモデル名と同じ名前は使用できません。", None
        merged_model_path = merge_models_add_diff(
            model_path_a,
            model_path_b,
            model_path_c,
            voice_weight,
            voice_pitch_weight,
            speech_style_weight,
            tempo_weight,
            output_name,
        )
    elif method == "weighted_sum":
        if output_name in [model_a_name, model_b_name, model_c_name]:
            return "Error: マージ元のモデル名と同じ名前は使用できません。", None
        merged_model_path = merge_models_weighted_sum(
            model_path_a,
            model_path_b,
            model_path_c,
            model_a_coeff,
            model_b_coeff,
            model_c_coeff,
            output_name,
        )
    else:  # add_null
        if output_name in [model_a_name, model_b_name]:
            return "Error: マージ元のモデル名と同じ名前は使用できません。", None
        merged_model_path = merge_models_add_null(
            model_path_a,
            model_path_b,
            voice_weight,
            voice_pitch_weight,
            speech_style_weight,
            tempo_weight,
            output_name,
        )
    return f"Success: モデルを{merged_model_path}に保存しました。", gr.Dropdown(
        choices=[DEFAULT_STYLE], value=DEFAULT_STYLE
    )


def merge_style_usual_gr(
    model_name_a: str,
    model_name_b: str,
    weight: float,
    output_name: str,
    style_tuple_list: list[tuple[str, ...]],
):
    if output_name == "":
        return "Error: 新しいモデル名を入力してください。", None
    new_styles = merge_style_usual(
        model_name_a,
        model_name_b,
        weight,
        output_name,
        style_tuple_list,
    )
    return f"Success: {output_name}のスタイルを保存しました。", gr.Dropdown(
        choices=new_styles, value=new_styles[0]
    )


def merge_style_add_diff_gr(
    model_name_a: str,
    model_name_b: str,
    model_name_c: str,
    weight: float,
    output_name: str,
    style_tuple_list: list[tuple[str, ...]],
):
    if output_name == "":
        return "Error: 新しいモデル名を入力してください。", None
    new_styles = merge_style_add_diff(
        model_name_a,
        model_name_b,
        model_name_c,
        weight,
        output_name,
        style_tuple_list,
    )
    return f"Success: {output_name}のスタイルを保存しました。", gr.Dropdown(
        choices=new_styles, value=new_styles[0]
    )


def merge_style_weighted_sum_gr(
    model_name_a: str,
    model_name_b: str,
    model_name_c: str,
    model_a_coeff: float,
    model_b_coeff: float,
    model_c_coeff: float,
    output_name: str,
    style_tuple_list: list[tuple[str, ...]],
):
    if output_name == "":
        return "Error: 新しいモデル名を入力してください。", None
    new_styles = merge_style_weighted_sum(
        model_name_a,
        model_name_b,
        model_name_c,
        model_a_coeff,
        model_b_coeff,
        model_c_coeff,
        output_name,
        style_tuple_list,
    )
    return f"Success: {output_name}のスタイルを保存しました。", gr.Dropdown(
        choices=new_styles, value=new_styles[0]
    )


def merge_style_add_null_gr(
    model_name_a: str,
    model_name_b: str,
    weight: float,
    output_name: str,
    style_tuple_list: list[tuple[str, ...]],
):
    if output_name == "":
        return "Error: 新しいモデル名を入力してください。", None
    new_styles = merge_style_add_null(
        model_name_a,
        model_name_b,
        weight,
        output_name,
        style_tuple_list,
    )
    return f"Success: {output_name}のスタイルを保存しました。", gr.Dropdown(
        choices=new_styles, value=new_styles[0]
    )


def simple_tts(
    model_name: str, text: str, style: str = DEFAULT_STYLE, style_weight: float = 1.0
):
    if model_name == "":
        return "Error: モデル名を入力してください。", None
    model_path = assets_root / model_name / f"{model_name}.safetensors"
    config_path = assets_root / model_name / "config.json"
    style_vec_path = assets_root / model_name / "style_vectors.npy"

    model = TTSModel(model_path, config_path, style_vec_path, device)

    return (
        "Success: 音声を生成しました。",
        model.infer(text, style=style, style_weight=style_weight),
    )


def update_three_model_names_dropdown(model_holder: TTSModelHolder):
    new_names, new_files, _ = model_holder.update_model_names_for_gradio()
    return new_names, new_files, new_names, new_files, new_names, new_files


def get_styles(model_name: str):
    config_path = assets_root / model_name / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    styles = list(config["data"]["style2id"].keys())
    return styles


def get_triple_styles(model_name_a: str, model_name_b: str, model_name_c: str):
    return get_styles(model_name_a), get_styles(model_name_b), get_styles(model_name_c)


def load_styles_gr(model_name_a: str, model_name_b: str):
    config_path_a = assets_root / model_name_a / "config.json"
    with open(config_path_a, encoding="utf-8") as f:
        config_a = json.load(f)
    styles_a = list(config_a["data"]["style2id"].keys())

    config_path_b = assets_root / model_name_b / "config.json"
    with open(config_path_b, encoding="utf-8") as f:
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

### マージ方法の選択

マージの方法には4つの方法があります。
- 通常のマージ `new = (1 - weight) * A + weight * B`: AとBのモデルを指定して、要素ごとに比率を指定して混ぜる
    - 単純にAとBの二人の話し方や声音を混ぜたいとき
- 差分マージ `new = A + weight * (B - C)`: AとBとCのモデルを指定して、「Bの要素からCの要素を引いたもの」をAに足す
    - 例えば、Bが「Cと同じ人だけど囁いているモデル」とすると、`B - C`は「囁きを表すベクトル」だと思えるので、それをAに足すことで、Aの声のままで囁き声を出すモデルができたりする
    - 他にも活用例はいろいろありそう
- 重み付き和 `new = a * A + b * B + c * C`: AとBとCのモデルを指定して、各モデルの係数を指定して混ぜる
    - 例えば`new = A - B` としておくと、結果としてできたモデルを別のモデルと「ヌルモデルの加算」で使うことで、差分マージが実現できる
    - 他にも何らかの活用法があるかもしれない
- ヌルモデルの加算 `new = A + weight * B`: AとBのモデルを指定して、Bのモデルに要素ごとに比率をかけたものをAに足す
    - Bのモデルは重み付き和などで `C - D` などとして作っている場合を想定している
    - 他にも何らかの活用法があるかもしれない


### マージの手順

1. マージ元のモデルたちを選択（`model_assets`フォルダの中から選ばれます）
2. マージ後のモデルの名前を入力
3. 指示に従って重みや係数を入力
4. 「モデルファイルのマージ」ボタンを押す (safetensorsファイルがマージされる)
5. 結果を簡易音声合成で確認
6. 必要に応じてスタイルベクトルのマージを行う

以上でマージは完了で、`model_assets/マージ後のモデル名`にマージ後のモデルが保存され、音声合成のときに使えます。

また`model_asses/マージ後のモデル名/recipe.json`には、マージの配合レシピが記録されます（推論にはいらないので配合メモ用です）。

一番下にマージしたモデルによる簡易的な音声合成機能もつけています。

## 注意

- 1.x系と2.x-JP-Extraのモデルマージは失敗するようです。
- 話者数が違うモデル同士はおそらくマージできません。
"""

style_merge_md = f"""
## 3. スタイルベクトルのマージ

1. マージ後のモデルにいくつスタイルを追加したいかを「作りたいスタイル数」で指定
2. マージ前のモデルのスタイルを「各モデルのスタイルを取得」ボタンで取得
3. どのスタイルたちから新しいスタイルを作るかを下の欄で入力
4. 「スタイルのマージ」をクリック

### スタイルベクトルの混ぜられ方

- 構造上の相性の関係で、スタイルベクトルを混ぜる重みは、加重和以外の場合は、上の「話し方」と同じ比率で混ぜられます。例えば「話し方」が0のときはモデルAのみしか使われません。
- 加重和の場合は、AとBとCの係数によって混ぜられます。
"""

usual_md = """
`weight` を下の各スライダーで定める数値とすると、各要素ごとに、
```
new_model = (1 - weight) * A + weight * B
```
としてマージされます。

つまり、`weight = 0` のときはモデルA、`weight = 1` のときはモデルBになります。
"""

add_diff_md = """
`weight` を下の各スライダーで定める数値とすると、各要素ごとに、
```
new_model = A + weight * (B - C)
```
としてマージされます。

通常のマージと違い、**重みを1にしてもAの要素はそのまま保たれます**。
"""

weighted_sum_md = """
モデルの係数をそれぞれ `a`, `b`, `c` とすると、 **全要素に対して**、
```
new_model = a * A + b * B + c * C
```
としてマージされます。

## TIPS

- A, B, C が全て通常モデルで、通常モデルを作りたい場合は、`a + b + c = 1`となるようにするのがよいと思います。
- `a + b + c = 0` とすると（たとえば `A - B`）、話者性を持たないヌルモデルを作ることができ、「ヌルモデルとの和」で結果を使うことが出来ます（差分マージの材料などに）
- 他にも、`a = 0.5, b = c = 0`などでモデルAを謎に小さくしたり大きくしたり負にしたりできるので、実験に使ってください。
"""

add_null_md = """
「ヌルモデル」を、いくつかのモデルの加重和であってその係数の和が0であるようなものとします（例えば `C - D` など）。

そうして作ったヌルモデルBと通常モデルAに対して、`weight` を下の各スライダーで定める数値とすると、各要素ごとに、
```
new_model = A + weight * B
```
としてマージされます。

通常のマージと違い、**重みを1にしてもAの要素はそのまま保たれます**。

実際にはヌルモデルでないBに対しても使えますが、その場合はおそらく音声が正常に生成されないモデルができる気がします。が、もしかしたら何かに使えるかもしれません。

囁きについて実験的に作ったヌルモデルを[こちら](https://huggingface.co/litagin/sbv2_null_models)に置いています。これを `B` に使うことで、任意のモデルを囁きモデルにある程度は変換できます。
"""

tts_md = f"""
## 2. 結果のテスト

マージ後のモデルで音声合成を行います。ただし、デフォルトではスタイルは`{DEFAULT_STYLE}`しか使えないので、他のスタイルを使いたい場合は、下の「スタイルベクトルのマージ」を行ってください。
"""


def method_change(x: str):
    assert x in [
        "usual",
        "add_diff",
        "weighted_sum",
        "add_null",
    ], f"Invalid method: {x}"
    # model_desc, c_col, model_a_coeff, model_b_coeff, model_c_coeff, weight_row, use_slerp_instead_of_lerp
    if x == "usual":
        return (
            gr.Markdown(usual_md),
            gr.Column(visible=False),
            gr.Number(visible=False),
            gr.Number(visible=False),
            gr.Number(visible=False),
            gr.Row(visible=True),
            gr.Checkbox(visible=True),
        )
    elif x == "add_diff":
        return (
            gr.Markdown(add_diff_md),
            gr.Column(visible=True),
            gr.Number(visible=False),
            gr.Number(visible=False),
            gr.Number(visible=False),
            gr.Row(visible=True),
            gr.Checkbox(visible=False),
        )
    elif x == "add_null":
        return (
            gr.Markdown(add_null_md),
            gr.Column(visible=False),
            gr.Number(visible=False),
            gr.Number(visible=False),
            gr.Number(visible=False),
            gr.Row(visible=True),
            gr.Checkbox(visible=False),
        )
    else:  # weighted_sum
        return (
            gr.Markdown(weighted_sum_md),
            gr.Column(visible=True),
            gr.Number(visible=True),
            gr.Number(visible=True),
            gr.Number(visible=True),
            gr.Row(visible=False),
            gr.Checkbox(visible=False),
        )


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
    initial_model_files = [
        str(f) for f in model_holder.model_files_dict[model_names[initial_id]]
    ]

    with gr.Blocks(theme=GRADIO_THEME) as app:
        gr.Markdown(
            "複数のStyle-Bert-VITS2モデルから、声質・話し方・話す速さを取り替えたり混ぜたり引いたりして新しいモデルを作成できます。"
        )
        with gr.Accordion(label="使い方", open=False):
            gr.Markdown(initial_md)
        method = gr.Radio(
            label="マージ方法",
            choices=[
                ("通常マージ", "usual"),
                ("差分マージ", "add_diff"),
                ("加重和", "weighted_sum"),
                ("ヌルモデルマージ", "add_null"),
            ],
            value="usual",
        )
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
                model_a_coeff = gr.Number(
                    label="モデルAの係数",
                    value=1.0,
                    step=0.1,
                    visible=False,
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
                model_b_coeff = gr.Number(
                    label="モデルBの係数",
                    value=-1.0,
                    step=0.1,
                    visible=False,
                )
            with gr.Column(scale=3, visible=False) as c_col:
                model_name_c = gr.Dropdown(
                    label="モデルC",
                    choices=model_names,
                    value=model_names[initial_id],
                )
                model_path_c = gr.Dropdown(
                    label="モデルファイル",
                    choices=initial_model_files,
                    value=initial_model_files[0],
                )
                model_c_coeff = gr.Number(
                    label="モデルCの係数",
                    value=0.0,
                    step=0.1,
                    visible=False,
                )
            refresh_button = gr.Button("更新", scale=1, visible=True)
        method_desc = gr.Markdown(usual_md)
        with gr.Column(variant="panel"):
            new_name = gr.Textbox(label="新しいモデル名", placeholder="new_model")
            with gr.Row() as weight_row:
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
                    visible=True,
                )
        with gr.Column(variant="panel"):
            gr.Markdown("## 1. モデルファイル (safetensors) のマージ")
            with gr.Row():
                model_merge_button = gr.Button(
                    "モデルファイルのマージ", variant="primary"
                )
                info_model_merge = gr.Textbox(label="情報")
        with gr.Column(variant="panel"):
            gr.Markdown(tts_md)
            text_input = gr.TextArea(
                label="テキスト", value="これはテストです。聞こえていますか？"
            )
            with gr.Row():
                with gr.Column():
                    style = gr.Dropdown(
                        label="スタイル",
                        choices=[DEFAULT_STYLE],
                        value=DEFAULT_STYLE,
                    )
                    emotion_weight = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=1,
                        step=0.1,
                        label="スタイルの強さ",
                    )
                tts_button = gr.Button("音声合成", variant="primary")
                tts_info = gr.Textbox(label="情報")
            audio_output = gr.Audio(label="結果")
        with gr.Column(variant="panel"):
            gr.Markdown(style_merge_md)
            style_a_list = gr.State([DEFAULT_STYLE])
            style_b_list = gr.State([DEFAULT_STYLE])
            style_c_list = gr.State([DEFAULT_STYLE])
            gr.Markdown("Hello world!")
            with gr.Row():
                style_count = gr.Number(label="作るスタイルの数", value=1, step=1)

                get_style_btn = gr.Button("各モデルのスタイルを取得", variant="primary")
            get_style_btn.click(
                get_triple_styles,
                inputs=[model_name_a, model_name_b, model_name_c],
                outputs=[style_a_list, style_b_list, style_c_list],
            )

            def join_names(*args):
                if all(arg == DEFAULT_STYLE for arg in args):
                    return DEFAULT_STYLE
                return "_".join(args)

            @gr.render(
                inputs=[
                    style_count,
                    style_a_list,
                    style_b_list,
                    style_c_list,
                    method,
                ]
            )
            def render_style(
                style_count, style_a_list, style_b_list, style_c_list, method
            ):
                a_components = []
                b_components = []
                c_components = []
                out_components = []
                if method in ["usual", "add_null"]:
                    for i in range(style_count):
                        with gr.Row():
                            style_a = gr.Dropdown(
                                label="モデルAのスタイル名",
                                key=f"style_a_{i}",
                                choices=style_a_list,
                                value=DEFAULT_STYLE,
                                interactive=i != 0,
                            )
                            style_b = gr.Dropdown(
                                label="モデルBのスタイル名",
                                key=f"style_b_{i}",
                                choices=style_b_list,
                                value=DEFAULT_STYLE,
                                interactive=i != 0,
                            )
                            style_out = gr.Textbox(
                                label="出力スタイル名",
                                key=f"style_out_{i}",
                                value=DEFAULT_STYLE,
                                interactive=i != 0,
                            )
                            style_a.change(
                                join_names,
                                inputs=[style_a, style_b],
                                outputs=[style_out],
                            )
                            style_b.change(
                                join_names,
                                inputs=[style_a, style_b],
                                outputs=[style_out],
                            )
                        a_components.append(style_a)
                        b_components.append(style_b)
                        out_components.append(style_out)
                    if method == "usual":

                        def _merge_usual(data):
                            style_tuple_list = [
                                (data[a], data[b], data[out])
                                for a, b, out in zip(
                                    a_components, b_components, out_components
                                )
                            ]
                            return merge_style_usual_gr(
                                data[model_name_a],
                                data[model_name_b],
                                data[speech_style_slider],
                                data[new_name],
                                style_tuple_list,
                            )

                        style_merge_btn.click(
                            _merge_usual,
                            inputs=set(
                                a_components
                                + b_components
                                + out_components
                                + [
                                    model_name_a,
                                    model_name_b,
                                    speech_style_slider,
                                    new_name,
                                ]
                            ),
                            outputs=[info_style_merge, style],
                        )
                    else:  # add_null

                        def _merge_add_null(data):
                            print("Method is add_null")
                            style_tuple_list = [
                                (data[a], data[b], data[out])
                                for a, b, out in zip(
                                    a_components, b_components, out_components
                                )
                            ]
                            return merge_style_add_null_gr(
                                data[model_name_a],
                                data[model_name_b],
                                data[speech_style_slider],
                                data[new_name],
                                style_tuple_list,
                            )

                        style_merge_btn.click(
                            _merge_add_null,
                            inputs=set(
                                a_components
                                + b_components
                                + out_components
                                + [
                                    model_name_a,
                                    model_name_b,
                                    speech_style_slider,
                                    new_name,
                                ]
                            ),
                            outputs=[info_style_merge, style],
                        )

                elif method in ["add_diff", "weighted_sum"]:
                    for i in range(style_count):
                        with gr.Row():
                            style_a = gr.Dropdown(
                                label="モデルAのスタイル名",
                                key=f"style_a_{i}",
                                choices=style_a_list,
                                value=DEFAULT_STYLE,
                                interactive=i != 0,
                            )
                            style_b = gr.Dropdown(
                                label="モデルBのスタイル名",
                                key=f"style_b_{i}",
                                choices=style_b_list,
                                value=DEFAULT_STYLE,
                                interactive=i != 0,
                            )
                            style_c = gr.Dropdown(
                                label="モデルCのスタイル名",
                                key=f"style_c_{i}",
                                choices=style_c_list,
                                value=DEFAULT_STYLE,
                                interactive=i != 0,
                            )
                            style_out = gr.Textbox(
                                label="出力スタイル名",
                                key=f"style_out_{i}",
                                value=DEFAULT_STYLE,
                                interactive=i != 0,
                            )
                            style_a.change(
                                join_names,
                                inputs=[style_a, style_b, style_c],
                                outputs=[style_out],
                            )
                            style_b.change(
                                join_names,
                                inputs=[style_a, style_b, style_c],
                                outputs=[style_out],
                            )
                            style_c.change(
                                join_names,
                                inputs=[style_a, style_b, style_c],
                                outputs=[style_out],
                            )

                        a_components.append(style_a)
                        b_components.append(style_b)
                        c_components.append(style_c)
                        out_components.append(style_out)
                    if method == "add_diff":

                        def _merge_add_diff(data):
                            style_tuple_list = [
                                (data[a], data[b], data[c], data[out])
                                for a, b, c, out in zip(
                                    a_components,
                                    b_components,
                                    c_components,
                                    out_components,
                                )
                            ]
                            return merge_style_add_diff_gr(
                                data[model_name_a],
                                data[model_name_b],
                                data[model_name_c],
                                data[speech_style_slider],
                                data[new_name],
                                style_tuple_list,
                            )

                        style_merge_btn.click(
                            _merge_add_diff,
                            inputs=set(
                                a_components
                                + b_components
                                + c_components
                                + out_components
                                + [
                                    model_name_a,
                                    model_name_b,
                                    model_name_c,
                                    speech_style_slider,
                                    new_name,
                                ]
                            ),
                            outputs=[info_style_merge, style],
                        )
                    else:  # weighted_sum

                        def _merge_weighted_sum(data):
                            style_tuple_list = [
                                (data[a], data[b], data[c], data[out])
                                for a, b, c, out in zip(
                                    a_components,
                                    b_components,
                                    c_components,
                                    out_components,
                                )
                            ]
                            return merge_style_weighted_sum_gr(
                                data[model_name_a],
                                data[model_name_b],
                                data[model_name_c],
                                data[model_a_coeff],
                                data[model_b_coeff],
                                data[model_c_coeff],
                                data[new_name],
                                style_tuple_list,
                            )

                        style_merge_btn.click(
                            _merge_weighted_sum,
                            inputs=set(
                                a_components
                                + b_components
                                + c_components
                                + out_components
                                + [
                                    model_name_a,
                                    model_name_b,
                                    model_name_c,
                                    model_a_coeff,
                                    model_b_coeff,
                                    model_c_coeff,
                                    new_name,
                                ]
                            ),
                            outputs=[info_style_merge, style],
                        )

            with gr.Row():
                add_btn = gr.Button("スタイルを増やす")
                del_btn = gr.Button("スタイルを減らす")
            add_btn.click(
                lambda x: x + 1,
                inputs=[style_count],
                outputs=[style_count],
            )
            del_btn.click(
                lambda x: x - 1 if x > 1 else 1,
                inputs=[style_count],
                outputs=[style_count],
            )
            style_merge_btn = gr.Button("スタイルのマージ", variant="primary")

            info_style_merge = gr.Textbox(label="情報")

        method.change(
            method_change,
            inputs=[method],
            outputs=[
                method_desc,
                c_col,
                model_a_coeff,
                model_b_coeff,
                model_c_coeff,
                weight_row,
                use_slerp_instead_of_lerp,
            ],
        )
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
        model_name_c.change(
            model_holder.update_model_files_for_gradio,
            inputs=[model_name_c],
            outputs=[model_path_c],
        )

        refresh_button.click(
            lambda: update_three_model_names_dropdown(model_holder),
            outputs=[
                model_name_a,
                model_path_a,
                model_name_b,
                model_path_b,
                model_name_c,
                model_path_c,
            ],
        )

        model_merge_button.click(
            merge_models_gr,
            inputs=[
                model_path_a,
                model_path_b,
                model_path_c,
                model_a_coeff,
                model_b_coeff,
                model_c_coeff,
                method,
                new_name,
                voice_slider,
                voice_pitch_slider,
                speech_style_slider,
                tempo_slider,
                use_slerp_instead_of_lerp,
            ],
            outputs=[info_model_merge, style],
        )

        # style_merge_button.click(
        #     merge_style_gr,
        #     inputs=[
        #         model_name_a,
        #         model_name_b,
        #         model_name_c,
        #         method,
        #         speech_style_slider,
        #         new_name,
        #         style_triple_list,
        #     ],
        #     outputs=[info_style_merge, style],
        # )

        tts_button.click(
            simple_tts,
            inputs=[new_name, text_input, style, emotion_weight],
            outputs=[tts_info, audio_output],
        )

    return app


if __name__ == "__main__":
    model_holder = TTSModelHolder(
        assets_root, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    app = create_merge_app(model_holder)
    app.launch(inbrowser=True)
