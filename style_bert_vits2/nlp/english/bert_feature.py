from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models, onnx_bert_models


if TYPE_CHECKING:
    import torch


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> torch.Tensor:
    """
    英語のテキストから BERT の特徴量を抽出する (PyTorch 推論)

    Args:
        text (str): 英語のテキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        device (str): 推論に利用するデバイス
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        torch.Tensor: BERT の特徴量
    """

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = bert_models.load_model(Languages.EN, device_map=device)
    bert_models.transfer_model(Languages.EN, device)

    style_res_mean = None
    with torch.no_grad():
        tokenizer = bert_models.load_tokenizer(Languages.EN)
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  # type: ignore
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        if assist_text:
            style_inputs = tokenizer(assist_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)  # type: ignore
            style_res = model(**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - assist_text_weight)
                + style_res_mean.repeat(word2phone[i], 1) * assist_text_weight
            )
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T


def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]],
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
) -> NDArray[Any]:
    """
    英語のテキストから BERT の特徴量を抽出する (ONNX 推論)

    Args:
        text (str): 英語のテキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        onnx_providers (list[str]): ONNX 推論で利用する ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider など)
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)

    Returns:
        NDArray[Any]: BERT の特徴量
    """

    tokenizer = onnx_bert_models.load_tokenizer(Languages.EN)
    inputs = tokenizer(text, return_tensors="np")

    session = onnx_bert_models.load_model(
        language=Languages.EN,
        onnx_providers=onnx_providers,
    )
    output_name = session.get_outputs()[0].name
    res = session.run(
        [output_name],
        {
            "input_ids": cast(NDArray[Any], inputs["input_ids"]).astype(np.int64),
            "attention_mask": cast(NDArray[Any], inputs["attention_mask"]).astype(np.int64),
        },
    )[0]

    style_res_mean = None
    if assist_text:
        style_inputs = tokenizer(assist_text, return_tensors="np")
        style_res = session.run(
            [output_name],
            {
                "input_ids": cast(NDArray[Any], style_inputs["input_ids"]).astype(np.int64),
                "attention_mask": cast(NDArray[Any], style_inputs["attention_mask"]).astype(np.int64),
            },
        )[0]
        style_res_mean = np.mean(style_res, axis=0)

    assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        if assist_text:
            assert style_res_mean is not None
            repeat_feature = (
                np.tile(res[i], (word2phone[i], 1)) * (1 - assist_text_weight)
                + np.tile(style_res_mean, (word2phone[i], 1)) * assist_text_weight
            )
        else:
            repeat_feature = np.tile(res[i], (word2phone[i], 1))
        phone_level_feature.append(repeat_feature)

    phone_level_feature = np.concatenate(phone_level_feature, axis=0)

    return phone_level_feature.T
