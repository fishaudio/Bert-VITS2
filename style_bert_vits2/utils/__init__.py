from collections.abc import Sequence
from typing import Any, Union


def torch_device_to_onnx_providers(
    device: str,
) -> Sequence[Union[str, tuple[str, dict[str, Any]]]]:
    if device.startswith("cuda"):
        return [
            # cudnn_conv_algo_search を DEFAULT にすると推論速度が大幅に向上する
            # ref: https://medium.com/neuml/debug-onnx-gpu-performance-c9290fe07459
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            # CUDA が利用できない場合、可能であれば DirectML を利用する
            ("DmlExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {}),
        ]
    else:
        return ["CPUExecutionProvider"]
