from typing import Any, Sequence, Union


def torch_device_to_onnx_providers(
    device: str,
) -> Sequence[Union[str, tuple[str, dict[str, Any]]]]:
    if device.startswith("cuda"):
        # cudnn_conv_algo_search を DEFAULT にすると推論速度が大幅に向上する
        # ref: https://medium.com/neuml/debug-onnx-gpu-performance-c9290fe07459
        return [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            ("CPUExecutionProvider", {}),
        ]
    else:
        return ["CPUExecutionProvider"]
