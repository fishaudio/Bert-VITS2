from typing import Any, Sequence, Union

import onnxruntime


def torch_device_to_onnx_providers(
    device: str,
) -> Sequence[Union[str, tuple[str, dict[str, Any]]]]:
    """
    PyTorch のデバイス種別を ONNX の ExecutionProvider に変換する

    Args:
        device (str): PyTorch のデバイス種別

    Returns:
        Sequence[Union[str, tuple[str, dict[str, Any]]]]: ExecutionProvider のリスト
    """

    if device.startswith("cuda"):
        return [
            # cudnn_conv_algo_search を DEFAULT にすると推論速度が大幅に向上する
            ## ref: https://medium.com/neuml/debug-onnx-gpu-performance-c9290fe07459
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "DEFAULT"}),
            # CUDA が利用できない場合、可能であれば DirectML を利用する (明示的な device_id 指定が必要)
            ## device_id: 0 は、システムにインストールされているプライマリディスプレイ用 GPU に対応する
            ## プライマリディスプレイ用 GPU (GPU 0) よりも性能の高い GPU が接続されている環境では、 適宜 device_id を変更する必要がある
            ## ref: https://github.com/w-okada/voice-changer/issues/410#issuecomment-1627994911
            ("DmlExecutionProvider", {"device_id": 0}),
            # arena_extend_strategy を kSameAsRequested にすると、推論セッションによって作成される
            # メモリアリーナが、実際に推論に必要な容量以上にメモリを確保する問題を防ぐことができる
            ## ref: https://github.com/microsoft/onnxruntime/issues/11627#issuecomment-1137668551
            ## ref: https://skottmckay.github.io/onnxruntime/docs/reference/api/c-api.html
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ]  # fmt: skip
    else:
        return [
            ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
        ]


def get_onnx_device_options(
    onnx_session: onnxruntime.InferenceSession,
    onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]],
) -> tuple[str, int, onnxruntime.RunOptions]:
    """
    ONNX 推論時のデバイス関連のオプションを取得する

    Args:
        onnx_session (onnxruntime.InferenceSession): 初期化済みの ONNX セッション
        onnx_providers (Sequence[Union[str, tuple[str, dict[str, Any]]]]): ExecutionProvider のリスト

    Returns:
        tuple[str, int, onnxruntime.RunOptions]: 入力テンソルの転送に使用するデバイス種別, デバイス ID, 実行オプション
    """

    # 実際に推論に用いられる ExecutionProvider を取得
    first_provider = onnx_session.get_providers()[0]

    # 入力テンソルを転送する推論デバイスを取得
    ## GPU への I/O Binding は CUDA と DirectML 以外はサポートされていないので、
    ## それ以外の ExecutionProvider が指定された場合は CPU に入力テンソルを転送する
    if first_provider == "CUDAExecutionProvider":
        device_type = "cuda"
    elif first_provider == "DmlExecutionProvider":
        device_type = "dml"
    else:
        device_type = "cpu"

    # 入力テンソルを転送する GPU デバイスの ID を取得
    ## ExecutionProvider に指定したオプションの中から device_id を取得し、入力テンソルの転送先として指定する
    ## InferenceSession で利用するデバイス ID と入力テンソルの転送先デバイス ID は一致している必要がある
    ## 本来は ExecutionProvider に指定したオプションは InferenceSession.get_provider_options() で取得できるはずだが、
    ## 手元環境では DmlExecutionProvider のみ常に空の辞書が返されることがあったため、当面 onnx_providers から直接オプションを取り出している
    ## CPU 推論時の device_id は 0 で固定
    device_id = 0
    onnx_providers_dict: dict[str, dict[str, Any]] = {}
    if device_type != "cpu":
        for provider in onnx_providers:
            if isinstance(provider, tuple):
                provider_name, options = provider
                onnx_providers_dict[provider_name] = options
            else:
                onnx_providers_dict[provider] = {}
        first_provider_options = onnx_providers_dict[first_provider]
        if "device_id" in first_provider_options:
            device_id = int(first_provider_options["device_id"])

    # 推論後にメモリアリーナを縮小し、メモリを解放する
    ## onnxruntime.SessionOptions の enable_cpu_mem_arena (デフォルト: True) により、デフォルトでは CPU 推論時にメモリアリーナが構築される
    ## メモリアリーナを無効化すると、推論にのみ使用されたメモリは推論後にすべて解放されるが、一方パフォーマンスがかなり落ちる
    ## そこでメモリアリーナを有効化した上で、推論後にメモリアリーナを縮小し、何回も音声合成するほど漸進的にメモリが消費される現象を回避する
    ## この設定は GPU 推論時にもある程度効果があると思われる
    ## ref: https://onnxruntime.ai/docs/get-started/with-c.html
    ## ref: https://github.com/microsoft/onnxruntime/issues/9313#issuecomment-2182919186
    ## ref: https://github.com/microsoft/onnxruntime/issues/11627
    ## ref: https://github.com/microsoft/onnxruntime/issues/22297#issuecomment-2438629814
    ## ref: https://github.com/microsoft/onnxruntime/blob/v1.20.1/onnxruntime/test/python/onnxruntime_test_python.py#L1626-L1647
    ## ref: https://github.com/microsoft/onnxruntime/blob/v1.20.1/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h#L19-L27
    run_options = onnxruntime.RunOptions()
    if first_provider == "CPUExecutionProvider":
        # CPU 推論時は cpu:0 を指定
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")  # fmt: skip
    elif first_provider == "DmlExecutionProvider":
        # DirectML 推論時はこのオプションはサポートされていないようなので、何も指定しない
        # "The registered allocator for device-id combination is not an arena based allocator: gpu:0" のようなエラーが出る…
        pass
    elif first_provider == "CUDAExecutionProvider":
        # CUDA 推論時は cpu:0;gpu:(device_id) を指定
        ## 公式テストコードを読む限り、CUDA だけでなく CPU のメモリも明示的に解放した方がよいらしい
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"cpu:0;gpu:{device_id}")  # fmt: skip

    return device_type, device_id, run_options
