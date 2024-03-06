"""
以下に記述されている関数のコメントはリファクタリング時に GPT-4 に生成させたもので、
コードと完全に一致している保証はない。あくまで参考程度とすること。
"""

import math
import torch
from torch.nn import functional as F
from typing import Any


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """
    モジュールの重みを初期化する

    Args:
        m (torch.nn.Module): 重みを初期化する対象のモジュール
        mean (float): 正規分布の平均
        std (float): 正規分布の標準偏差
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """
    カーネルサイズと膨張率からパディングの大きさを計算する

    Args:
        kernel_size (int): カーネルのサイズ
        dilation (int): 膨張率

    Returns:
        int: 計算されたパディングの大きさ
    """
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape: list[list[Any]]) -> list[Any]:
    """
    パディングの形状を変換する

    Args:
        pad_shape (list[list[Any]]): 変換前のパディングの形状

    Returns:
        list[Any]: 変換後のパディングの形状
    """
    layer = pad_shape[::-1]
    new_pad_shape = [item for sublist in layer for item in sublist]
    return new_pad_shape


def intersperse(lst: list[Any], item: Any) -> list[Any]:
    """
    リストの要素の間に特定のアイテムを挿入する

    Args:
        lst (list[Any]): 元のリスト
        item (Any): 挿入するアイテム

    Returns:
        list[Any]: 新しいリスト
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p: torch.Tensor, logs_p: torch.Tensor, m_q: torch.Tensor, logs_q: torch.Tensor) -> torch.Tensor:
    """
    2つの正規分布間の KL ダイバージェンスを計算する

    Args:
        m_p (torch.Tensor): P の平均
        logs_p (torch.Tensor): P の対数標準偏差
        m_q (torch.Tensor): Q の平均
        logs_q (torch.Tensor): Q の対数標準偏差

    Returns:
        torch.Tensor: KL ダイバージェンスの値。
    """
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl


def rand_gumbel(shape: torch.Size) -> torch.Tensor:
    """
    Gumbel 分布からサンプリングし、オーバーフローを防ぐ

    Args:
        shape (torch.Size): サンプルの形状

    Returns:
        torch.Tensor: Gumbel 分布からのサンプル
    """
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    """
    引数と同じ形状のテンソルで、Gumbel 分布からサンプリングする

    Args:
        x (torch.Tensor): 形状を基にするテンソル

    Returns:
        torch.Tensor: Gumbel 分布からのサンプル
    """
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4) -> torch.Tensor:
    """
    テンソルからセグメントをスライスする

    Args:
        x (torch.Tensor): 入力テンソル
        ids_str (torch.Tensor): スライスを開始するインデックス
        segment_size (int, optional): スライスのサイズ (デフォルト: 4)

    Returns:
        torch.Tensor: スライスされたセグメント
    """
    gather_indices = ids_str.view(x.size(0), 1, 1).repeat(
        1, x.size(1), 1
    ) + torch.arange(segment_size, device=x.device)
    return torch.gather(x, 2, gather_indices)


def rand_slice_segments(x: torch.Tensor, x_lengths: torch.Tensor | None = None, segment_size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ランダムなセグメントをスライスする

    Args:
        x (torch.Tensor): 入力テンソル
        x_lengths (torch.Tensor, optional): 各バッチの長さ (デフォルト: None)
        segment_size (int, optional): スライスのサイズ (デフォルト: 4)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: スライスされたセグメントと開始インデックス
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t  # type: ignore
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=0)  # type: ignore
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length: int, channels: int, min_timescale: float = 1.0, max_timescale: float = 1.0e4) -> torch.Tensor:
    """
    1D タイミング信号を取得する

    Args:
        length (int): シグナルの長さ
        channels (int): シグナルのチャネル数
        min_timescale (float, optional): 最小のタイムスケール (デフォルト: 1.0)
        max_timescale (float, optional): 最大のタイムスケール (デフォルト: 1.0e4)

    Returns:
        torch.Tensor: タイミング信号
    """
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(x: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4) -> torch.Tensor:
    """
    1D タイミング信号をテンソルに追加する

    Args:
        x (torch.Tensor): 入力テンソル
        min_timescale (float, optional): 最小のタイムスケール (デフォルト: 1.0)
        max_timescale (float, optional): 最大のタイムスケール (デフォルト: 1.0e4)

    Returns:
        torch.Tensor: タイミング信号が追加されたテンソル
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4, axis: int = 1) -> torch.Tensor:
    """
    1D タイミング信号をテンソルに連結する

    Args:
        x (torch.Tensor): 入力テンソル
        min_timescale (float, optional): 最小のタイムスケール (デフォルト: 1.0)
        max_timescale (float, optional): 最大のタイムスケール (デフォルト: 1.0e4)
        axis (int, optional): 連結する軸 (デフォルト: 1)

    Returns:
        torch.Tensor: タイミング信号が連結されたテンソル
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length: int) -> torch.Tensor:
    """
    後続のマスクを生成する

    Args:
        length (int): マスクのサイズ

    Returns:
        torch.Tensor: 生成されたマスク
    """
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script  # type: ignore
def fused_add_tanh_sigmoid_multiply(input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor) -> torch.Tensor:
    """
    加算、tanh、sigmoid の活性化関数を組み合わせた演算を行う

    Args:
        input_a (torch.Tensor): 入力テンソル A
        input_b (torch.Tensor): 入力テンソル B
        n_channels (torch.Tensor): チャネル数

    Returns:
        torch.Tensor: 演算結果
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def shift_1d(x: torch.Tensor) -> torch.Tensor:
    """
    与えられたテンソルを 1D でシフトする

    Args:
        x (torch.Tensor): シフトするテンソル

    Returns:
        torch.Tensor: シフトされたテンソル
    """
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """
    シーケンスマスクを生成する

    Args:
        length (torch.Tensor): 各シーケンスの長さ
        max_length (int | None): 最大のシーケンス長さ。指定されていない場合は length の最大値を使用

    Returns:
        torch.Tensor: 生成されたシーケンスマスク
    """
    if max_length is None:
        max_length = length.max()  # type: ignore
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)  # type: ignore
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    パスを生成する

    Args:
        duration (torch.Tensor): 各時間ステップの持続時間
        mask (torch.Tensor): マスクテンソル

    Returns:
        torch.Tensor: 生成されたパス
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters: torch.Tensor | list[torch.Tensor], clip_value: float | None, norm_type: float = 2.0) -> float:
    """
    勾配の値をクリップする

    Args:
        parameters (torch.Tensor | list[torch.Tensor]): クリップするパラメータ
        clip_value (float | None): クリップする値。None の場合はクリップしない
        norm_type (float): ノルムの種類

    Returns:
        float: 総ノルム
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0.0
    for p in parameters:
        assert p.grad is not None
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
