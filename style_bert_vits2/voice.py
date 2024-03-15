from typing import Any

import numpy as np
import pyworld
from numpy.typing import NDArray


def adjust_voice(
    fs: int,
    wave: NDArray[Any],
    pitch_scale: float = 1.0,
    intonation_scale: float = 1.0,
) -> tuple[int, NDArray[Any]]:
    """
    音声のピッチと抑揚を調整する。
    変更すると若干音質が劣化するので、どちらも初期値のままならそのまま返す。

    Args:
        fs (int): 音声のサンプリング周波数
        wave (NDArray[Any]): 音声データ
        pitch_scale (float, optional): ピッチの高さ. Defaults to 1.0.
        intonation_scale (float, optional): 抑揚の平均からの変更比率. Defaults to 1.0.

    Returns:
        tuple[int, NDArray[Any]]: 調整後の音声データのサンプリング周波数と音声データ
    """

    if pitch_scale == 1.0 and intonation_scale == 1.0:
        # 初期値の場合は、音質劣化を避けるためにそのまま返す
        return fs, wave

    # pyworld で f0 を加工して合成
    # pyworld よりもよいのがあるかもしれないが……

    wave = wave.astype(np.double)

    # 質が高そうだしとりあえずharvestにしておく
    f0, t = pyworld.harvest(wave, fs)

    sp = pyworld.cheaptrick(wave, f0, t, fs)
    ap = pyworld.d4c(wave, f0, t, fs)

    non_zero_f0 = [f for f in f0 if f != 0]
    f0_mean = sum(non_zero_f0) / len(non_zero_f0)

    for i, f in enumerate(f0):
        if f == 0:
            continue
        f0[i] = pitch_scale * f0_mean + intonation_scale * (f - f0_mean)

    wave = pyworld.synthesize(f0, sp, ap, fs)
    return fs, wave
