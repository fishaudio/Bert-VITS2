from typing import Any

import numpy as np
from numpy.typing import NDArray


def adjust_voice(
    fs: int,
    wave: NDArray[Any],
    pitch_scale: float = 1.0,
    intonation_scale: float = 1.0,
) -> tuple[int, NDArray[Any]]:

    if pitch_scale == 1.0 and intonation_scale == 1.0:
        # 初期値の場合は、音質劣化を避けるためにそのまま返す
        return fs, wave

    try:
        import pyworld
    except ImportError:
        raise ImportError(
            "pyworld is not installed. Please install it by `pip install pyworld`"
        )

    # pyworld で f0 を加工して合成
    # pyworld よりもよいのがあるかもしれないが……
    ## pyworld は Cython で書かれているが、スタブファイルがないため型補完が全く効かない…

    wave = wave.astype(np.double)

    # 質が高そうだしとりあえずharvestにしておく
    f0, t = pyworld.harvest(wave, fs)  # type: ignore

    sp = pyworld.cheaptrick(wave, f0, t, fs)  # type: ignore
    ap = pyworld.d4c(wave, f0, t, fs)  # type: ignore

    non_zero_f0 = [f for f in f0 if f != 0]
    f0_mean = sum(non_zero_f0) / len(non_zero_f0)

    for i, f in enumerate(f0):
        if f == 0:
            continue
        f0[i] = pitch_scale * f0_mean + intonation_scale * (f - f0_mean)

    wave = pyworld.synthesize(f0, sp, ap, fs)  # type: ignore
    return fs, wave
