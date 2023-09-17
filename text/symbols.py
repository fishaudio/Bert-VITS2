from itertools import chain

punctuation = ["!", "?", "…", ",", ".", "'", "-"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# chinese
zh_symbols = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
num_zh_tones = 6

# japanese
jp_symbols = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "w",
    "y",
    "z",
    "zy",
]
num_jp_tones = 1

# English
en_symbols = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
num_en_tones = 4

symbol_systems = [
    (pad, [pad]),
    ("ZH", zh_symbols),
    ("JP", jp_symbols),
    ("EN", en_symbols),
    ("PU", pu_symbols),
]

symbols = list(
    chain.from_iterable(
        [f"{lang}_{s}" for s in symbols] for lang, symbols in symbol_systems
    )
)

# combine all tones
num_tones = 1 + num_zh_tones + num_jp_tones + num_en_tones  # 1 for padding

# language maps
language_id_map = {pad: 0, "ZH": 1, "JP": 2, "EN": 3}
language_unicode_range_map = {
    "ZH": [(0x4E00, 0x9FFF)],
    "JP": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    "EN": [(0x0000, 0x007F)],
}
num_languages = len(language_id_map.keys())

language_tone_start_map = {
    pad: 0,
    "ZH": 1,
    "JP": 1 + num_zh_tones,
    "EN": 1 + num_zh_tones + num_jp_tones,
}
