import re

import inflect


__INFLECT = inflect.engine()
__COMMA_NUMBER_PATTERN = re.compile(r"([0-9][0-9\,]+[0-9])")
__DECIMAL_NUMBER_PATTERN = re.compile(r"([0-9]+\.[0-9]+)")
__POUNDS_PATTERN = re.compile(r"£([0-9\,]*[0-9]+)")
__DOLLARS_PATTERN = re.compile(r"\$([0-9\.\,]*[0-9]+)")
__ORDINAL_PATTERN = re.compile(r"[0-9]+(st|nd|rd|th)")
__NUMBER_PATTERN = re.compile(r"[0-9]+")


def normalize_text(text: str) -> str:
    text = __normalize_numbers(text)
    text = replace_punctuation(text)
    text = re.sub(r"([,;.\?\!])([\w])", r"\1 \2", text)
    return text


def replace_punctuation(text: str) -> str:
    REPLACE_MAP = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "．": ".",
        "…": "...",
        "···": "...",
        "・・・": "...",
        "·": ",",
        "・": ",",
        "、": ",",
        "$": ".",
        "“": "'",
        "”": "'",
        '"': "'",
        "‘": "'",
        "’": "'",
        "（": "'",
        "）": "'",
        "(": "'",
        ")": "'",
        "《": "'",
        "》": "'",
        "【": "'",
        "】": "'",
        "[": "'",
        "]": "'",
        "—": "-",
        "−": "-",
        "～": "-",
        "~": "-",
        "「": "'",
        "」": "'",
    }
    pattern = re.compile("|".join(re.escape(p) for p in REPLACE_MAP))
    replaced_text = pattern.sub(lambda x: REPLACE_MAP[x.group()], text)
    # replaced_text = re.sub(
    #     r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    #     + "".join(punctuation)
    #     + r"]+",
    #     "",
    #     replaced_text,
    # )
    return replaced_text


def __normalize_numbers(text: str) -> str:
    text = re.sub(__COMMA_NUMBER_PATTERN, __remove_commas, text)
    text = re.sub(__POUNDS_PATTERN, r"\1 pounds", text)
    text = re.sub(__DOLLARS_PATTERN, __expand_dollars, text)
    text = re.sub(__DECIMAL_NUMBER_PATTERN, __expand_decimal_point, text)
    text = re.sub(__ORDINAL_PATTERN, __expand_ordinal, text)
    text = re.sub(__NUMBER_PATTERN, __expand_number, text)
    return text


def __expand_dollars(m: re.Match[str]) -> str:
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def __remove_commas(m: re.Match[str]) -> str:
    return m.group(1).replace(",", "")


def __expand_ordinal(m: re.Match[str]) -> str:
    return __INFLECT.number_to_words(m.group(0))  # type: ignore


def __expand_number(m: re.Match[str]) -> str:
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + __INFLECT.number_to_words(num % 100)  # type: ignore
        elif num % 100 == 0:
            return __INFLECT.number_to_words(num // 100) + " hundred"  # type: ignore
        else:
            return __INFLECT.number_to_words(
                num, andword="", zero="oh", group=2  # type: ignore
            ).replace(
                ", ", " "
            )  # type: ignore
    else:
        return __INFLECT.number_to_words(num, andword="")  # type: ignore


def __expand_decimal_point(m: re.Match[str]) -> str:
    return m.group(1).replace(".", " point ")
