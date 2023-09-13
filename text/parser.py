import re
import string
from text.symbols import language_id_map, language_unicode_range_map, punctuation
import itertools
from text import chinese, japanese, english
from functools import lru_cache
from transformers import AutoTokenizer

LANGUAGE_TO_MODULE_MAP = {
    "ZH": chinese,
    "JP": japanese,
    "EN": english,
}

# This files is designed to parse the text, doing some normalization,
# and return an annotated text.
# Example: 1, 2, <JP>3</JP>, 4, <ZH>5</ZH>
# For better compatibility, we also support tree-like structure, like:
# 1, 2, <JP>3, <EN>4</EN></JP>, 5, <ZH>6</ZH>


class Segment:
    def __init__(self, text, language=None):
        self.text = text
        self.language = language.upper() if language is not None else None

    def __repr__(self):
        return f"<Segment {self.language}: '{self.text}'>"

    def __str__(self):
        return self.text


SYMBOLS_MAPPING = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
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
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

REPLACE_SYMBOL_REGEX = re.compile(
    "|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys())
)
ALL_KNOWN_UTF8_RANGE = list(
    itertools.chain.from_iterable(language_unicode_range_map.values())
)
REMOVE_UNKNOWN_SYMBOL_REGEX = re.compile(
    "[^"
    + "".join(
        f"{re.escape(chr(start))}-{re.escape(chr(end))}"
        for start, end in ALL_KNOWN_UTF8_RANGE
    )
    + "]"
)


def parse_text_to_segments(text, order=None):
    """
    Parse the text and return a list of segments.
    :param text: The text to be parsed.
    :param order: The order of languages. If None, use ["ZH", "JP", "EN"].
    :return: A list of segments.
    """

    if order is None:
        order = ["ZH", "JP", "EN"]

    order = [language.upper() for language in order]
    assert all(language in language_id_map for language in order)

    # Clean the text
    text = text.strip()
    # Replace all chinese symbols with their english counterparts
    text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)
    text = REMOVE_UNKNOWN_SYMBOL_REGEX.sub("", text)

    texts = re.split(r"(<.*?>)", text)
    texts = [text for text in texts if text.strip() != ""]

    stack = []
    segments = []
    for text in texts:
        if text.startswith("<") and text.endswith(">") and text[1] != "/":
            current_language = text[1:-1]
            # The following line should be updated later
            assert current_language.upper() in language_id_map
            stack.append(current_language)
        elif text.startswith("</") and text.endswith(">"):
            language = stack.pop()
            if language != text[2:-1]:
                raise ValueError(f"Language mismatch: {language} != {text[2:-1]}")
        elif stack:
            segments.append(Segment(text, stack[-1]))
        else:
            segments.extend(parse_unknown_segment(text, order))

    return segments


def parse_unknown_segment(text, order):
    last_idx, last_language = 0, None

    for idx, char in enumerate(text):
        if char in punctuation or char in string.digits:
            # If the punctuation / number is in the middle of the text,
            # we should not split the text.
            detected_language = last_language or order[0]
        else:
            detected_language = None

            for language in order:
                for start, end in language_unicode_range_map[language]:
                    if start <= ord(char) <= end:
                        detected_language = language
                        break

                if detected_language is not None:
                    break

            assert (
                detected_language is not None
            ), f"Incorrect language: {char}, clean before calling this function."

        if last_language is None:
            last_language = detected_language

        if detected_language != last_language:
            yield Segment(text[last_idx:idx], last_language)
            last_idx = idx
            last_language = detected_language

    if last_idx != len(text):
        yield Segment(text[last_idx:], last_language)


def segments_g2p(segments):
    all_words, all_phones, all_tones, all_word2ph, all_languages = [], [], [], [], []

    for i in segments:
        words, phones, tones, word2ph = LANGUAGE_TO_MODULE_MAP[i.language].g2p(i.text)
        assert sum(word2ph) == len(phones) == len(tones)

        all_words.extend(words)
        all_phones.extend(phones)
        all_tones.extend(tones)
        all_word2ph.extend(word2ph)
        all_languages.extend([i.language] * len(phones))

        assert len(words) == len(word2ph), f"{i.language}, {words}, {word2ph}"

    assert sum(all_word2ph) == len(all_phones) == len(all_tones)
    assert len(all_word2ph) == len(all_words), f"{all_word2ph}, {all_words}"

    return all_words, all_phones, all_tones, all_word2ph, all_languages


@lru_cache(maxsize=-1)
def get_tokenizer(model_name="xlm-roberta-large"):
    return AutoTokenizer.from_pretrained(model_name)


def get_bert_alignment(words, phones, word2ph, model_name="xlm-roberta-large"):
    tokenizer = get_tokenizer(model_name)
    text = "".join(words)
    assignment = [None] * len(text)

    word2ph_cum_sum = [0]
    word_pos_cum_sum = [0]
    for i in range(len(word2ph)):
        word2ph_cum_sum.append(word2ph_cum_sum[-1] + word2ph[i])
        word_pos_cum_sum.append(word_pos_cum_sum[-1] + len(words[i]))
        word = words[i]

        for j in range(len(word)):
            assignment[word_pos_cum_sum[i] + j] = (
                word2ph_cum_sum[-2],
                word2ph_cum_sum[-1],
            )

    encoded = tokenizer.encode_plus(text, return_offsets_mapping=True)

    # Print token to word mapping
    complex_tokens = [
        {
            "token_id": token_id,
            "offset": offset,
            "is_special": token_id in tokenizer.all_special_ids,
        }
        for token_id, offset in zip(encoded["input_ids"], encoded["offset_mapping"])
    ]

    # Decode the token
    for i in range(len(complex_tokens)):
        complex_tokens[i]["token"] = tokenizer.decode([complex_tokens[i]["token_id"]])

    # Use offset to map the word and then the phone
    for i in range(len(complex_tokens)):
        if complex_tokens[i]["is_special"] or complex_tokens[i]["token"].strip() == "":
            complex_tokens[i]["offset"] = None
            continue

        start, end = complex_tokens[i]["offset"]
        complex_tokens[i]["word"] = text[start:end]

        phone_start, phone_end = assignment[start:end][0]
        for a, b in assignment[start:end]:
            if a < phone_start:
                phone_start = a
            if b > phone_end:
                phone_end = b

        complex_tokens[i]["phones"] = phones[phone_start:phone_end]
        complex_tokens[i]["offset"] = (phone_start, phone_end)

    return complex_tokens


if __name__ == "__main__":
    segments = parse_text_to_segments(
        "毕业然后复活卡b站推荐bug<zh>加流量。<en>Hugging face, B GM</en>声音很大吗</zh>？那我改一下Ё。 <jp>君の虜になってしまえばきっと</jp>"  # noqa: E501
    )
    print(segments)

    all_words, all_phones, all_tones, all_word2ph, all_languages = segments_g2p(
        segments
    )
    print(all_words, all_phones, all_tones, all_word2ph, all_languages)
    print("".join(all_words))

    complex_tokens = get_bert_alignment(all_words, all_phones, all_word2ph)
    for i in complex_tokens:
        print(i)
