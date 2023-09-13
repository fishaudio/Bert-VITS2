import json
from collections import defaultdict
from random import shuffle

from tqdm import tqdm
import click
from text.parser import parse_text_to_segments, segments_g2p, get_bert_alignment
from concurrent.futures import ProcessPoolExecutor


def get_data(line: str):
    utt, spk, text = line.strip().split("|")
    segments = parse_text_to_segments(text)
    words, phones, tones, word2ph, languages = segments_g2p(segments)

    bert_alignment = get_bert_alignment(words, phones, word2ph)
    tokens = [i["token"] for i in bert_alignment]
    token_ids = [i["token_id"] for i in bert_alignment]
    offsets = [i["offset"] for i in bert_alignment]

    return dict(
        path=utt,
        spk=spk,
        words=words,
        phones=phones,
        tones=tones,
        word2ph=word2ph,
        languages=languages,
        tokens=tokens,
        token_ids=token_ids,
        offsets=offsets,
    )


def get_data_safe(line: str):
    try:
        return get_data(line)
    except Exception as e:
        return e


@click.command()
@click.option(
    "--transcription-path",
    default="filelists/genshin.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--train-path", default="filelists/train.jsonl")
@click.option("--val-path", default="filelists/val.jsonl")
@click.option(
    "--config-path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--append/--no-append", default=False)
def main(
    transcription_path: str,
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    append: bool,
):
    if append is False:
        # Clear the file
        open(train_path, "w", encoding="utf-8").close()
        open(val_path, "w", encoding="utf-8").close()

    current_sid = 0
    spk_utt_map = defaultdict(list)
    spk_id_map = {}

    with ProcessPoolExecutor() as executor:
        lines = open(transcription_path, encoding="utf-8").readlines()

        for data in tqdm(executor.map(get_data_safe, lines), total=len(lines)):
            if isinstance(data, Exception):
                print("err!", line, data)
                continue

            spk = data["spk"]
            data = json.dumps(data, ensure_ascii=False)

            spk_utt_map[spk].append(data)
            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "a", encoding="utf-8") as f:
        for line in train_list:
            f.write(f"{line}\n")

    with open(val_path, "a", encoding="utf-8") as f:
        for line in val_list:
            f.write(f"{line}\n")

    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
