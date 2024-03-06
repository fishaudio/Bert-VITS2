import json
import os
from collections import defaultdict
from random import shuffle
from typing import Optional

import click
from tqdm import tqdm

from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT
from config import config
from text.cleaner import clean_text

preprocess_text_config = config.preprocess_text_config


# Count lines for tqdm
def count_lines(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for _ in file)


@click.command()
@click.option(
    "--transcription-path",
    default=preprocess_text_config.transcription_path,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=preprocess_text_config.cleaned_path)
@click.option("--train-path", default=preprocess_text_config.train_path)
@click.option("--val-path", default=preprocess_text_config.val_path)
@click.option(
    "--config-path",
    default=preprocess_text_config.config_path,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-lang", default=preprocess_text_config.val_per_lang)
@click.option("--max-val-total", default=preprocess_text_config.max_val_total)
@click.option("--clean/--no-clean", default=preprocess_text_config.clean)
@click.option("-y", "--yml_config")
@click.option("--use_jp_extra", is_flag=True)
@click.option("--yomi_error", default="raise")
def preprocess(
    transcription_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_lang: int,
    max_val_total: int,
    clean: bool,
    yml_config: str,  # 这个不要删
    use_jp_extra: bool,
    yomi_error: str,
):
    assert yomi_error in ["raise", "skip", "use"]
    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path + ".cleaned"

    error_log_path = os.path.join(os.path.dirname(cleaned_path), "text_error.log")
    if os.path.exists(error_log_path):
        os.remove(error_log_path)
    error_count = 0

    if clean:
        total_lines = count_lines(transcription_path)
        with open(cleaned_path, "w", encoding="utf-8") as out_file:
            with open(transcription_path, "r", encoding="utf-8") as trans_file:
                for line in tqdm(trans_file, file=SAFE_STDOUT, total=total_lines):
                    try:
                        utt, spk, language, text = line.strip().split("|")
                        norm_text, phones, tones, word2ph = clean_text(
                            text=text,
                            language=language,
                            use_jp_extra=use_jp_extra,
                            raise_yomi_error=(yomi_error != "use"),
                        )

                        out_file.write(
                            "{}|{}|{}|{}|{}|{}|{}\n".format(
                                utt,
                                spk,
                                language,
                                norm_text,
                                " ".join(phones),
                                " ".join([str(i) for i in tones]),
                                " ".join([str(i) for i in word2ph]),
                            )
                        )
                    except Exception as e:
                        logger.error(
                            f"An error occurred at line:\n{line.strip()}\n{e}",
                            encoding="utf-8",
                        )
                        with open(error_log_path, "a", encoding="utf-8") as error_log:
                            error_log.write(f"{line.strip()}\n{e}\n\n")
                        error_count += 1

    transcription_path = cleaned_path
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(transcription_path, "r", encoding="utf-8") as f:
        audioPaths = set()
        countSame = 0
        countNotFound = 0
        for line in f.readlines():
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            if utt in audioPaths:
                # 过滤数据集错误：相同的音频匹配多个文本，导致后续bert出问题
                logger.warning(f"Same audio matches multiple texts: {line}")
                countSame += 1
                continue
            if not os.path.isfile(utt):
                # 过滤数据集错误：不存在对应音频
                logger.warning(f"Audio not found: {utt}")
                countNotFound += 1
                continue
            audioPaths.add(utt)
            spk_utt_map[language].append(line)
            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1
        if countSame > 0 or countNotFound > 0:
            logger.warning(
                f"Total repeated audios: {countSame}, Total number of audio not found: {countNotFound}"
            )

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_lang]
        train_list += utts[val_per_lang:]

    shuffle(val_list)
    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    json_config = json.load(open(config_path, encoding="utf-8"))
    json_config["data"]["spk2id"] = spk_id_map
    json_config["data"]["n_speakers"] = len(spk_id_map)
    # 新增写入：写入训练版本、数据集路径
    # json_config["version"] = latest_version
    json_config["data"]["training_files"] = os.path.normpath(train_path).replace(
        "\\", "/"
    )
    json_config["data"]["validation_files"] = os.path.normpath(val_path).replace(
        "\\", "/"
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=2, ensure_ascii=False)
    if error_count > 0:
        if yomi_error == "skip":
            logger.warning(
                f"An error occurred in {error_count} lines. Proceed with lines without errors. Please check {error_log_path} for details."
            )
        else:
            # yom_error == "raise"と"use"の場合。
            # "use"の場合は、そもそもyomi_error = Falseで処理しているので、
            # ここが実行されるのは他の例外のときなので、エラーをraiseする。
            logger.error(
                f"An error occurred in {error_count} lines. Please check {error_log_path} for details."
            )
            raise Exception(
                f"An error occurred in {error_count} lines. Please check `Data/you_model_name/text_error.log` file for details."
            )
            # 何故か{error_log_path}をraiseすると文字コードエラーが起きるので上のように書いている
    else:
        logger.info(
            "Training set and validation set generation from texts is complete!"
        )


if __name__ == "__main__":
    preprocess()
