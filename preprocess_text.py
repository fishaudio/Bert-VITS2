import json
from collections import defaultdict
from random import shuffle
from typing import Optional
import os

from tqdm import tqdm
import click
from text.cleaner import clean_text_auto
from tools.filelist_utils import get_text_bert_auto, LangType
from tools.sentence import split_by_language
from config import config
from infer import latest_version
import torch
import torch.multiprocessing as mp
import utils

preprocess_text_config = config.preprocess_text_config


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
):
    device = config.bert_gen_config.device
    if config.bert_gen_config.use_multi_device:
        rank = mp.current_process()._identity
        rank = rank[0] if len(rank) > 0 else 0
        if torch.cuda.is_available():
            gpu_id = rank % torch.cuda.device_count()
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")

    hps = utils.get_hparams_from_file(config_path)
    add_blank = hps.data.add_blank

    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path + ".cleaned"

    if clean:
        with open(cleaned_path, "w", encoding="utf-8") as out_file:
            with open(transcription_path, "r", encoding="utf-8") as trans_file:
                lines = trans_file.readlines()
                # print(lines, ' ', len(lines))
                if len(lines) != 0:
                    for line in tqdm(lines):
                        # try:
                        utt, spk, text = line.strip().split("|")
                        bert_path = utt.replace(".WAV", ".wav").replace(
                            ".wav", ".bert.pt"
                        )

                        sentences_list = split_by_language(
                            text, target_languages=["zh", "en"]
                        )
                        (
                            norm_texts,
                            phones,
                            tones,
                            word2phs,
                            langs,
                        ) = clean_text_auto(sentences_list)
                        (
                            bert,
                            ja_bert,
                            en_bert,
                            agg_phones,
                            agg_tones,
                            lang_ids,
                        ) = get_text_bert_auto(
                            norm_texts,
                            phones,
                            tones,
                            word2phs,
                            langs,
                            device,
                            add_blank,
                        )
                        out_file.write(
                            "{}|{}|{}|{}|{}|{}|{}\n".format(
                                utt,
                                spk,
                                text,
                                json.dumps(norm_texts),
                                json.dumps(agg_phones.tolist()),
                                json.dumps(agg_tones.tolist()),
                                json.dumps(lang_ids.tolist()),
                            )
                        )
                        berts = torch.stack((bert, ja_bert, en_bert))
                        assert berts.shape[-1] == agg_phones.shape[-1]
                        torch.save(berts, bert_path)

                        # except Exception as e:
                        #     print(line)
                        #     print(f"生成训练集和验证集时发生错误！, 详细信息:\n{e}")

    transcription_path = cleaned_path
    lines = []
    spk_id_map = {}
    current_sid = 0

    with open(transcription_path, "r", encoding="utf-8") as f:
        audioPaths = set()
        countSame = 0
        countNotFound = 0
        for line in f.readlines():
            (
                utt,
                spk,
                norm_texts,
                phones,
                tones,
                word2phs,
                langs,
            ) = line.strip().split("|")
            if utt in audioPaths:
                # 过滤数据集错误：相同的音频匹配多个文本，导致后续bert出问题
                print(f"重复音频文本：{line}")
                countSame += 1
                continue
            if not os.path.isfile(utt):
                # 过滤数据集错误：不存在对应音频
                print(f"没有找到对应的音频：{utt}")
                countNotFound += 1
                continue
            audioPaths.add(utt)
            lines.append(line)
            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1
        print(f"总重复音频数：{countSame}，总未找到的音频数:{countNotFound}")

    train_list = []
    val_list = []

    shuffle(lines)
    val_list.extend(lines[:val_per_lang])
    train_list.extend(lines[val_per_lang:])

    shuffle(val_list)
    if len(val_list) > max_val_total:
        train_list.extend(val_list[max_val_total:])
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
    json_config["version"] = latest_version
    json_config["data"]["training_files"] = os.path.normpath(train_path).replace(
        "\\", "/"
    )
    json_config["data"]["validation_files"] = os.path.normpath(val_path).replace(
        "\\", "/"
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(json_config, f, indent=2, ensure_ascii=False)
    print("训练集和验证集生成完成！")


if __name__ == "__main__":
    preprocess()
