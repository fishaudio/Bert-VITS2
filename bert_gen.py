import torch
from multiprocessing import Pool
import commons
import utils
from tqdm import tqdm
from text import cleaned_text_to_sequence, get_bert
import argparse
import torch.multiprocessing as mp


def process_line(line):
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    wav_path, _, language_str, text, phones, tone, word2ph = line.strip().split("|")
    phone = phones.split(" ")
    tone = [int(i) for i in tone.split(" ")]
    word2ph = [int(i) for i in word2ph.split(" ")]
    word2ph = [i for i in word2ph]
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    bert_path = wav_path.replace(".wav", ".bert.pt")

    try:
        bert = torch.load(bert_path)
        assert bert.shape[-1] == len(phone)
    except Exception:
        bert = get_bert(text, word2ph, language_str, device)
        assert bert.shape[-1] == len(phone)
        torch.save(bert, bert_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/config.json")
    parser.add_argument("--num_processes", type=int, default=2)
    args = parser.parse_args()
    config_path = args.config
    hps = utils.get_hparams_from_file(config_path)
    lines = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    with open(hps.data.validation_files, encoding="utf-8") as f:
        lines.extend(f.readlines())

    num_processes = args.num_processes
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(process_line, lines), total=len(lines)):
            pass
