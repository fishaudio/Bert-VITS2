import os
import argparse
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from config import config


def get_filename(path):
    file_name = os.path.basename(path)
    file_name = os.path.splitext(file_name)[0]
    return file_name


def process(items):
    temp_dir, file, args = items

    in_file = os.path.join(args.in_dir, file)
    file_name = get_filename(file)

    out_file = os.path.join(args.out_dir, file)

    if not os.path.exists(f"{temp_dir}/{file_name}/vocals.wav"):
        if not os.path.exists(in_file):
            print(f"File not exists: {in_file}")
            return
        print(f"Process: {in_file}")

        command = f"demucs --two-stems=vocals {in_file}"
        os.system(command)
        print(f"Seperated: {in_file}")

    # 复制到outdir
    if not os.path.exists(out_file):
        os.system(f"cp {temp_dir}/{file_name}/vocals.wav {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        default=config.resample_config.in_dir,
        help="path to source dir",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=config.resample_config.out_dir,
        help="path to target dir",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="cpu_processes",
    )
    args, _ = parser.parse_known_args()
    # autodl 无卡模式会识别出46个cpu
    if args.processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes = args.processes
    pool = Pool(processes=processes)

    tasks = []

    for _, _, filenames in os.walk(args.in_dir):
        # 子级目录
        temp_dir = r"./separated/htdemucs/"
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        for filename in filenames:
            if filename.endswith(".wav"):
                twople = (temp_dir, filename, args)
                tasks.append(twople)

    for _ in tqdm(
        pool.imap_unordered(process, tasks),
    ):
        pass

    pool.close()
    pool.join()

    print("音频分离人声完毕!")
