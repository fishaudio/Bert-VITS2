import argparse
import shutil
from tempfile import NamedTemporaryFile

from loguru import logger


def remove_chars_from_file(chars_to_remove, input_file, output_file):
    rm_cnt = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            NamedTemporaryFile('w', delete=False, encoding='utf-8') as f_tmp:
        for line in f_in:
            if any(char in line for char in chars_to_remove):
                logger.info(f"删除了这一行:\n {line.strip()}")
                rm_cnt += 1
            else:
                f_tmp.write(line)

    shutil.move(f_tmp.name, output_file)
    logger.critical(f"总计移除了: {rm_cnt} 行")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove lines from a file containing specified characters.")

    parser.add_argument("-c", "--chars", type=str, required=True,
                        help="String of characters. If a line contains any of these characters, it will be removed.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the input file.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the output file.")

    args = parser.parse_args()

    # Setting up basic logging configuration for loguru
    logger.add("removed_lines.log", rotation="1 MB")

    remove_chars_from_file(args.chars, args.input, args.output)
