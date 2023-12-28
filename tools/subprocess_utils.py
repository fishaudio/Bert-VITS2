import subprocess
import sys

from .log import logger

python = sys.executable


def run_script_with_log(cmd: list[str]) -> tuple[bool, str]:
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        [python] + cmd,
        stdout=sys.stdout,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Error: {' '.join(cmd)}")
        print(result.stderr)
        return False, result.stderr
    elif result.stderr:
        logger.warning(f"Warning: {' '.join(cmd)}")
        print(result.stderr)
        return True, result.stderr
    logger.success(f"Success: {' '.join(cmd)}")
    return True, ""


def second_elem_of(original_function):
    def inner_function(*args, **kwargs):
        return original_function(*args, **kwargs)[1]

    return inner_function
