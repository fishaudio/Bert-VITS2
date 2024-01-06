import subprocess
import sys

from .log import logger
from .stdout_wrapper import SAFE_STDOUT

python = sys.executable


def run_script_with_log(cmd: list[str], ignore_warning=False) -> tuple[bool, str]:
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        [python] + cmd,
        stdout=SAFE_STDOUT,  # type: ignore
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Error: {' '.join(cmd)}\n{result.stderr}")
        return False, result.stderr
    elif result.stderr and not ignore_warning:
        logger.warning(f"Warning: {' '.join(cmd)}\n{result.stderr}")
        return True, result.stderr
    logger.success(f"Success: {' '.join(cmd)}")
    return True, ""


def second_elem_of(original_function):
    def inner_function(*args, **kwargs):
        return original_function(*args, **kwargs)[1]

    return inner_function
