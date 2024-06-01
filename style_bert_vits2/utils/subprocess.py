import subprocess
import sys
from typing import Any, Callable

from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


def run_script_with_log(
    cmd: list[str], ignore_warning: bool = False
) -> tuple[bool, str]:
    """
    指定されたコマンドを実行し、そのログを記録する。

    Args:
        cmd: 実行するコマンドのリスト
        ignore_warning: 警告を無視するかどうかのフラグ

    Returns:
        tuple[bool, str]: 実行が成功したかどうかのブール値と、エラーまたは警告のメッセージ（ある場合）
    """

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        [sys.executable] + cmd,
        stdout=SAFE_STDOUT,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        logger.error(f"Error: {' '.join(cmd)}\n{result.stderr}")
        return False, result.stderr
    elif result.stderr and not ignore_warning:
        logger.warning(f"Warning: {' '.join(cmd)}\n{result.stderr}")
        return True, result.stderr
    logger.success(f"Success: {' '.join(cmd)}")

    return True, ""


def second_elem_of(
    original_function: Callable[..., tuple[Any, Any]]
) -> Callable[..., Any]:
    """
    与えられた関数をラップし、その戻り値の 2 番目の要素のみを返す関数を生成する。

    Args:
        original_function (Callable[..., tuple[Any, Any]])): ラップする元の関数

    Returns:
        Callable[..., Any]: 元の関数の戻り値の 2 番目の要素のみを返す関数
    """

    def inner_function(*args, **kwargs) -> Any:  # type: ignore
        return original_function(*args, **kwargs)[1]

    return inner_function
