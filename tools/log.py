"""
logger封装
"""
from loguru import logger

from .stdout_wrapper import SAFE_STDOUT

# 移除所有默认的处理器
logger.remove()

# 自定义格式并添加到标准输出
log_format = (
    "<g>{time:MM-DD HH:mm:ss}</g> |<lvl>{level:^8}</lvl>| {file}:{line} | {message}"
)

logger.add(SAFE_STDOUT, format=log_format, backtrace=True, diagnose=True)
