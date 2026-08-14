# -*- coding: utf-8 -*-
"""统一日志模块（流程位置：横切所有模块的基础设施）

功能：
    - setup_logger() 初始化根 logger 'kline'：同时输出到控制台和
      logs/kline_日期.log 文件（按天一个文件，追加写入）
    - 各模块通过 get_logger('train') 获得子 logger 'kline.train'，
      日志自动带模块名，便于按模块过滤分析问题
    - 防重复初始化：多次调用 setup_logger 不会重复添加 handler
      （否则每条日志会打印多遍）

用法：
    from src.logger import setup_logger, get_logger
    setup_logger()                 # 入口脚本调用一次
    log = get_logger('train')      # 各模块获取子 logger
    log.info('epoch %d loss %.6f', epoch, loss)
"""

import logging
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 根 logger 名：所有子 logger（kline.train / kline.eval...）挂在它下面，
# 只需给根挂 handler，子 logger 的日志会自动向上传播到根输出
ROOT_LOGGER_NAME = "kline"

# 日志行格式：时间 [级别] 模块名: 内容
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """初始化根 logger（幂等：重复调用直接返回已配置的 logger）

    参数:
        log_dir: 日志文件目录，默认项目根下的 logs/
        level:   日志级别，默认 INFO
    返回:
        配置好的根 logger
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if logger.handlers:  # 已初始化过，直接复用，防止 handler 重复导致日志翻倍
        return logger

    logger.setLevel(level)
    # propagate=False：阻止日志继续传给 python 全局根 logger，
    # 避免与其他库（如 pytest）的日志配置相互干扰、重复输出
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # 控制台输出
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 文件输出：按天一个文件，追加写入；encoding 显式 utf-8 防止中文乱码
    if log_dir is None:
        log_dir = os.path.join(ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"kline_{time.strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """获取模块子 logger（日志会带上 'kline.模块名' 前缀）

    参数:
        module_name: 模块名，如 'train' / 'eval' / 'dataset'
    返回:
        子 logger，输出走根 logger 的 handler
    """
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{module_name}")
