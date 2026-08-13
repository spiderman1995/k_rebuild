# -*- coding: utf-8 -*-
"""阶段 7 测试关卡：日志系统

通过标准（见 开发计划.md 第二期）：
    1. 日志同时输出到控制台和 logs/*.log 文件，行内含时间/级别/模块名
    2. 重复调用 setup_logger 不重复添加 handler（防止日志翻倍）
    3. 子 logger（kline.train 等）的日志正常写入文件
    4. 训练一遍后日志文件包含 epoch/损失/学习率字段
"""

import logging
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import ROOT_LOGGER_NAME, get_logger, setup_logger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 测试用独立日志目录（D盘项目内，测试后清理）
TMP_LOG_DIR = os.path.join(ROOT, "tests", "_log_tmp")


def _reset_root_logger():
    """清空根 logger 的 handler，让每个测试从干净状态开始

    直接操作 logging 模块的全局状态：logging.getLogger 同名返回同一对象，
    上一个测试添加的 handler 会残留到下一个测试，必须显式清理。
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


def _log_file_path():
    """当前日期对应的日志文件路径（与 logger.py 的命名规则一致）"""
    return os.path.join(TMP_LOG_DIR, f"kline_{time.strftime('%Y%m%d')}.log")


def test_log_to_console_and_file():
    """日志写入文件，且行内含时间戳、级别、模块名三要素"""
    _reset_root_logger()
    try:
        setup_logger(log_dir=TMP_LOG_DIR)
        get_logger("unittest").info("测试消息abc123")
        content = open(_log_file_path(), encoding="utf-8").read()
        assert "测试消息abc123" in content, "日志内容未写入文件"
        assert "[INFO]" in content, "日志缺少级别字段"
        assert "kline.unittest" in content, "日志缺少模块名"
        # 时间戳格式 'YYYY-MM-DD HH:MM:SS'：检查年份开头即可
        assert content.startswith(time.strftime("%Y-")), "日志缺少时间戳"
    finally:
        _reset_root_logger()
        shutil.rmtree(TMP_LOG_DIR, ignore_errors=True)


def test_setup_idempotent():
    """重复调用 setup_logger：handler 数量不增加，日志不翻倍"""
    _reset_root_logger()
    try:
        logger1 = setup_logger(log_dir=TMP_LOG_DIR)
        n_handlers = len(logger1.handlers)
        logger2 = setup_logger(log_dir=TMP_LOG_DIR)  # 第二次调用
        assert logger1 is logger2, "两次 setup 返回了不同 logger"
        assert len(logger2.handlers) == n_handlers, "重复 setup 导致 handler 翻倍"

        get_logger("dup").info("唯一消息xyz")
        content = open(_log_file_path(), encoding="utf-8").read()
        # count 统计子串出现次数：若 handler 重复，这条日志会写两遍
        assert content.count("唯一消息xyz") == 1, "同一条日志被写入多次"
    finally:
        _reset_root_logger()
        shutil.rmtree(TMP_LOG_DIR, ignore_errors=True)


def test_training_writes_log_fields():
    """跑 2 epoch 训练，日志文件应包含配置、epoch、损失、学习率字段"""
    import argparse

    import torch

    from src.train import train

    _reset_root_logger()
    tmp_ckpt = os.path.join(ROOT, "tests", "_log_train_tmp")
    try:
        setup_logger(log_dir=TMP_LOG_DIR)
        args = argparse.Namespace(
            img_dir=os.path.join(ROOT, "tests", "fixtures", "kline36"),
            epochs=2, batch_size=8, lr=1e-3, latent_dim=64, alpha=1.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
            ckpt_dir=tmp_ckpt, log_every=1, resume=False,
        )
        train(args)
        content = open(_log_file_path(), encoding="utf-8").read()
        assert "训练启动" in content and "latent_dim" in content, "缺少启动配置日志"
        assert "epoch" in content and "loss" in content, "缺少 epoch/损失日志"
        assert "lr" in content, "缺少学习率日志"
        assert "kline.dataset" in content, "缺少数据集加载日志"
        assert "训练完成" in content, "缺少结束日志"
    finally:
        _reset_root_logger()
        shutil.rmtree(TMP_LOG_DIR, ignore_errors=True)
        shutil.rmtree(tmp_ckpt, ignore_errors=True)
