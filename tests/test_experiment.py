# -*- coding: utf-8 -*-
"""阶段 9a 测试关卡：维度对照实验脚本（机制测试）

通过标准（见 开发计划.md 第二期）：
    1. 小轮数（3 epoch）+ 两个小维度跑通全流程
    2. 输出 JSON 结构完整：每个维度含 psnr/ssim/best_loss 字段
    3. 判定规则 pick_minimal_sufficient 逻辑正确（用构造数据单测）

说明：机制测试不要求指标高（3 epoch 训不出好模型），
指标高低由阶段 9b 的正式实验（600 epoch）负责。
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_latent import pick_minimal_sufficient, run_experiment
from src.logger import ROOT_LOGGER_NAME, setup_logger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_DIR = os.path.join(ROOT, "tests", "_exp_tmp")


def _reset_root_logger():
    """清空根 logger 的 handler（与 test_logger.py 同款，保证测试隔离）"""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


def test_pick_rule_logic():
    """判定规则单测：用构造数据覆盖三种情形"""
    # 情形1：128 与最优(512)差 0.5dB 且 >=30，应选更小的 128
    records = [
        {"latent_dim": 512, "psnr": 33.0, "ssim": 0.87},
        {"latent_dim": 256, "psnr": 32.8, "ssim": 0.87},
        {"latent_dim": 128, "psnr": 32.5, "ssim": 0.86},
        {"latent_dim": 64, "psnr": 28.0, "ssim": 0.80},  # 掉出 30dB，不合格
    ]
    assert pick_minimal_sufficient(records)["latent_dim"] == 128

    # 情形2：小维度 PSNR 虽 >=30 但比最优低超过 1dB，不能入选
    records = [
        {"latent_dim": 256, "psnr": 34.0, "ssim": 0.87},
        {"latent_dim": 64, "psnr": 31.0, "ssim": 0.85},  # 差 3dB > 1dB
    ]
    assert pick_minimal_sufficient(records)["latent_dim"] == 256

    # 情形3：全部低于 30dB，应返回 None
    records = [
        {"latent_dim": 256, "psnr": 25.0, "ssim": 0.8},
        {"latent_dim": 64, "psnr": 24.0, "ssim": 0.8},
    ]
    assert pick_minimal_sufficient(records) is None


def test_experiment_machinery():
    """全流程机制测试：3 epoch × 2 个维度，检查 JSON 结构完整性

    同时验证关键日志真实写入了文件：logging 会静默吞掉格式化错误
    （如格式串混入全角字符），只有检查日志文件内容才能抓住这类 bug。
    """
    _reset_root_logger()
    log_dir = os.path.join(TMP_DIR, "logs")
    try:
        setup_logger(log_dir=log_dir)
        args = argparse.Namespace(
            dims=[32, 16],  # 用极小维度，纯粹跑机制，快
            epochs=3,
            img_dir=os.path.join(ROOT, "分钟k线图"),
            out_dir=os.path.join(TMP_DIR, "results"),
            work_dir=os.path.join(TMP_DIR, "ckpt"),
            seed=42,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        result = run_experiment(args)

        # 关卡 1：记录数量 = 维度数量，且按维度降序
        records = result["records"]
        assert [r["latent_dim"] for r in records] == [32, 16], "维度记录不完整或顺序错误"

        # 关卡 2：每条记录字段齐全且值域合法
        for r in records:
            assert 0 < r["psnr"] < 100, f"PSNR 非法: {r}"
            assert 0 <= r["ssim"] <= 1, f"SSIM 非法: {r}"
            assert r["best_loss"] > 0 and r["train_seconds"] >= 0

        # 关卡 3：JSON 文件存在且可解析、内容一致
        with open(result["json_path"], encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["records"] == records, "JSON 与内存记录不一致"
        assert "rule" in saved and "chosen" in saved, "JSON 缺少规则/结论字段"

        # 关卡 4：关键日志真实落盘（表头/结论行若格式化失败不会写入文件）
        log_file = os.path.join(log_dir, f"kline_{time.strftime('%Y%m%d')}.log")
        content = open(log_file, encoding="utf-8").read()
        assert "维度" in content and "PSNR(dB)" in content, "汇总表头未写入日志"
        assert "结论" in content, "实验结论未写入日志"
    finally:
        _reset_root_logger()
        shutil.rmtree(TMP_DIR, ignore_errors=True)
