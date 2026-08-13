# -*- coding: utf-8 -*-
"""阶段 5 测试关卡：评估脚本

通过标准（见 开发计划.md）：
    1. 指标表行数 = 36（每张图一行）
    2. 36 张对比图文件全部生成
    3. 指标值域合法：SSIM ∈ [0,1]，PSNR > 0 且有限

说明：本测试只验证评估"机制"正确，不要求指标高——用一个未训练的
随机初始化模型生成 checkpoint 即可，指标高低是阶段 6 的验收内容。
"""

import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import evaluate
from src.model_cae import CAE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 临时目录放项目内（D盘）：本机 C 盘已满，见 test_train.py 同款说明
TMP_DIR = os.path.join(ROOT, "tests", "_eval_tmp")


def _make_untrained_checkpoint(path: str, latent_dim: int = 64):
    """生成一个未训练模型的 checkpoint，格式与 train.py 保存的一致"""
    model = CAE(latent_dim=latent_dim)
    torch.save({
        "model_state": model.state_dict(),
        "latent_dim": latent_dim,
        "alpha": 1.0,
        "epoch": 0,
        "loss": float("inf"),
    }, path)


def test_evaluate_machinery():
    """评估机制验证：行数、文件、值域三项检查"""
    os.makedirs(TMP_DIR, exist_ok=True)
    ckpt_path = os.path.join(TMP_DIR, "untrained.pth")
    out_dir = os.path.join(TMP_DIR, "results")
    try:
        _make_untrained_checkpoint(ckpt_path)
        args = argparse.Namespace(
            ckpt=ckpt_path,
            img_dir=os.path.join(ROOT, "分钟k线图"),
            out_dir=out_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        result = evaluate(args)

        # 关卡 1：指标表行数 = 36
        per_image = result["per_image"]
        assert len(per_image) == 36, f"指标行数 {len(per_image)} != 36"

        # 关卡 2：36 张对比图全部生成
        # startswith 过滤出评估输出的 compare_ 前缀文件
        n_files = len([f for f in os.listdir(out_dir) if f.startswith("compare_")])
        assert n_files == 36, f"对比图数量 {n_files} != 36"

        # 关卡 3：指标值域合法
        for name, s, p in per_image:
            assert 0.0 <= s <= 1.0, f"{name} SSIM 越界: {s}"
            assert 0.0 < p < float("inf"), f"{name} PSNR 非法: {p}"
        # 未训练模型的平均指标不应高于合理上限（防止评估逻辑写反）
        assert result["mean_ssim"] < 0.9, "未训练模型 SSIM 过高，评估逻辑可疑"
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
