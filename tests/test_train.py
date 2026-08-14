# -*- coding: utf-8 -*-
"""阶段 4 测试关卡：训练脚本（冒烟测试）

通过标准（见 开发计划.md 阶段4实验记录）：
    1. 36 张小轮数训练后，MSE < 纯白图基线 MSE（0.0266）的 50%
       —— 纯白基线是"模型只学会画背景"的退化解水平，显著低于它
       才能证明模型在学 K 线结构，而不是只学背景
    2. checkpoint 文件生成，且 torch.load 后模型可前向
注意：本测试实际跑 GPU/CPU 训练数十 epoch，耗时 1~2 分钟属正常。
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_cae import CAE
from src.train import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 纯白图基线的 MSE（阶段4实验实测值）：模型如果只学会输出白色背景，
# MSE 就停在这个水平附近。冒烟关卡要求显著低于它，证明学到了K线结构
WHITE_BASELINE_MSE = 0.0266


def _smoke_args(ckpt_dir: str) -> argparse.Namespace:
    """构造冒烟训练参数：与阶段4验证实验一致（latent 256 / batch 8 / 纯MSE）

    150 epoch 依据：实验记录显示 120 epoch 时 MSE=0.0103 已过关卡线
    （0.0133），150 epoch（实测约 0.006）留出稳定余量。
    """
    return argparse.Namespace(
        img_dir=os.path.join(ROOT, "tests", "fixtures", "kline36"),
        epochs=150,
        batch_size=8,
        lr=1e-3,
        latent_dim=256,
        alpha=1.0,  # 纯 MSE，见 开发计划.md 阶段4实验记录
        device="cuda" if torch.cuda.is_available() else "cpu",
        ckpt_dir=ckpt_dir,
        log_every=30,
    )


def test_smoke_training():
    """冒烟训练：MSE 击穿纯白基线的 50% + checkpoint 可加载可前向

    临时 checkpoint 放在项目内（D盘）而非 pytest 默认的 tmp_path：
    本机 C 盘已满，写 C 盘临时目录会导致 checkpoint 保存失败，
    测试结束后用 finally 清理，不污染正式的 checkpoints/ 目录。
    """
    tmp_dir = os.path.join(ROOT, "tests", "_smoke_ckpt_tmp")
    torch.manual_seed(42)  # 固定随机种子，保证测试结果可复现
    try:
        # vars() 把 Namespace 转成字典后按关键字传入（接口隔离改造）
        result = train(**vars(_smoke_args(tmp_dir)))
        _assert_gates(result)
    finally:
        # shutil.rmtree 递归删除临时目录；ignore_errors 防止文件占用导致测试报错
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _assert_gates(result: dict):
    """冒烟测试的两道关卡（拆出函数便于 try/finally 包裹主流程）"""
    history = result["history"]

    # 关卡 1：末期损失（后3个epoch均值，alpha=1 时即纯 MSE）
    # 必须低于纯白基线的 50%，证明模型学到的不只是白色背景
    late = sum(history[-3:]) / 3
    assert late < WHITE_BASELINE_MSE * 0.5, (
        f"末期 MSE {late:.6f} 未击穿纯白基线 {WHITE_BASELINE_MSE} 的 50%"
        f"（关卡线 {WHITE_BASELINE_MSE * 0.5:.6f}），模型可能只学会了输出背景"
    )

    # 关卡 2：checkpoint 存在且可恢复出能前向的模型
    ckpt_path = result["ckpt_path"]
    assert os.path.isfile(ckpt_path), "checkpoint 文件未生成"
    # weights_only=False：checkpoint 里除权重外还存了超参数配置字典
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CAE(latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        x_hat, z = model(torch.rand(1, 3, 448, 448))
    assert x_hat.shape == (1, 3, 448, 448)
    assert z.shape == (1, ckpt["latent_dim"])
