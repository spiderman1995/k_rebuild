# -*- coding: utf-8 -*-
"""训练脚本（流程位置：串联 dataset/model/losses，产出模型 checkpoint）

用法示例（在项目根目录执行）:
    python src/train.py --epochs 300 --latent-dim 256 --alpha 0.5
    python src/train.py --epochs 20 --batch-size 4        # 快速冒烟
可调参数见 parse_args()。训练完成后最优模型保存到 checkpoints/cae_best.pth，
checkpoint 内含模型权重和全部配置，评估脚本可自恢复模型结构。
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

# 保证以 `python src/train.py` 方式直接运行时能找到 src 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import KLineDataset
from src.logger import get_logger, setup_logger
from src.losses import CombinedLoss
from src.model_cae import CAE

# 项目根目录（本文件在 src/ 下，取上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模块级子 logger：日志带 'kline.train' 前缀，方便按模块过滤
log = get_logger("train")


def parse_args():
    """解析命令行参数，全部超参数集中在此处定义"""
    parser = argparse.ArgumentParser(description="CAE 训练脚本")
    parser.add_argument("--img-dir", default=os.path.join(ROOT, "分钟k线图"),
                        help="训练图片文件夹")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批大小（6GB 显存下 448 输入建议 <=8）")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 初始学习率")
    parser.add_argument("--latent-dim", type=int, default=256,
                        help="特征向量维度（特征充分性实验的核心变量）")
    # 默认纯 MSE（alpha=1）：实验发现 SSIM 项在训练早期会把梯度引向
    # "输出全白背景"的退化解（白色占91%像素，SSIM对背景过于满意），
    # 纯 MSE 能更快学到 K 线结构；SSIM 保留为评估指标。详见 开发计划.md 阶段4记录
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="组合损失中 MSE 的权重, L=a*MSE+(1-a)*(1-SSIM)，默认纯MSE")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="训练设备, cuda 或 cpu")
    parser.add_argument("--ckpt-dir", default=os.path.join(ROOT, "checkpoints"),
                        help="checkpoint 保存目录")
    parser.add_argument("--log-every", type=int, default=10,
                        help="每多少个 epoch 打印一次日志")
    # action='store_true'：命令行出现 --resume 即为 True，不带则为 False
    parser.add_argument("--resume", action="store_true",
                        help="从 ckpt-dir 下的 cae_best.pth 断点续训")
    return parser.parse_args()


def train(args) -> dict:
    """执行完整训练流程

    参数:
        args: parse_args() 返回的命令行参数命名空间
    返回:
        dict: {'best_loss': 最优epoch平均损失, 'ckpt_path': checkpoint路径,
               'history': 每epoch平均损失列表}
    """
    setup_logger()  # 幂等初始化：控制台 + logs/kline_日期.log
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---------- 数据 ----------
    dataset = KLineDataset(args.img_dir)
    # shuffle=True 每个 epoch 打乱样本顺序；小数据集下 num_workers=0 开销最低
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ---------- 模型 / 损失 / 优化器 ----------
    model = CAE(latent_dim=args.latent_dim).to(device)
    criterion = CombinedLoss(alpha=args.alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 损失停滞（10 个 epoch 不再下降）时学习率减半，帮助后期精细收敛
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    log.info("训练启动 | 设备: %s | 样本: %d | 参数量: %s | latent_dim: %d | "
             "batch: %d | lr: %.1e | alpha: %.2f | epochs: %d",
             device, len(dataset), f"{model.count_parameters():,}",
             args.latent_dim, args.batch_size, args.lr, args.alpha, args.epochs)

    best_loss = float("inf")
    start_epoch = 1
    ckpt_path = os.path.join(args.ckpt_dir, "cae_best.pth")

    # ---------- 断点续训 ----------
    # getattr 兼容测试代码中手工构造、没有 resume 字段的 Namespace
    if getattr(args, "resume", False) and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt["latent_dim"] != args.latent_dim:
            raise ValueError(
                f"checkpoint latent_dim={ckpt['latent_dim']} "
                f"与本次参数 {args.latent_dim} 不一致，拒绝续训"
            )
        model.load_state_dict(ckpt["model_state"])
        # 恢复 Adam 的动量/二阶矩 与 学习率调度器状态，保证续训曲线平滑衔接；
        # 旧格式 checkpoint 没有这两项时跳过（Adam 冷启动会短暂扰动损失）
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        best_loss = ckpt["loss"]
        start_epoch = ckpt["epoch"] + 1
        log.info("断点续训：从 epoch %d 继续（历史最优损失 %.6f，lr %.2e）",
                 start_epoch, best_loss, optimizer.param_groups[0]["lr"])

    history = []  # 记录本次运行每个 epoch 的平均损失，供测试和绘图使用
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()  # 切换训练模式（启用 BatchNorm 的批统计更新）
        epoch_loss = 0.0
        for imgs, _names in loader:
            imgs = imgs.to(device)
            x_hat, _z = model(imgs)
            loss = criterion(x_hat, imgs)

            optimizer.zero_grad()  # 清空上一步梯度（PyTorch 梯度默认累加）
            loss.backward()
            optimizer.step()
            # loss.item() 取标量值；乘以 batch 张数以便算逐样本平均
            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(dataset)
        history.append(avg_loss)
        scheduler.step(avg_loss)  # 把当前损失喂给调度器判断是否降学习率

        if avg_loss < best_loss:
            best_loss = avg_loss
            # checkpoint 同时保存权重、优化器状态和配置：
            # 评估端无需手动对齐超参数，续训端可平滑衔接 Adam 动量
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "latent_dim": args.latent_dim,
                "alpha": args.alpha,
                "epoch": epoch,
                "loss": best_loss,
            }, ckpt_path)
            # debug 级别：INFO 下不刷屏，排查 checkpoint 问题时把级别调低即可看到
            log.debug("checkpoint 更新于 epoch %d（loss %.6f）", epoch, best_loss)

        if epoch % args.log_every == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]  # 当前学习率（可能已被调度器衰减）
            log.info("epoch %4d/%d | loss %.6f | best %.6f | lr %.2e | 耗时 %.0fs",
                     epoch, args.epochs, avg_loss, best_loss, lr_now, time.time() - t0)

    log.info("训练完成，最优损失 %.6f，checkpoint: %s", best_loss, ckpt_path)
    return {"best_loss": best_loss, "ckpt_path": ckpt_path, "history": history}


if __name__ == "__main__":
    train(parse_args())
