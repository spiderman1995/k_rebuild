# -*- coding: utf-8 -*-
"""评估脚本（流程位置：训练产出 checkpoint 后的质量验收环节）

功能：
    1. 载入 checkpoint（自动恢复 latent_dim 等配置）
    2. 对数据集逐张计算 SSIM 和 PSNR
    3. 把"原图 | 重建图"并排对比图保存到 results/
    4. 打印指标汇总表（均值 / 最差样本）

用法示例（在项目根目录执行）:
    python src/evaluate.py
    python src/evaluate.py --ckpt checkpoints/cae_best.pth --out-dir results

指标解读（校准依据见 开发计划.md 阶段4实验记录）：
    - PSNR ≥ 30dB 且 SSIM ≥ 0.86 为重建达标线
    - 参照：纯白图基线 PSNR=15.8dB / SSIM=0.834；原图横移1像素 SSIM=0.869
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ssim as ssim_func

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compat import load_checkpoint
from src.dataset import KLineDataset
from src.logger import get_logger, setup_logger
from src.machine import get_machine_tag
from src.model_cae import CAE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 模块级子 logger：日志带 'kline.eval' 前缀
log = get_logger("eval")

# 输出按机器分文件夹（防多机 git 冲突，见 get_machine_tag）
MACHINE = get_machine_tag()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CAE 重建质量评估")
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "cae_best.pth"),
                        help="checkpoint 路径")
    parser.add_argument("--img-dir", default=os.path.join(ROOT, "分钟k线图"),
                        help="评估图片文件夹")
    parser.add_argument("--out-dir",
                        default=os.path.join(ROOT, "results", MACHINE),
                        help="对比图与指标输出目录（默认按机器分文件夹）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> CAE:
    """从 checkpoint 恢复模型（结构参数存于 checkpoint，无需手动对齐）

    参数:
        ckpt_path: checkpoint 文件路径
        device:    加载到的设备
    返回:
        eval 模式的 CAE 模型
    """
    # load_checkpoint 兼容 torch 1.10~2.x（203 机器钉在 1.10，见 compat.py）
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model = CAE(latent_dim=ckpt["latent_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()  # 固定 BatchNorm 统计量（评估必须，否则指标失真）
    return model


def save_comparison(x: torch.Tensor, x_hat: torch.Tensor, name: str,
                    out_dir: str):
    """保存"原图|重建图"横向拼接对比图（单一职责：只管可视化落盘）

    参数:
        x, x_hat: 原图与重建图, shape (1, 3, H, W)
        name:     原文件名（输出为 compare_<name>）
        out_dir:  输出目录
    """
    # dim=2 是宽度维，横向拼接为 (3, H, 2W)，左原右重建
    pair = torch.cat([x[0], x_hat[0]], dim=2).cpu()
    # CHW -> HWC 并还原到 0~255 的 uint8 像素
    arr = (pair.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(out_dir, f"compare_{name}"))


def evaluate(ckpt: str, img_dir: str, out_dir: str, device: str = None,
             save_images: bool = True) -> dict:
    """执行完整评估流程

    显式关键字参数（接口隔离）：程序化调用只传需要的参数；
    命令行入口用 evaluate(**vars(parse_args())) 调用。

    参数:
        ckpt:        checkpoint 路径
        img_dir:     评估图片文件夹
        out_dir:     指标与对比图输出目录
        device:      计算设备，None 时自动选择
        save_images: False 时只算指标不存对比图（供大批量实验调用）
    返回:
        dict: {'per_image': [(文件名, ssim, psnr), ...],
               'mean_ssim': 平均SSIM, 'mean_psnr': 平均PSNR}
    """
    setup_logger()  # 幂等初始化：控制台 + logs/kline_日期.log
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(ckpt, device)
    dataset = KLineDataset(img_dir)
    log.info("评估启动 | checkpoint: %s | 样本: %d | 设备: %s",
             ckpt, len(dataset), device)

    per_image = []
    with torch.no_grad():  # 纯评估不需要梯度，省显存提速
        for i in range(len(dataset)):
            img, name = dataset[i]
            x = img.unsqueeze(0).to(device)   # (3,448,448) -> (1,3,448,448)
            x_hat, _z = model(x)

            # size_average=False 返回 batch 内每张图各自的 SSIM，取 [0] 得标量
            s = ssim_func(x_hat, x, data_range=1.0, size_average=False)[0].item()
            mse = torch.mean((x_hat - x) ** 2).item()
            # PSNR = 10*log10(MAX^2/MSE)，像素范围 [0,1] 时 MAX=1
            psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
            per_image.append((name, s, psnr))

            if save_images:
                save_comparison(x, x_hat, name, out_dir)

    mean_ssim = sum(s for _, s, _ in per_image) / len(per_image)
    mean_psnr = sum(p for _, _, p in per_image) / len(per_image)
    # min(key=...) 按元组第2/3个元素找最差样本
    worst_ssim = min(per_image, key=lambda t: t[1])
    worst_psnr = min(per_image, key=lambda t: t[2])

    # ---------- 汇总输出 ----------
    log.info("%-28s %8s %10s", "文件名", "SSIM", "PSNR(dB)")
    for name, s, p in per_image:
        log.info("%-28s %8.4f %10.2f", name, s, p)
    log.info("-" * 50)
    log.info("平均: SSIM=%.4f, PSNR=%.2fdB", mean_ssim, mean_psnr)
    log.info("最差 SSIM: %s (%.4f)", worst_ssim[0], worst_ssim[1])
    log.info("最差 PSNR: %s (%.2fdB)", worst_psnr[0], worst_psnr[2])
    if save_images:
        log.info("对比图已保存到: %s", out_dir)

    return {"per_image": per_image, "mean_ssim": mean_ssim, "mean_psnr": mean_psnr}


if __name__ == "__main__":
    # vars() 把 Namespace 转成字典，** 解包为关键字参数
    evaluate(**vars(parse_args()))
