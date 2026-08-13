# -*- coding: utf-8 -*-
"""最小充分维度对照实验（流程位置：框架验收后的核心特征实验）

目的：回答"特征向量最少要多少维，才装得下重建这张K线图所需的信息"。

方法（控制变量法）：
    除 latent_dim 外一切相同（同数据、同轮数、同学习率、同随机种子），
    对每个候选维度各训练一个 CAE 并评估 PSNR/SSIM，得到"维度→质量"曲线。

判定规则（见 开发计划.md 第二期阶段9）：
    满足 [PSNR >= 30dB] 且 [PSNR >= 最优维度PSNR - 1dB] 的最小维度
    即为"最小充分维度"。

用法示例（在项目根目录执行）:
    python src/experiment_latent.py                        # 默认 512/256/128/64
    python src/experiment_latent.py --dims 128 64 --epochs 100   # 自定义

注意：本地 36 张上的结论是初筛（衡量拟合能力）；迁移远程后应在
验证集上复核（衡量泛化充分性），用 --img-dir 指向验证集即可。
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import evaluate
from src.logger import get_logger, setup_logger
from src.train import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = get_logger("experiment")

# 达标线（校准依据见 开发计划.md 阶段4实验记录）
PSNR_FLOOR = 30.0   # PSNR 绝对下限 (dB)
PSNR_MARGIN = 1.0   # 允许比最优维度低多少 dB 仍算"不掉质量"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="latent 维度对照实验")
    # nargs='+' 表示接受一个或多个值: --dims 512 256 128 64
    parser.add_argument("--dims", type=int, nargs="+", default=[512, 256, 128, 64],
                        help="待对照的 latent 维度列表")
    parser.add_argument("--epochs", type=int, default=600, help="每个维度的训练轮数")
    parser.add_argument("--img-dir", default=os.path.join(ROOT, "分钟k线图"),
                        help="数据文件夹（远程复核时指向验证集目录）")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results"),
                        help="实验结果 JSON 输出目录")
    parser.add_argument("--work-dir", default=os.path.join(ROOT, "checkpoints", "latent_exp"),
                        help="各维度 checkpoint 的存放目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（控制变量）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_one_dim(dim: int, args) -> dict:
    """训练并评估单个维度，返回该维度的指标记录

    参数:
        dim:  latent 维度
        args: 命令行参数
    返回:
        dict: {'latent_dim', 'psnr', 'ssim', 'best_loss', 'train_seconds'}
    """
    ckpt_dir = os.path.join(args.work_dir, f"dim{dim}")
    t0 = time.time()

    # 每个维度都重设同一种子：保证权重初始化、数据打乱顺序全部一致，
    # 消除随机性差异，让维度成为唯一变量
    torch.manual_seed(args.seed)
    train_result = train(argparse.Namespace(
        img_dir=args.img_dir, epochs=args.epochs, batch_size=8, lr=1e-3,
        latent_dim=dim, alpha=1.0, device=args.device,
        ckpt_dir=ckpt_dir, log_every=max(1, args.epochs // 5), resume=False,
    ))

    eval_result = evaluate(argparse.Namespace(
        ckpt=train_result["ckpt_path"], img_dir=args.img_dir,
        out_dir=os.path.join(ckpt_dir, "eval"), device=args.device,
    ))

    record = {
        "latent_dim": dim,
        "psnr": round(eval_result["mean_psnr"], 2),
        "ssim": round(eval_result["mean_ssim"], 4),
        "best_loss": train_result["best_loss"],
        "train_seconds": round(time.time() - t0),
    }
    log.info("维度 %d 完成: PSNR=%.2fdB SSIM=%.4f 耗时 %ds",
             dim, record["psnr"], record["ssim"], record["train_seconds"])
    return record


def pick_minimal_sufficient(records: list) -> dict:
    """按判定规则选出最小充分维度

    规则: PSNR >= 30dB 且 PSNR >= 最优维度 - 1dB 的最小维度。
    若无维度达标，返回 None（说明所有候选维度都不够或训练不充分）。

    参数:
        records: run_one_dim 返回的记录列表
    返回:
        选中的记录 dict，或 None
    """
    best_psnr = max(r["psnr"] for r in records)
    # 先过滤出"达标"的维度，再取其中维度最小者
    qualified = [r for r in records
                 if r["psnr"] >= PSNR_FLOOR and r["psnr"] >= best_psnr - PSNR_MARGIN]
    if not qualified:
        return None
    return min(qualified, key=lambda r: r["latent_dim"])


def run_experiment(args) -> dict:
    """执行完整对照实验并输出结论

    返回:
        dict: {'records': 各维度指标列表(按维度降序),
               'chosen': 最小充分维度记录或 None,
               'json_path': 结果文件路径}
    """
    setup_logger()
    os.makedirs(args.out_dir, exist_ok=True)
    log.info("维度对照实验启动 | dims=%s | epochs=%d | seed=%d",
             args.dims, args.epochs, args.seed)

    # reverse=True 从大维度到小维度跑：大维度是"能力上限"参照，先拿到基准
    records = [run_one_dim(d, args) for d in sorted(args.dims, reverse=True)]
    chosen = pick_minimal_sufficient(records)

    # ---------- 汇总 ----------
    log.info("%8s %10s %8s %12s", "维度", "PSNR(dB)", "SSIM", "训练耗时(s)")
    for r in records:
        log.info("%8d %10.2f %8.4f %12d",
                 r["latent_dim"], r["psnr"], r["ssim"], r["train_seconds"])
    if chosen:
        log.info("结论: 最小充分维度 = %d（PSNR=%.2fdB >= %.0fdB 且距最优 <= %.0fdB）",
                 chosen["latent_dim"], chosen["psnr"], PSNR_FLOOR, PSNR_MARGIN)
    else:
        log.warning("结论: 无维度达标，需增大维度候选或加大训练轮数")

    json_path = os.path.join(args.out_dir, "latent_experiment.json")
    # ensure_ascii=False 让中文以原文写入 JSON 而不是 \uXXXX 转义
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "epochs": args.epochs, "seed": args.seed, "img_dir": args.img_dir,
            "rule": f"PSNR>={PSNR_FLOOR}dB 且 距最优<={PSNR_MARGIN}dB 的最小维度",
            "records": records,
            "chosen": chosen,
        }, f, ensure_ascii=False, indent=2)
    log.info("实验结果已写入: %s", json_path)

    return {"records": records, "chosen": chosen, "json_path": json_path}


if __name__ == "__main__":
    run_experiment(parse_args())
