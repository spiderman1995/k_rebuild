# -*- coding: utf-8 -*-
"""四组"破坏重建"对照实验（流程位置：阶段 11 主实验脚本）

实验协议（见 开发计划.md 阶段 11，由用户确定）：
    - 数据：只用 2×2 均值降采样的 224×224 图（pic_to_224x224/）
    - 划分：7:1:2 训练/验证/测试（seed 固定），四组共用同一划分
    - 每组 20 epoch，纯 MSE 损失；逐 epoch 记录训练/验证/测试三条 loss
    - 四组目标统一为"从(被破坏的)输入重建原图"，损失可横向比较：
        cae     组① 输入=原图（不破坏，压缩重建基线，CAE-224）
        inpaint 组② 输入=随机挖矩形块（CAE-224）
        mae     组③ 输入=按16×16网格遮75%（ViT-Tiny 从零）
        color   组④ 输入=灰度化（CAE-224）
    - 验证/测试的破坏图案用固定种子，保证逐 epoch 曲线平滑可比

产出：
    results/compare4/compare4.json     全部损失历史与配置
    results/compare4/loss_curves.png   每组三条曲线 + 四组验证集对比

用法（1 万张就位后，在项目根目录）：
    python src/experiment_compare4.py --img-dir pic_to_224x224 --epochs 20
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corruptions import erase_rects, mask_patches, to_grayscale
from src.dataset import KLineDataset
from src.logger import get_logger, setup_logger
from src.model_cae import CAE
from src.model_mae import MAE
from src.split import split_files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = get_logger("compare4")

# 四组方法定义：名称 -> (中文标签, 破坏函数)；破坏函数签名 f(x, generator)
METHODS = {
    "cae": ("① CAE压缩重建", lambda x, g: x),  # 不破坏
    "inpaint": ("② 遮挡修复", lambda x, g: erase_rects(x, generator=g)),
    "mae": ("③ MAE(ViT)", lambda x, g: mask_patches(x, generator=g)),
    "color": ("④ 去色重彩", lambda x, g: to_grayscale(x)),
}
# 验证/测试破坏图案的固定种子（与训练随机破坏区分开）
EVAL_CORRUPT_SEED = 12345


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="四组破坏重建对照实验")
    parser.add_argument("--img-dir", default=os.path.join(ROOT, "pic_to_224x224"),
                        help="224×224 图片目录")
    parser.add_argument("--epochs", type=int, default=20, help="每组训练轮数")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument("--latent-dim", type=int, default=256, help="CAE 特征维度")
    parser.add_argument("--seed", type=int, default=42, help="划分与初始化种子")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results", "compare4"))
    parser.add_argument("--ckpt-dir", default=os.path.join(ROOT, "checkpoints", "compare4"))
    parser.add_argument("--methods", default="cae,inpaint,mae,color",
                        help="逗号分隔的方法子集（调试用）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(method: str, latent_dim: int) -> torch.nn.Module:
    """按方法名构造模型：mae 用 ViT-Tiny，其余用 CAE-224（控制变量）"""
    if method == "mae":
        return MAE()
    return CAE(latent_dim=latent_dim, input_size=224)


def run_epoch(model, loader, corrupt_fn, device, optimizer=None,
              eval_seed: int = None) -> float:
    """跑一遍数据集，返回平均 MSE 损失

    参数:
        model:      当前模型
        loader:     DataLoader
        corrupt_fn: 破坏函数 f(x, generator)
        device:     计算设备
        optimizer:  传入则为训练模式（反向传播更新），否则纯评估
        eval_seed:  评估时的破坏种子；训练时传 None（每 batch 随机破坏）
    返回:
        整个数据集的逐样本平均损失
    """
    training = optimizer is not None
    model.train() if training else model.eval()
    # 评估用固定种子 generator：每个 epoch 的破坏图案完全一致，曲线平滑可比
    gen = None
    if eval_seed is not None:
        gen = torch.Generator().manual_seed(eval_seed)

    total, n = 0.0, 0
    # torch.set_grad_enabled 按训练/评估开关梯度计算（评估省显存）
    with torch.set_grad_enabled(training):
        for imgs, _names in loader:
            corrupted = corrupt_fn(imgs, gen)          # 在 CPU 上破坏后再上卡
            imgs, corrupted = imgs.to(device), corrupted.to(device)
            x_hat, _z = model(corrupted)
            loss = torch.nn.functional.mse_loss(x_hat, imgs)  # 目标恒为原图
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * imgs.size(0)
            n += imgs.size(0)
    return total / n


def plot_in_subprocess(json_path: str, plot_path: str):
    """在独立子进程中出图（不 import torch，绕开 OpenMP 运行时冲突）

    本机 torch 与 anaconda MKL 各带一份 OpenMP 运行时（OMP Error #15），
    同进程混用 torch 和 matplotlib 会直接崩溃；绘图逻辑在
    src/plot_compare4.py，其进程内只有 matplotlib，不会触发冲突。

    参数:
        json_path: 已写好的 compare4.json 路径
        plot_path: 输出 PNG 路径
    """
    script = os.path.join(ROOT, "src", "plot_compare4.py")
    # sys.executable 是当前 Python 解释器路径，保证子进程用同一环境
    result = subprocess.run(
        [sys.executable, script, "--json", json_path, "--out", plot_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # 子进程失败不应吞掉：把 stderr 带进异常，实验数据(JSON)已保存不受影响
        raise RuntimeError(f"绘图子进程失败:\n{result.stderr}")


def run_compare(args) -> dict:
    """执行完整四组对照实验

    返回:
        dict: {'history': 各方法损失历史, 'json_path': .., 'plot_path': ..}
    """
    setup_logger()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---------- 数据：7:1:2 划分，四组共用 ----------
    files = sorted(f for f in os.listdir(args.img_dir) if f.endswith(".png"))
    train_f, val_f, test_f = split_files(files, ratios=(0.7, 0.1, 0.2), seed=args.seed)
    make_loader = lambda fs, shuffle: DataLoader(  # noqa: E731 简短工厂，仅本函数用
        KLineDataset(args.img_dir, files=fs, size=224),
        batch_size=args.batch_size, shuffle=shuffle)
    train_loader = make_loader(train_f, True)
    val_loader = make_loader(val_f, False)
    test_loader = make_loader(test_f, False)
    log.info("对照实验启动 | 数据 %s | 划分 %d/%d/%d | epochs=%d | 设备 %s",
             args.img_dir, len(train_f), len(val_f), len(test_f),
             args.epochs, device)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    history = {}
    for method in methods:
        label, corrupt_fn = METHODS[method]
        torch.manual_seed(args.seed)  # 每组同种子初始化，控制变量
        model = build_model(method, args.latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        log.info("[%s] 开始训练 | 参数量 %s", label, f"{model.count_parameters():,}")

        h = {"label": label, "train": [], "val": [], "test": []}
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            tr = run_epoch(model, train_loader, corrupt_fn, device, optimizer)
            va = run_epoch(model, val_loader, corrupt_fn, device,
                           eval_seed=EVAL_CORRUPT_SEED)
            te = run_epoch(model, test_loader, corrupt_fn, device,
                           eval_seed=EVAL_CORRUPT_SEED + 1)
            h["train"].append(tr)
            h["val"].append(va)
            h["test"].append(te)
            log.info("[%s] epoch %2d/%d | 训练 %.5f | 验证 %.5f | 测试 %.5f | %ds",
                     label, ep, args.epochs, tr, va, te, time.time() - t0)
        history[method] = h
        # 每组各存一个 checkpoint，便于后续复用编码器提特征
        torch.save({"model_state": model.state_dict(), "method": method,
                    "latent_dim": args.latent_dim, "epochs": args.epochs},
                   os.path.join(args.ckpt_dir, f"{method}.pth"))

    # ---------- 产出：先写 JSON（数据落盘），再子进程出图 ----------
    json_path = os.path.join(args.out_dir, "compare4.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "protocol": "仅用224降采样图; 7:1:2划分; 每组20epoch; 统一MSE重建原图",
            "epochs": args.epochs, "seed": args.seed,
            "split": {"train": len(train_f), "val": len(val_f), "test": len(test_f)},
            "history": history,
        }, f, ensure_ascii=False, indent=2)
    plot_path = os.path.join(args.out_dir, "loss_curves.png")
    plot_in_subprocess(json_path, plot_path)
    log.info("实验完成 | 曲线图: %s | 数据: %s", plot_path, json_path)
    return {"history": history, "json_path": json_path, "plot_path": plot_path}


if __name__ == "__main__":
    run_compare(parse_args())
