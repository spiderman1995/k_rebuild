# -*- coding: utf-8 -*-
"""四组对照实验的 loss 曲线绘图（独立进程运行，流程位置：compare4 的出图环节）

为什么单独成文件：本机 torch 自带的 OpenMP 运行时与 anaconda MKL 的
OpenMP 运行时冲突（OMP Error #15），torch 和 matplotlib 在同一进程里
混用会直接崩溃。因此本模块**不 import torch**，由 experiment_compare4.py
用子进程调用，画图进程里只有 matplotlib，冲突不会发生。

用法（一般由 experiment_compare4 自动调用，也可手动重画）：
    python src/plot_compare4.py --json results/compare4/compare4.json ^
           --out results/compare4/loss_curves.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # 无界面后端：只存图不弹窗
import matplotlib.pyplot as plt

# 中文字体：Windows 用微软雅黑，避免图上中文变方框
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示


def plot_curves(history: dict, epochs: int, out_path: str):
    """绘制损失曲线：上排每组三条(训练/验证/测试)，下排四组验证集对比

    参数:
        history:  {方法名: {'label':.., 'train': [...], 'val': [...], 'test': [...]}}
        epochs:   横轴长度（epoch 数）
        out_path: 输出 PNG 路径
    """
    n = max(len(history), 1)
    # constrained_layout 自动排版；GridSpec(2, n) 上排 n 格、下排跨全列
    fig = plt.figure(figsize=(4.2 * n, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, n)
    xs = range(1, epochs + 1)

    for i, (_m, h) in enumerate(history.items()):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(xs, h["train"], label="训练", color="#1f77b4")
        ax.plot(xs, h["val"], label="验证", color="#ff7f0e")
        ax.plot(xs, h["test"], label="测试", color="#2ca02c")
        ax.set_title(h["label"])
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE 损失")
        ax.set_yscale("log")  # 损失跨数量级下降，对数轴更易读
        ax.grid(alpha=0.3)
        ax.legend()

    # 下排：四组验证集损失同图横向对比
    ax_all = fig.add_subplot(gs[1, :])
    for _m, h in history.items():
        ax_all.plot(xs, h["val"], label=h["label"])
    ax_all.set_title("四组方案验证集损失对比（统一目标：重建原图，纯 MSE，可直接比较）")
    ax_all.set_xlabel("epoch")
    ax_all.set_ylabel("验证 MSE 损失")
    ax_all.set_yscale("log")
    ax_all.grid(alpha=0.3)
    ax_all.legend()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    """命令行入口：从 compare4.json 读取历史并出图"""
    parser = argparse.ArgumentParser(description="compare4 loss 曲线绘图")
    parser.add_argument("--json", required=True, help="compare4.json 路径")
    parser.add_argument("--out", required=True, help="输出 PNG 路径")
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    plot_curves(data["history"], data["epochs"], args.out)
    print(f"曲线图已保存: {args.out}")


if __name__ == "__main__":
    main()
