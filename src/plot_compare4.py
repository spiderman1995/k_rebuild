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


# 三个数据集切分对应的中文标题
SPLIT_TITLES = {"train": "训练集", "val": "验证集", "test": "测试集"}


def plot_curves(history: dict, out_path: str):
    """绘制损失曲线：训练/验证/测试各一张子图，每张子图放所有方法的曲线

    布局按用户要求（2026-08-13）：同一切分下所有方法同框，方法间直接对比。
    各方法曲线长度可以不同（合并了不同 epoch 数的运行结果），
    横轴按各自历史长度画，短曲线自然止于自己的最后一轮。

    参数:
        history:  {方法名: {'label':.., 'train': [...], 'val': [...], 'test': [...]}}
        out_path: 输出 PNG 路径
    """
    # 三张子图横排：训练 | 验证 | 测试
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    for ax, split in zip(axes, ("train", "val", "test")):
        for _m, h in history.items():
            # 曲线标注方法名与其轮数（不同方法轮数可不同）
            ax.plot(range(1, len(h[split]) + 1), h[split],
                    label=f"{h['label']} ({len(h[split])}ep)")
        ax.set_title(f"{SPLIT_TITLES[split]}损失对比（纯 MSE，重建原图，方法间可直接比较）")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE 损失")
        ax.set_yscale("log")  # 损失跨数量级下降，对数轴更易读
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

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
    plot_curves(data["history"], args.out)
    print(f"曲线图已保存: {args.out}")


if __name__ == "__main__":
    main()
