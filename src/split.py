# -*- coding: utf-8 -*-
"""数据划分工具（流程位置：正式训练前的数据准备，为远程 120 万张做准备）

功能：把文件清单按 7:2:1 随机（但可复现）切分为训练/验证/测试集：
    - 训练集：拟合模型
    - 验证集：调超参、早停
    - 测试集：只在最终验收时用一次

本地 36 张仅用于验证切分机制；正式使用场景是远程服务器的大规模数据。
"""

import os
import random

from src.logger import get_logger

log = get_logger("split")


def split_files(files, ratios=(0.7, 0.2, 0.1), seed: int = 42):
    """把文件清单随机切分为 train/val/test 三份（可复现）

    参数:
        files:  文件名列表
        ratios: (训练, 验证, 测试) 比例，和必须为 1
        seed:   随机种子。同样的 files+seed 永远得到同样的切分，
                保证实验可复现、多次运行不会把验证集泄漏进训练集
    返回:
        (train, val, test) 三个文件名列表，互斥且并集为全量
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"比例之和必须为 1，收到 {ratios}")
    if not files:
        raise ValueError("文件清单为空")

    shuffled = sorted(files)  # 先排序消除来源顺序差异，保证跨平台可复现
    # random.Random(seed) 创建独立随机器，不污染全局随机状态
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    # 边界下标：训练集止于 n_train，验证集止于 n_train+n_val，其余归测试集。
    # 测试集用"剩余全部"而不是按比例取整，保证三份加起来不丢样本
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    log.info("数据切分完成: 总数 %d -> 训练 %d / 验证 %d / 测试 %d (seed=%d)",
             n, len(train), len(val), len(test), seed)
    return train, val, test


def split_dir(img_dir: str, ratios=(0.7, 0.2, 0.1), seed: int = 42):
    """扫描图片文件夹并切分（split_files 的便捷封装）

    参数:
        img_dir: 图片文件夹路径
        ratios/seed: 同 split_files
    返回:
        (train, val, test) 三个文件名列表
    """
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(".png")]
    return split_files(files, ratios=ratios, seed=seed)
