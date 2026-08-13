# -*- coding: utf-8 -*-
"""阶段 1 测试关卡：数据加载模块

通过标准（见 开发计划.md）：
    1. 36 张图片全部加载为 (3, 448, 448) float32，取值 [0, 1]
    2. Alpha 通道已去除（输出只有 3 通道）
    3. 像素颜色与 PIL 直接读取的原图一致（抽查验证）
    4. DataLoader 可正常按 batch 迭代
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

# 把项目根目录加入模块搜索路径，使 `from src.dataset import ...` 可用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import KLineDataset

# 项目根目录下的图片文件夹
IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "分钟k线图"
)


def test_dataset_length():
    """样本数应等于文件夹中的 PNG 数量（36 张）"""
    ds = KLineDataset(IMG_DIR)
    assert len(ds) == 36, f"预期 36 个样本，实际 {len(ds)}"


def test_all_images_shape_dtype_range():
    """全部 36 张：shape (3,448,448)、float32、取值 [0,1]"""
    ds = KLineDataset(IMG_DIR)
    for i in range(len(ds)):
        img, name = ds[i]
        assert img.shape == (3, 448, 448), f"{name} shape 错误: {img.shape}"
        assert img.dtype == torch.float32, f"{name} dtype 错误: {img.dtype}"
        assert img.min() >= 0.0 and img.max() <= 1.0, (
            f"{name} 取值超出 [0,1]: [{img.min()}, {img.max()}]"
        )


def test_pixel_values_match_original():
    """抽查第一张图：张量像素值应与 PIL 直接读取的 RGB 值一致"""
    ds = KLineDataset(IMG_DIR)
    img, name = ds[0]

    # 用 PIL 独立读取同一张图作为对照
    ref = np.asarray(
        Image.open(os.path.join(IMG_DIR, name)).convert("RGB"), dtype=np.float32
    ) / 255.0

    # 张量是 CHW，对照数组是 HWC，用 permute 转回 HWC 再比
    img_hwc = img.permute(1, 2, 0).numpy()
    assert np.allclose(img_hwc, ref), f"{name} 像素值与原图不一致"


def test_invert_option():
    """反色开关：invert=True 时应满足 img_inv == 1 - img"""
    ds_raw = KLineDataset(IMG_DIR, invert=False)
    ds_inv = KLineDataset(IMG_DIR, invert=True)
    img_raw, _ = ds_raw[0]
    img_inv, _ = ds_inv[0]
    assert torch.allclose(img_inv, 1.0 - img_raw), "反色结果不等于 1 - 原图"


def test_dataloader_batching():
    """DataLoader 按 batch=8 迭代：batch 形状 (8,3,448,448)，共 36 张无遗漏"""
    ds = KLineDataset(IMG_DIR)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    total = 0
    for imgs, names in loader:
        # 最后一个 batch 可能不足 8 张，其余应为 8
        assert imgs.shape[1:] == (3, 448, 448)
        assert imgs.shape[0] == len(names)
        total += imgs.shape[0]
    assert total == 36, f"DataLoader 迭代总数 {total} != 36"
