# -*- coding: utf-8 -*-
"""阶段 11a 测试关卡：破坏变换 + 224 数据支持

通过标准（见 开发计划.md 第三期）：
    1. 三种破坏输出 shape 不变、取值仍在 [0,1]、不修改原张量
    2. 遮挡比例正确（mask_patches 75% 的格子被填灰）
    3. 同种子 generator 破坏图案完全一致（验证/测试曲线可比的前提）
    4. 去色后三通道相等，红绿信息被抹掉
    5. KLineDataset 以 size=224 可加载降采样图
"""

import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corruptions import FILL_VALUE, erase_rects, mask_patches, to_grayscale
from src.dataset import KLineDataset
from src.prepare_224 import downsample_one

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(ROOT, "tests", "fixtures", "kline36")
TMP_DIR = os.path.join(ROOT, "tests", "_corrupt_tmp")


def _batch(n=2, size=224):
    """构造随机图片 batch（值域 [0,1]）"""
    # manual_seed 保证测试自身可复现
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, 3, size, size, generator=g)


def test_shapes_range_and_no_inplace():
    """三种破坏：shape 不变、值域 [0,1]、原张量不被修改"""
    x = _batch()
    x_backup = x.clone()
    for fn in (lambda t: erase_rects(t), lambda t: mask_patches(t), to_grayscale):
        out = fn(x)
        assert out.shape == x.shape, "破坏后 shape 变了"
        assert out.min() >= 0 and out.max() <= 1, "破坏后值域越界"
        assert torch.equal(x, x_backup), "破坏函数修改了原张量（应 clone）"


def test_mask_patches_ratio():
    """mask_patches：被填灰的格子数应恰为 75%（14×14 网格中的 147 格）"""
    x = _batch(n=1)
    out = mask_patches(x, patch=16, ratio=0.75)
    # 逐格检查：一个格子内三通道全部等于 FILL_VALUE 即视为被遮
    masked = 0
    for r in range(14):
        for c in range(14):
            block = out[0, :, r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
            if torch.all(block == FILL_VALUE):
                masked += 1
    assert masked == int(14 * 14 * 0.75), f"遮挡格数 {masked} != 147"


def test_generator_reproducible():
    """同种子 generator：两次破坏图案逐像素一致；不同种子不同"""
    x = _batch()
    a = erase_rects(x, generator=torch.Generator().manual_seed(7))
    b = erase_rects(x, generator=torch.Generator().manual_seed(7))
    c = erase_rects(x, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b), "同种子挖块图案不一致"
    assert not torch.equal(a, c), "不同种子挖块图案相同"

    a = mask_patches(x, generator=torch.Generator().manual_seed(7))
    b = mask_patches(x, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b), "同种子遮 patch 图案不一致"


def test_grayscale_weakens_color():
    """去色：三通道相等；红绿差从 0.667 大幅压缩（实测约 0.081，非完全抹除）"""
    x = _batch()
    out = to_grayscale(x)
    assert torch.equal(out[:, 0], out[:, 1]) and torch.equal(out[:, 1], out[:, 2]), (
        "去色后三通道应相等"
    )
    # 数据集配色：红(214,39,40) 与 绿(44,160,44)
    red = torch.tensor([214, 39, 40]).view(1, 3, 1, 1) / 255.0
    green = torch.tensor([44, 160, 44]).view(1, 3, 1, 1) / 255.0
    rgb_diff = (red - green).abs().max().item()          # 原始最大通道差 ≈ 0.667
    gray_diff = abs(to_grayscale(red)[0, 0, 0, 0] - to_grayscale(green)[0, 0, 0, 0])
    # 去色应把红绿差压缩到原来的 1/5 以下（实测 0.081/0.667 ≈ 1/8）
    assert gray_diff < rgb_diff * 0.2, (
        f"红绿灰度差 {gray_diff:.3f} 未显著小于原始通道差 {rgb_diff:.3f}"
    )


def test_dataset_loads_224():
    """KLineDataset(size=224) 能加载降采样图并校验尺寸"""
    os.makedirs(TMP_DIR, exist_ok=True)
    try:
        # 用 prepare_224 的降采样函数从夹具生成 3 张 224 图
        for name in sorted(os.listdir(FIXTURE_DIR))[:3]:
            downsample_one(os.path.join(FIXTURE_DIR, name),
                           os.path.join(TMP_DIR, name))
        ds = KLineDataset(TMP_DIR, size=224)
        img, _ = ds[0]
        assert img.shape == (3, 224, 224), f"224 加载 shape 错误: {img.shape}"
        assert img.dtype == torch.float32 and img.max() <= 1.0
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
