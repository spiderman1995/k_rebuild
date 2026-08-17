# -*- coding: utf-8 -*-
"""阶段 13 测试关卡：ViT Patch 遮挡 + 224 数据支持。

通过标准：
    1. 遮挡输出 shape/值域不变且不修改原张量
    2. 默认遮挡恰好为 25%（14×14 网格中的 49 格）
    3. 同种子图案可复现，不同种子图案不同
    4. 非法遮挡率被拒绝
    5. KLineDataset 以 size=224 正常加载
"""

import os
import shutil
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corruptions import FILL_VALUE, MASK_RATIO, mask_patches
from src.dataset import KLineDataset
from src.prepare_224 import downsample_one

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(ROOT, "tests", "fixtures", "kline36")
TMP_DIR = os.path.join(ROOT, "tests", "_corrupt_tmp")


def _batch(n=2, size=224):
    """构造可复现的随机图片 batch，值域 [0,1]。"""
    generator = torch.Generator().manual_seed(0)
    return torch.rand(n, 3, size, size, generator=generator)


def _masked_block_count(image: torch.Tensor, patch: int = 16) -> int:
    """统计单张图片中被完整填为灰色的 patch 数。"""
    grid = image.shape[-1] // patch
    masked = 0
    for row in range(grid):
        for col in range(grid):
            block = image[:,
                          row * patch:(row + 1) * patch,
                          col * patch:(col + 1) * patch]
            if torch.all(block == FILL_VALUE):
                masked += 1
    return masked


def test_shape_range_and_no_inplace():
    """遮挡不改变 shape/值域，也不原地修改输入。"""
    x = _batch()
    backup = x.clone()
    out = mask_patches(x)
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1
    assert torch.equal(x, backup), "mask_patches 不应原地修改输入"


def test_default_mask_ratio_is_25_percent():
    """默认遮挡 196×25%=49 个 Patch16。"""
    x = _batch(n=1)
    out = mask_patches(x)
    expected = int(14 * 14 * MASK_RATIO)
    assert MASK_RATIO == 0.25
    assert _masked_block_count(out[0]) == expected == 49


def test_generator_reproducible():
    """同种子逐像素一致，不同种子产生不同遮挡位置。"""
    x = _batch()
    a = mask_patches(x, generator=torch.Generator().manual_seed(7))
    b = mask_patches(x, generator=torch.Generator().manual_seed(7))
    c = mask_patches(x, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize("ratio", [-0.01, 1.01])
def test_invalid_ratio_rejected(ratio):
    """遮挡率必须位于闭区间 [0,1]。"""
    with pytest.raises(ValueError, match="ratio"):
        mask_patches(_batch(n=1), ratio=ratio)


def test_dataset_loads_224():
    """KLineDataset(size=224) 能加载降采样图。"""
    os.makedirs(TMP_DIR, exist_ok=True)
    try:
        for name in sorted(os.listdir(FIXTURE_DIR))[:3]:
            downsample_one(os.path.join(FIXTURE_DIR, name),
                           os.path.join(TMP_DIR, name))
        dataset = KLineDataset(TMP_DIR, size=224)
        image, _ = dataset[0]
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32 and image.max() <= 1.0
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
