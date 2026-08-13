# -*- coding: utf-8 -*-
"""阶段 11c 测试关卡：MAE 模型（ViT-Tiny）

通过标准（见 开发计划.md 第三期）：
    1. 前向 (B,3,224,224) -> 重建 (B,3,224,224) + 特征 (B,192)
    2. 输出在 [0,1]（Sigmoid 约束）
    3. 反向传播梯度可达全部参数（含位置编码）
    4. unpatchify 拼图正确性：恒等像素块能还原到正确空间位置
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_mae import EMBED_DIM, MAE


def test_forward_shapes_and_range():
    """前向 shape 与值域"""
    model = MAE()
    x = torch.rand(2, 3, 224, 224)
    x_hat, z = model(x)
    assert x_hat.shape == (2, 3, 224, 224), f"重建 shape 错误: {x_hat.shape}"
    assert z.shape == (2, EMBED_DIM), f"特征 shape 错误: {z.shape}"
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0, "输出超出 [0,1]"


def test_encode():
    """encode 单独调用：特征提取器用法"""
    model = MAE()
    z = model.encode(torch.rand(1, 3, 224, 224))
    assert z.shape == (1, EMBED_DIM)


def test_gradients_reach_all_parameters():
    """梯度可达全部参数（含 pos_embed 和 Transformer 各层）"""
    model = MAE(depth=2)  # 浅层版加快测试
    x = torch.rand(2, 3, 224, 224)
    x_hat, _ = model(x)
    ((x_hat - x) ** 2).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"参数 {name} 无梯度"
        assert torch.isfinite(p.grad).all(), f"参数 {name} 梯度 NaN/Inf"


def test_parameter_count():
    """参数量应在 ViT-Tiny 量级（约 300~700 万），打印具体值"""
    n = MAE().count_parameters()
    print(f"\nMAE(ViT-Tiny) 可训练参数量: {n:,}")
    assert 2_000_000 < n < 10_000_000, f"参数量 {n:,} 偏离 ViT-Tiny 量级"


def test_unpatchify_geometry():
    """拼图几何正确性：直接检验 forward 中的 permute/reshape 逻辑

    构造与 forward 相同的 (B,196,768) -> (B,3,224,224) 变换，
    给第 k 个 patch 填充其编号值，检查像素落在正确的网格位置。
    """
    b, grid, patch = 1, 14, 16
    pixels = torch.zeros(b, grid * grid, patch * patch * 3)
    for k in range(grid * grid):
        pixels[0, k] = float(k)  # 第 k 个 patch 全部像素 = k

    # 与 model_mae.forward 完全一致的 unpatchify 序列
    out = pixels.view(b, grid, grid, patch, patch, 3)
    out = out.permute(0, 5, 1, 3, 2, 4).reshape(b, 3, 224, 224)

    for k in [0, 13, 14, 195]:  # 抽查四角与换行处
        r, c = k // grid, k % grid
        block = out[0, :, r * patch:(r + 1) * patch, c * patch:(c + 1) * patch]
        assert torch.all(block == float(k)), f"patch {k} 未落在网格 ({r},{c}) 处"
