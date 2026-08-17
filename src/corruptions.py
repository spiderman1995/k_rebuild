# -*- coding: utf-8 -*-
"""Patch 遮挡变换（流程位置：三组重建对照的 ViT 掩码输入）。

当前实验只保留一种输入破坏：按 16×16 patch 网格随机遮挡 25%。
CAE 与无掩码 ViT 对照均直接输入原图。

遮挡区域填充值为灰色 0.5：与白色背景(1.0)和深色线条(约0.2)
都能区分，模型可以明确识别缺失位置。
"""

import torch

# 破坏区域的统一填充值（灰色）
FILL_VALUE = 0.5
# 当前三组对照协议固定遮挡四分之一 patch。
MASK_RATIO = 0.25


def mask_patches(x: torch.Tensor, patch: int = 16,
                 ratio: float = MASK_RATIO,
                 generator: torch.Generator = None) -> torch.Tensor:
    """按 patch 网格随机遮挡（ViT 掩码重建组，SimMIM 风格）。

    参数:
        x:         图片张量, shape (B, 3, H, W)，H/W 需被 patch 整除
        patch:     网格边长；默认 16，与 ViT-Tiny/16 对齐
        ratio:     遮挡比例；当前实验默认 0.25
        generator: 可选随机器；固定种子可复现同一遮挡图案
    返回:
        破坏后的新张量（不修改原张量），shape 同输入
    """
    if x.shape[-1] % patch or x.shape[-2] % patch:
        raise ValueError(f"图片边长 {x.shape[-2:]} 必须能被 patch={patch} 整除")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio 必须在 [0,1]，收到 {ratio}")

    x = x.clone()
    b, _, h, w = x.shape
    gh, gw = h // patch, w // patch          # 224/16 = 14×14 网格
    n_mask = int(gh * gw * ratio)            # 25% 时为 196*0.25 = 49 格
    for i in range(b):
        # randperm 生成全部格子的随机排列，前 n_mask 个作为遮挡位置。
        idx = torch.randperm(gh * gw, generator=generator)[:n_mask]
        rows, cols = idx // gw, idx % gw
        for row, col in zip(rows.tolist(), cols.tolist()):
            x[i, :,
              row * patch:(row + 1) * patch,
              col * patch:(col + 1) * patch] = FILL_VALUE
    return x
