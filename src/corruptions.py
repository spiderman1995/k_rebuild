# -*- coding: utf-8 -*-
"""输入破坏变换模块（流程位置：四组"破坏重建"对照实验的破坏环节）

对应 开发计划.md 阶段 11 的三种破坏方式（作用于 [0,1] 的图片张量）：
    erase_rects   —— 组② 遮挡修复：随机挖掉若干矩形块
    mask_patches  —— 组③ MAE：按 patch 网格随机遮 75%
    to_grayscale  —— 组④ 去色重彩：抹掉红绿颜色信息

填充值统一用灰色 0.5：与白色背景(1.0)和深色线条(≈0.2)都能区分，
模型能明确识别"哪里是被破坏的区域"。

所有随机破坏都接受 torch.Generator：
    - 训练时传 None（每个 batch 破坏图案不同，等于数据增广）
    - 验证/测试时传固定种子的 generator（每轮破坏图案一致，loss 曲线平滑可比）
"""

import torch

# 破坏区域的统一填充值（灰色）
FILL_VALUE = 0.5


def erase_rects(x: torch.Tensor, n_rects: int = 8, rect_size: int = 32,
                generator: torch.Generator = None) -> torch.Tensor:
    """随机挖掉若干矩形块（组② 遮挡修复的破坏方式）

    参数:
        x:         图片张量, shape (B, 3, H, W), 取值 [0,1]
        n_rects:   每张图挖掉的矩形数量
        rect_size: 矩形边长（像素）
        generator: 可选随机器；传入固定种子可复现同一组挖块位置
    返回:
        破坏后的新张量（不修改原张量）, shape 同输入
    """
    x = x.clone()  # clone 保证不改动调用方的原图
    b, _, h, w = x.shape
    # 为每张图、每个矩形生成随机左上角坐标, shape (B, n_rects)
    ys = torch.randint(0, h - rect_size + 1, (b, n_rects), generator=generator)
    xs = torch.randint(0, w - rect_size + 1, (b, n_rects), generator=generator)
    for i in range(b):
        for j in range(n_rects):
            y0, x0 = ys[i, j].item(), xs[i, j].item()
            # 切片赋值：三通道同时填充灰色
            x[i, :, y0:y0 + rect_size, x0:x0 + rect_size] = FILL_VALUE
    return x


def mask_patches(x: torch.Tensor, patch: int = 16, ratio: float = 0.75,
                 generator: torch.Generator = None) -> torch.Tensor:
    """按 patch 网格随机遮挡（组③ MAE 的破坏方式，SimMIM 风格）

    把图切成 patch×patch 的网格，随机选 ratio 比例的格子填充灰色。

    参数:
        x:         图片张量, shape (B, 3, H, W)，H/W 需被 patch 整除
        patch:     网格边长，16 对应 ViT-B/16 的 patch 大小
        ratio:     遮挡比例，MAE 论文的经验值为 0.75
        generator: 可选随机器（用途同 erase_rects）
    返回:
        破坏后的新张量, shape 同输入
    """
    if x.shape[-1] % patch or x.shape[-2] % patch:
        raise ValueError(f"图片边长 {x.shape[-2:]} 必须能被 patch={patch} 整除")
    x = x.clone()
    b, _, h, w = x.shape
    gh, gw = h // patch, w // patch          # 网格行列数（224/16 = 14×14）
    n_mask = int(gh * gw * ratio)            # 要遮住的格子数
    for i in range(b):
        # randperm 生成 0..gh*gw-1 的随机排列，取前 n_mask 个作为被遮格子
        idx = torch.randperm(gh * gw, generator=generator)[:n_mask]
        rows, cols = idx // gw, idx % gw     # 一维编号还原为网格行列坐标
        for r, c in zip(rows.tolist(), cols.tolist()):
            x[i, :, r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = FILL_VALUE
    return x


def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """去色（组④ 去色重彩的破坏方式）：RGB → 亮度灰度，复制回三通道

    用 ITU-R BT.601 亮度公式 Y = 0.299R + 0.587G + 0.114B。
    实测：红(214,39,40)与绿(44,160,44)的通道差从 0.667 压缩到约 0.081
    ——颜色信息被大幅削弱但**非完全抹除**（灰度图中红绿仍有微弱亮度差，
    模型可部分借助该线索），解读组④结果时需知道这一点。

    参数:
        x: 图片张量, shape (B, 3, H, W)
    返回:
        灰度图复制成三通道的新张量, shape (B, 3, H, W)
        （保持三通道是为了四组模型输入通道数一致，控制变量）
    """
    # keepdim=True 保留通道维: (B,3,H,W) -> (B,1,H,W)
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    # repeat 沿通道维复制 3 份: (B,1,H,W) -> (B,3,H,W)
    return gray.repeat(1, 3, 1, 1)
