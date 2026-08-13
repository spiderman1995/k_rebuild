# -*- coding: utf-8 -*-
"""损失函数模块（流程位置：训练时度量"原图与重建图相似度"）

实现 README.md 第五节的组合损失：
    L = alpha * MSE + (1 - alpha) * (1 - SSIM)

- MSE  逐像素比较，保证像素级精度
- SSIM 结构相似性（取值 [0,1]，1 为完全相同），直接对应
  "两张图相似度"的项目目标；1-SSIM 使其变成越小越好的损失
- alpha=1 退化为纯 MSE，alpha=0 退化为纯 SSIM 损失
"""

import torch
import torch.nn as nn
from pytorch_msssim import SSIM


class CombinedLoss(nn.Module):
    """MSE + SSIM 组合重建损失

    参数:
        alpha: MSE 的权重，取值 [0,1]，默认 0.5（README 推荐起点）。
               SSIM 项权重为 1 - alpha。
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha 必须在 [0,1]，收到 {alpha}")
        self.alpha = alpha
        self.mse = nn.MSELoss()
        # data_range=1.0：输入像素范围是 [0,1]（dataset.py 已归一化）
        # channel=3：RGB 三通道
        self.ssim = SSIM(data_range=1.0, channel=3)

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """计算组合损失

        参数:
            x_hat: 重建图片, shape (B, 3, 448, 448), 取值 [0,1]
            x:     原始图片, shape (B, 3, 448, 448), 取值 [0,1]
        返回:
            标量损失张量（越小表示重建越接近原图）
        """
        mse_term = self.mse(x_hat, x)
        # SSIM 越大越相似，取 1-SSIM 转为"越小越好"的损失方向
        ssim_term = 1.0 - self.ssim(x_hat, x)
        return self.alpha * mse_term + (1.0 - self.alpha) * ssim_term


def ssim_metric(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    """计算两批图片的 SSIM 值（评估指标用，非损失）

    参数:
        x_hat, x: shape (B, 3, H, W), 取值 [0,1]
    返回:
        SSIM 标量 float，范围 [0,1]，越接近 1 越相似
    """
    # no_grad：纯评估不需要计算图，省显存
    with torch.no_grad():
        return SSIM(data_range=1.0, channel=3)(x_hat, x).item()


def psnr_metric(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    """计算峰值信噪比 PSNR（评估指标用）

    公式: PSNR = 10 * log10(MAX^2 / MSE)，像素范围 [0,1] 时 MAX=1。
    典型参考：>30dB 重建质量较好，>40dB 非常好。

    参数:
        x_hat, x: shape (B, 3, H, W), 取值 [0,1]
    返回:
        PSNR 标量 float，单位 dB（两图完全相同时为无穷大 inf）
    """
    with torch.no_grad():
        mse = torch.mean((x_hat - x) ** 2).item()
        if mse == 0:
            return float("inf")
        # 10 * log10(1 / mse)：MAX=1 时公式的化简形式
        return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()
