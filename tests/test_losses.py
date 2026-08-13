# -*- coding: utf-8 -*-
"""阶段 3 测试关卡：损失函数

通过标准（见 开发计划.md）：
    1. 相同图片 -> loss≈0 且 SSIM≈1
    2. 加噪图片 -> loss 显著大于 0
    3. 损失可反向传播（梯度非空且有限）
    4. alpha=1/0 分别退化为纯 MSE / 纯 SSIM
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import KLineDataset
from src.losses import CombinedLoss, psnr_metric, ssim_metric

IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "分钟k线图"
)


def _load_batch(n: int = 2) -> torch.Tensor:
    """取真实数据前 n 张组成 batch，shape (n,3,448,448)"""
    ds = KLineDataset(IMG_DIR)
    # torch.stack 把 n 个 (3,448,448) 张量沿新建的第 0 维堆叠成 (n,3,448,448)
    return torch.stack([ds[i][0] for i in range(n)])


def test_identical_images_zero_loss():
    """相同图片：loss≈0，SSIM≈1"""
    x = _load_batch()
    loss = CombinedLoss(alpha=0.5)(x, x)
    assert loss.item() < 1e-6, f"相同图片 loss 应≈0，实际 {loss.item()}"
    assert abs(ssim_metric(x, x) - 1.0) < 1e-6, "相同图片 SSIM 应≈1"


def test_noisy_images_positive_loss():
    """加噪图片：loss 显著大于 0，且噪声越大 loss 越大"""
    x = _load_batch()
    # clamp 把加噪后的值截回 [0,1]，保持与真实输入同值域
    x_small_noise = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    x_large_noise = (x + 0.5 * torch.randn_like(x)).clamp(0, 1)
    crit = CombinedLoss(alpha=0.5)
    loss_small = crit(x_small_noise, x).item()
    loss_large = crit(x_large_noise, x).item()
    assert loss_small > 1e-3, f"小噪声 loss 应显著>0，实际 {loss_small}"
    assert loss_large > loss_small, "大噪声 loss 应大于小噪声 loss"


def test_loss_backward():
    """损失反向传播：梯度存在且全部有限"""
    x = _load_batch()
    x_hat = x.clone().requires_grad_(True)  # 模拟模型输出（可求导叶子节点）
    loss = CombinedLoss(alpha=0.5)(x_hat, x + 0.05)
    loss.backward()
    assert x_hat.grad is not None, "反向传播后梯度为空"
    assert torch.isfinite(x_hat.grad).all(), "梯度出现 NaN/Inf"


def test_alpha_degenerate_cases():
    """alpha=1 应等于纯 MSE；alpha=0 应等于纯 1-SSIM"""
    x = _load_batch()
    x_noisy = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)

    loss_mse_only = CombinedLoss(alpha=1.0)(x_noisy, x).item()
    ref_mse = torch.nn.functional.mse_loss(x_noisy, x).item()
    assert abs(loss_mse_only - ref_mse) < 1e-6, "alpha=1 未退化为纯 MSE"

    loss_ssim_only = CombinedLoss(alpha=0.0)(x_noisy, x).item()
    ref_ssim = 1.0 - ssim_metric(x_noisy, x)
    assert abs(loss_ssim_only - ref_ssim) < 1e-4, "alpha=0 未退化为纯 SSIM 损失"


def test_metrics_sanity():
    """指标合法性：SSIM∈[0,1]；PSNR 相同图片为 inf、加噪图片为有限正数"""
    x = _load_batch()
    x_noisy = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    s = ssim_metric(x_noisy, x)
    assert 0.0 <= s <= 1.0, f"SSIM 超出 [0,1]: {s}"
    assert psnr_metric(x, x) == float("inf"), "相同图片 PSNR 应为 inf"
    p = psnr_metric(x_noisy, x)
    assert 0 < p < 100, f"加噪 PSNR 不合理: {p}"
