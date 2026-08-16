# -*- coding: utf-8 -*-
"""阶段 2 测试关卡：CAE 模型

通过标准（见 开发计划.md）：
    1. 前向 shape 正确：(B,3,448,448) -> z (B,latent_dim) -> (B,3,448,448)
    2. encode / decode 可单独调用
    3. 反向传播后梯度可达全部参数
    4. latent_dim 可配置
    5. 输出取值在 (0,1)（Sigmoid 约束）
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_cae import CAE


def test_forward_shapes():
    """前向传播：输入 (2,3,448,448)，重建输出同形状，z 为 (2,256)"""
    model = CAE(latent_dim=256)
    x = torch.rand(2, 3, 448, 448)
    x_hat, z = model(x)
    assert x_hat.shape == (2, 3, 448, 448), f"重建 shape 错误: {x_hat.shape}"
    assert z.shape == (2, 256), f"特征 shape 错误: {z.shape}"


def test_encode_decode_separately():
    """encode/decode 单独调用：特征提取器用法必须可用"""
    model = CAE(latent_dim=128)
    x = torch.rand(1, 3, 448, 448)
    z = model.encode(x)
    assert z.shape == (1, 128)
    x_hat = model.decode(z)
    assert x_hat.shape == (1, 3, 448, 448)


def test_latent_dim_configurable():
    """latent_dim 可配置：64/512 两个极端都应正常工作"""
    for d in (64, 512):
        model = CAE(latent_dim=d)
        z = model.encode(torch.rand(1, 3, 448, 448))
        assert z.shape == (1, d), f"latent_dim={d} 时特征 shape 错误: {z.shape}"


def test_output_range():
    """输出经 Sigmoid，必须落在 [0,1] 内"""
    model = CAE(latent_dim=64)
    x_hat, _ = model(torch.rand(2, 3, 448, 448))
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0, "输出超出 [0,1]"


def test_gradients_reach_all_parameters():
    """反向传播后所有可训练参数都应有梯度（无断路的计算图）"""
    model = CAE(latent_dim=64)
    x = torch.rand(2, 3, 448, 448)
    x_hat, _ = model(x)
    loss = ((x_hat - x) ** 2).mean()  # 简单 MSE 触发反向传播
    loss.backward()
    # named_parameters 返回 (参数名, 参数张量) 对，便于失败时报出具体层名
    for name, p in model.named_parameters():
        assert p.grad is not None, f"参数 {name} 没有梯度（计算图断路）"
        assert torch.isfinite(p.grad).all(), f"参数 {name} 梯度出现 NaN/Inf"


def test_parameter_count_reasonable():
    """参数量检查：latent=256 时应在千万级（约 1~5 千万），并打印具体值"""
    model = CAE(latent_dim=256)
    n = model.count_parameters()
    print(f"\nCAE(latent_dim=256) 可训练参数量: {n:,}")
    assert 5_000_000 < n < 60_000_000, f"参数量 {n:,} 超出预期范围"


def test_input_size_224():
    """224 输入模式（阶段11b）：前向/反向 shape 正确，梯度可达"""
    model = CAE(latent_dim=256, input_size=224)
    x = torch.rand(2, 3, 224, 224)
    x_hat, z = model(x)
    assert x_hat.shape == (2, 3, 224, 224), f"224 重建 shape 错误: {x_hat.shape}"
    assert z.shape == (2, 256)
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0
    ((x_hat - x) ** 2).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"224 模式参数 {name} 无梯度"


def test_input_size_default_unchanged():
    """回归：默认 input_size 仍为 448，旧调用方式行为不变"""
    model = CAE(latent_dim=64)
    assert model.input_size == 448
    x_hat, _ = model(torch.rand(1, 3, 448, 448))
    assert x_hat.shape == (1, 3, 448, 448)


def test_input_size_invalid():
    """非法 input_size 应报错而非静默构造出错误模型"""
    import pytest
    with pytest.raises(ValueError):
        CAE(latent_dim=64, input_size=300)


def test_residual_decoder_512_contract():
    """新统一解码器必须只接收真实 512 维 z，并保持前向/反向完整。"""
    model = CAE(latent_dim=512, input_size=224,
                decoder_init_seed=42, decoder_variant="residual")
    x = torch.rand(1, 3, 224, 224)
    x_hat, z = model(x)
    assert z.shape == (1, 512)
    assert x_hat.shape == x.shape
    ((x_hat - x) ** 2).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"残差解码器参数 {name} 无梯度"


def test_decoder_variant_validation():
    """未知解码器版本必须显式报错，不能静默退回其他结构。"""
    import pytest
    with pytest.raises(ValueError, match="decoder variant"):
        CAE(latent_dim=512, input_size=224, decoder_variant="unknown")
