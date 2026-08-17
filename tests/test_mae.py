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

from src.model_mae import (EMBED_DIM, SPATIAL_FEATURE_DIM, MAE,
                           spatial_pool_tokens)


def test_spatial_pool_preserves_four_regions():
    """2×2 空间汇聚应保留四个区域，而不是先压成单个192维均值。"""
    grid = torch.zeros(1, 14, 14, 1)
    grid[:, :7, :7] = 1.0
    grid[:, :7, 7:] = 2.0
    grid[:, 7:, :7] = 3.0
    grid[:, 7:, 7:] = 4.0
    tokens = grid.reshape(1, 14 * 14, 1)
    pooled = spatial_pool_tokens(tokens)
    assert pooled.shape == (1, 4)
    assert torch.equal(pooled, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    full_width = spatial_pool_tokens(torch.rand(2, 196, EMBED_DIM))
    assert full_width.shape == (2, SPATIAL_FEATURE_DIM) == (2, 768)


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


def test_latent_dim_contract():
    """里氏替换契约（12-L）：MAE 与 CAE 同样通过 latent_dim 读取 z 维度"""
    from src.model_cae import CAE
    mae = MAE()
    assert mae.latent_dim == EMBED_DIM, "MAE.latent_dim 应等于 embed_dim"
    # z 的实际维度必须与 latent_dim 属性一致（契约的实质）
    _x_hat, z = mae(torch.rand(1, 3, 224, 224))
    assert z.shape[1] == mae.latent_dim
    cae = CAE(latent_dim=64, input_size=224)
    _x_hat, z = cae(torch.rand(1, 3, 224, 224))
    assert z.shape[1] == cae.latent_dim


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


def test_pretrained_mode():
    """预训练模式（阶段11f）：加载 timm 权重，前向 shape 正确、梯度可达

    需要 timm 已安装且权重已缓存（HF_HOME 指向 D:\\hf_cache）；
    环境不满足时跳过而非失败（203 等新机器上可能尚未下载权重）。
    """
    import pytest
    try:
        import timm  # noqa: F401  只测可导入性
    except ImportError:
        pytest.skip("timm 未安装")
    if os.path.isdir(r"D:\hf_cache"):
        os.environ.setdefault("HF_HOME", r"D:\hf_cache")

    try:
        model = MAE(pretrained=True)
    except Exception as e:  # 权重不可得（无网络/无缓存）时跳过
        pytest.skip(f"预训练权重不可得: {e}")

    x = torch.rand(2, 3, 224, 224)
    x_hat, z = model(x)
    assert x_hat.shape == (2, 3, 224, 224)
    assert z.shape == (2, EMBED_DIM)
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0
    # 参数应分布在 backbone.* 前缀下（微调分组学习率的依据）
    names = [n for n, _ in model.named_parameters()]
    assert any(n.startswith("backbone.") for n in names), "缺少 backbone 前缀参数"
    # 梯度可达（含预训练主干——微调模式主干也要更新）
    ((x_hat - x) ** 2).mean().backward()
    n_with_grad = sum(1 for _, p in model.named_parameters() if p.grad is not None)
    assert n_with_grad == len(names), "存在未接入计算图的参数"


def test_unified_decoder_identical_across_models():
    """统一解码器（阶段11h 核心性质）：同种子下 CAE 与 MAEUnified 的
    解码器初始权重逐位一致；前向/梯度正常

    需要 timm 权重缓存，不可得时跳过（同 test_pretrained_mode）。
    """
    import pytest
    try:
        import timm  # noqa: F401
    except ImportError:
        pytest.skip("timm 未安装")
    if os.path.isdir(r"D:\hf_cache"):
        os.environ.setdefault("HF_HOME", r"D:\hf_cache")

    from src.model_cae import CAE
    from src.model_mae import MAEUnified

    try:
        uni = MAEUnified(latent_dim=512, decoder_init_seed=42,
                         decoder_variant="residual")
    except Exception as e:
        pytest.skip(f"预训练权重不可得: {e}")
    cae = CAE(latent_dim=512, input_size=224, decoder_init_seed=42,
              decoder_variant="residual")

    # 核心断言：两个模型的解码器（fc + conv 全部张量）初始权重逐位相等
    uni_sd = {**dict(uni.decoder_fc.state_dict()),
              **{f"conv.{k}": v for k, v in uni.decoder_conv.state_dict().items()}}
    cae_sd = {**dict(cae.decoder_fc.state_dict()),
              **{f"conv.{k}": v for k, v in cae.decoder_conv.state_dict().items()}}
    assert uni_sd.keys() == cae_sd.keys(), "解码器结构不一致"
    for k in uni_sd:
        assert torch.equal(uni_sd[k], cae_sd[k]), f"解码器权重 {k} 不一致"

    # 前向 shape 与梯度
    x = torch.rand(2, 3, 224, 224)
    x_hat, z = uni(x)
    assert x_hat.shape == (2, 3, 224, 224)
    assert z.shape == (2, 512)
    assert uni.pre_projection_dim == SPATIAL_FEATURE_DIM == 768
    assert uni.pre_projection_dim >= uni.latent_dim, (
        "投影前维度必须不小于 z512，禁止重新引入192→512形式扩维"
    )
    ((x_hat - x) ** 2).mean().backward()
    for name, p in uni.named_parameters():
        assert p.grad is not None, f"参数 {name} 无梯度"

    # 种子 fork 不应污染全局随机流：连续两次同种子构造应完全一致
    d1 = CAE(latent_dim=256, input_size=224, decoder_init_seed=7)
    d2 = CAE(latent_dim=256, input_size=224, decoder_init_seed=7)
    for (k1, v1), (_k2, v2) in zip(d1.decoder_fc.state_dict().items(),
                                   d2.decoder_fc.state_dict().items()):
        assert torch.equal(v1, v2), f"同种子两次构造解码器不一致: {k1}"


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
