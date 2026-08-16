# -*- coding: utf-8 -*-
"""卷积自编码器 CAE 模型定义（流程位置：核心模型，主方案 B）

结构（见 README.md 第四节）：
    编码器: 6 层 stride=2 卷积逐步下采样，把 (3,448,448) 压缩为特征向量 z
        3x448x448 -> 32x224x224 -> 64x112x112 -> 128x56x56
                  -> 256x28x28  -> 512x14x14  -> 512x7x7 -> 展平全连接 -> z
    解码器: 全连接恢复 512x7x7，再 6 层 [上采样+卷积] 对称还原 (3,448,448)
        上采样用 Upsample(nearest)+Conv 而非 ConvTranspose，避免棋盘伪影

特征向量 z 即项目要提取的图片特征；训练完成后可单独调用 encode() 做特征提取。
"""

import torch
import torch.nn as nn

# 编码器最深层的空间规格：输入边长经 N 次 stride=2 下采样到 7
# （448 → 6 次，224 → 5 次，均收敛到 7×7）
BOTTLENECK_CHANNELS = 512
BOTTLENECK_SIZE = 7
# 展平后的维度 512*7*7 = 25088，是编码器全连接层的输入
FLAT_DIM = BOTTLENECK_CHANNELS * BOTTLENECK_SIZE * BOTTLENECK_SIZE

# 各输入尺寸对应的编码器通道序列（每项一个 stride=2 卷积块，尾项须为 512）
# 448: 6 块下采样 448→7；224: 5 块下采样 224→7
ENCODER_CHANNELS = {
    448: [32, 64, 128, 256, 512, 512],
    224: [32, 64, 128, 256, 512],
}


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """编码器基础块：stride=2 卷积（尺寸减半）+ BN + ReLU

    参数:
        in_ch:  输入通道数
        out_ch: 输出通道数
    返回:
        nn.Sequential, 对输入 (B, in_ch, H, W) 输出 (B, out_ch, H/2, W/2)
    """
    return nn.Sequential(
        # kernel=4, stride=2, padding=1 是标准的"尺寸精确减半"组合
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """解码器基础块：最近邻上采样（尺寸加倍）+ 卷积 + BN + ReLU

    用 Upsample+Conv 替代 ConvTranspose2d，可避免转置卷积常见的棋盘伪影。
    对输入 (B, in_ch, H, W) 输出 (B, out_ch, 2H, 2W)。
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _group_count(channels: int) -> int:
    """返回可整除 channels 的 GroupNorm 组数（优先每组至少 8 个通道）。"""
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0 and channels // groups >= 8:
            return groups
    return 1


class ResidualUpsampleBlock(nn.Module):
    """适度增强的上采样残差块：稳定小 batch 训练并改善细线重建。

    最近邻上采样仍与旧版一致；主分支增加第二个 3x3 卷积，旁路保留
    低频信息。GroupNorm 不依赖 batch 统计，适合 224/448 图片导致的小 batch。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv2d(in_ch, out_ch, kernel_size=1))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


def build_decoder(latent_dim: int, input_size: int = 448,
                  init_seed: int = None, variant: str = "legacy"):
    """构造 CAE 同款解码器，返回 (decoder_fc, decoder_conv) 两个模块

    供 CAE 自身与"统一解码器"对照实验（不同编码器共享同一解码器）使用。

    参数:
        latent_dim: 特征向量维度（decoder_fc 的输入维）
        input_size: 目标图片边长（448/224），决定上采样块数
        init_seed:  不为 None 时，先临时固定随机种子再建层、随后恢复
                    原随机状态——多次调用（乃至不同模型里调用）得到的
                    解码器初始权重**逐位一致**，实现严格的控制变量
        variant:    legacy=历史 BatchNorm 单卷积块；residual=GroupNorm
                    双卷积残差块。保留 legacy 以兼容既有 checkpoint。
    返回:
        (decoder_fc, decoder_conv):
            decoder_fc:   Linear(latent_dim -> 512*7*7)
            decoder_conv: 上采样堆栈, (B,512,7,7) -> (B,3,S,S) + Sigmoid
    """
    if variant not in {"legacy", "residual"}:
        raise ValueError(f"未知 decoder variant: {variant}")
    channels = ENCODER_CHANNELS[input_size]

    def _make():
        fc = nn.Linear(latent_dim, FLAT_DIM)
        # channels[::-1] 反转通道序列；zip(rev, rev[1:]) 把相邻两项配成
        # (输入通道, 输出通道) 对，如 448 版: (512,512),(512,256)...(64,32)，
        # 自然得到 len-1 个上采样块（448:5 块 7→224；224:4 块 7→112）
        rev = channels[::-1]
        block = _deconv_block if variant == "legacy" else ResidualUpsampleBlock
        blocks = [block(i, o) for i, o in zip(rev, rev[1:])]
        conv = nn.Sequential(
            *blocks,
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        return fc, conv

    if init_seed is None:
        return _make()
    # fork 随机状态：固定种子建层后恢复，不影响调用方后续的随机序列
    rng_state = torch.get_rng_state()
    torch.manual_seed(init_seed)
    fc, conv = _make()
    torch.set_rng_state(rng_state)
    return fc, conv


class CAE(nn.Module):
    """卷积自编码器

    参数:
        latent_dim: 特征向量 z 的维度（当前正式接口默认 512）。
                    这是"特征是否充分"实验的核心可调参数，
                    可从大到小尝试（512/256/128/64）找最小充分维度。
        input_size: 输入图片边长，支持 448（默认，原图）和 224（降采样图）。
                    两种尺寸都下采样到 512×7×7 瓶颈，仅卷积块数不同。
        decoder_init_seed: 不为 None 时解码器用该种子独立初始化（见
                    build_decoder），供"统一解码器"对照实验保证各组
                    解码器初始权重一致；默认 None 沿用全局随机流
        decoder_variant: legacy 保持历史 checkpoint 兼容；residual 为当前
                    正式对照使用的 GroupNorm 残差解码器
    """

    def __init__(self, latent_dim: int = 512, input_size: int = 448,
                 decoder_init_seed: int = None,
                 decoder_variant: str = "legacy"):
        super().__init__()
        if input_size not in ENCODER_CHANNELS:
            raise ValueError(
                f"input_size 仅支持 {list(ENCODER_CHANNELS)}，收到 {input_size}"
            )
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.decoder_variant = decoder_variant
        channels = ENCODER_CHANNELS[input_size]

        # ---------- 编码器 ----------
        # 每个卷积块空间尺寸减半；448 输入 6 块、224 输入 5 块，都收敛到 7×7
        # 例 448: (B,3,448,448)→32×224→64×112→128×56→256×28→512×14→512×7
        # 例 224: (B,3,224,224)→32×112→64×56→128×28→256×14→512×7
        enc_blocks = []
        in_ch = 3
        for out_ch in channels:
            enc_blocks.append(_conv_block(in_ch, out_ch))
            in_ch = out_ch
        self.encoder_conv = nn.Sequential(*enc_blocks)
        # 展平后压缩到 latent_dim 维特征向量
        self.encoder_fc = nn.Linear(FLAT_DIM, latent_dim)

        # ---------- 解码器 ----------
        # 结构见 build_decoder：Linear 恢复 512×7×7 + 对称上采样堆栈；
        # 抽成共享构造函数是为了让其他编码器（如 ViT）能用同一个解码器
        self.decoder_fc, self.decoder_conv = build_decoder(
            latent_dim, input_size, decoder_init_seed, decoder_variant
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码：图片 -> 特征向量（训练完成后单独用它做特征提取）

        参数:
            x: 输入图片, shape (B, 3, S, S), S=input_size(448/224), 取值 [0, 1]
        返回:
            z: 特征向量, shape (B, latent_dim)
        """
        feat = self.encoder_conv(x)          # (B, 512, 7, 7)
        # flatten(1) 表示从第 1 维开始展平（保留 batch 维）:
        # (B, 512, 7, 7) -> (B, 512*7*7) = (B, 25088)
        feat = feat.flatten(1)
        return self.encoder_fc(feat)         # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码：特征向量 -> 重建图片

        参数:
            z: 特征向量, shape (B, latent_dim)
        返回:
            x_hat: 重建图片, shape (B, 3, S, S), S=input_size, 取值 (0, 1)
        """
        feat = self.decoder_fc(z)            # (B, 25088)
        # view 把一维向量重排回卷积特征图布局; -1 表示 batch 维自动推断
        feat = feat.view(-1, BOTTLENECK_CHANNELS, BOTTLENECK_SIZE, BOTTLENECK_SIZE)
        return self.decoder_conv(feat)       # (B, 3, 448, 448)

    def forward(self, x: torch.Tensor) -> tuple:
        """完整前向：编码 + 解码

        参数:
            x: 输入图片, shape (B, 3, S, S), S=input_size
        返回:
            (x_hat, z): 重建图片 (B,3,S,S) 与特征向量 (B,latent_dim)
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def count_parameters(self) -> int:
        """统计可训练参数总量（用于模型规模检查）"""
        # 生成器表达式：遍历所有参数张量，p.numel() 是单个张量的元素个数，
        # sum 把它们累加成总参数量
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
