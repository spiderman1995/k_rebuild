# -*- coding: utf-8 -*-
"""MAE 模型：ViT-Tiny 编码器 + 轻量解码头（流程位置：四组对照的组③）

设计（见 开发计划.md 阶段 11 实验协议）：
    - 输入是"已被 mask_patches 破坏的整图"（SimMIM 风格的全序列输入，
      区别于原版 MAE 只喂可见 patch——这样与其他三组保持
      "破坏输入 → 重建原图"的统一框架，损失可横向比较）
    - 编码器两种模式（pretrained 开关）：
        False: 手写精简 ViT-Tiny（patch16, dim192, depth6, heads3）**从零训练**
        True:  timm 的 ImageNet 预训练 ViT-Tiny（12 层，571 万参数），
               权重 `vit_tiny_patch16_224.augreg_in21k_ft_in1k`（已缓存 D:\\hf_cache，
               运行前需设环境变量 HF_HOME 指向缓存盘，避免往已满的 C 盘下载）
    - 解码头：每个 patch token 经线性层直接回归 16×16×3 的像素块，
      再拼回整图（MAE 论文同款的极简解码思路）

特征提取：encode() 返回全部 patch token 的均值池化向量 (B, embed_dim)。
"""

import torch
import torch.nn as nn

# ViT-Tiny 规格（Touvron et al. DeiT-Tiny 同款尺寸）
IMG_SIZE = 224
PATCH = 16
N_PATCHES = (IMG_SIZE // PATCH) ** 2   # 14*14 = 196 个 patch
EMBED_DIM = 192
DEPTH = 6
N_HEADS = 3


class MAE(nn.Module):
    """ViT-Tiny 编码器 + 线性像素解码头

    参数:
        embed_dim:  token 维度，默认 192（ViT-Tiny）
        depth:      Transformer 层数，默认 6（仅从零模式生效）
        n_heads:    注意力头数，默认 3（仅从零模式生效）
        pretrained: True 时用 timm 的 ImageNet 预训练 ViT-Tiny 做编码器
                    （12 层，参数在 'backbone.' 前缀下，便于微调时单独设小学习率）
    """

    def __init__(self, embed_dim: int = EMBED_DIM, depth: int = DEPTH,
                 n_heads: int = N_HEADS, pretrained: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        # 契约对齐（里氏替换）：CAE/MAEUnified 都用 latent_dim 表示特征 z
        # 的维度，MAE 的 z 维度就是 embed_dim，加别名使三类模型可统一读取
        self.latent_dim = embed_dim
        self.pretrained = pretrained

        if pretrained:
            # 延迟 import：只有用预训练时才要求装 timm
            import timm
            # num_classes=0 去掉分类头，只留特征主干
            self.backbone = timm.create_model(
                "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                pretrained=True, num_classes=0,
            )
            # ImageNet 归一化常数：预训练权重是在此分布上训练的，
            # 喂 [0,1] 原值会偏离其输入分布。register_buffer 随模型迁移设备、
            # 存入 checkpoint，但不参与梯度更新
            self.register_buffer(
                "in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer(
                "in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            # ---------- patch 嵌入（从零模式） ----------
            # 用 kernel=stride=16 的卷积一步完成"切块+线性投影"：
            # (B,3,224,224) -> (B,192,14,14)，每个空间位置即一个 patch token
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=PATCH, stride=PATCH)
            # 可学习位置编码：(1, 196, 192)，broadcast 加到每个样本上
            self.pos_embed = nn.Parameter(torch.zeros(1, N_PATCHES, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)  # ViT 标准初始化

            # ---------- Transformer 编码器 ----------
            # batch_first=True: 输入 shape (B, 序列长, 维度)，与我们的布局一致
            # norm_first=True: Pre-LN 结构，从零训练更稳定
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
                activation="gelu", batch_first=True, norm_first=True, dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
            self.norm = nn.LayerNorm(embed_dim)

        # ---------- 解码头（两种模式共用） ----------
        # 每个 token 线性回归出自己那块 16×16×3=768 个像素
        self.decoder_head = nn.Linear(embed_dim, PATCH * PATCH * 3)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        """图片 -> 编码后的 patch token 序列

        参数:
            x: (被破坏的)图片, shape (B, 3, 224, 224)
        返回:
            tokens, shape (B, 196, embed_dim)
        """
        if self.pretrained:
            # 先转到 ImageNet 归一化分布，再走预训练主干
            x = (x - self.in_mean) / self.in_std
            # forward_features 返回 (B, 197, 192)：第 0 个是 CLS token，
            # 它不对应任何图像块，重建用不上，切掉只留 196 个 patch token
            return self.backbone.forward_features(x)[:, 1:]
        # (B,3,224,224) -> (B,192,14,14) -> flatten(2): (B,192,196)
        # -> transpose(1,2) 交换后两维: (B,196,192)，即序列长×维度
        t = self.patch_embed(x).flatten(2).transpose(1, 2)
        t = t + self.pos_embed          # 注入位置信息
        return self.norm(self.encoder(t))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """特征提取：全部 token 均值池化为单个特征向量

        参数:
            x: 图片, shape (B, 3, 224, 224)
        返回:
            z: 特征向量, shape (B, embed_dim)
        """
        return self._tokens(x).mean(dim=1)  # 沿序列维求均值

    def forward(self, x: torch.Tensor) -> tuple:
        """前向：破坏图 -> 重建整图

        参数:
            x: 被破坏的图片, shape (B, 3, 224, 224)
        返回:
            (x_hat, z): 重建图 (B,3,224,224) 与特征向量 (B,embed_dim)
        """
        tokens = self._tokens(x)                     # (B, 196, 192)
        pixels = self.decoder_head(tokens)           # (B, 196, 768)
        # 拼回整图（unpatchify）:
        # (B,196,768) -> (B,14,14,16,16,3): 每格还原为 16×16×3 的像素块
        b = pixels.shape[0]
        grid = IMG_SIZE // PATCH
        pixels = pixels.view(b, grid, grid, PATCH, PATCH, 3)
        # permute 重排为图片布局 (B,3,14,16,14,16)，
        # 相邻的 (格行,格内行)(格列,格内列) 合并后即 (B,3,224,224)
        x_hat = pixels.permute(0, 5, 1, 3, 2, 4).reshape(b, 3, IMG_SIZE, IMG_SIZE)
        x_hat = torch.sigmoid(x_hat)                 # 像素约束到 [0,1]
        return x_hat, tokens.mean(dim=1)

    def count_parameters(self) -> int:
        """统计可训练参数总量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MAEUnified(nn.Module):
    """统一解码器版：预训练 ViT-Tiny 编码器 + 投影瓶颈 + CAE 同款解码器

    为"四组解码器完全一致"的对照实验（开发计划 阶段 11h）而设：
        编码器: timm 预训练 ViT-Tiny → 196 个 patch token 均值池化 (B,192)
                → 线性投影到 latent_dim 维特征 z（显式全局瓶颈，
                与 CAE 组的信息通道等价——这是与 MAE 类的关键差别）
        解码器: 与 CAE-224 **同结构**（build_decoder 构造），配合
                decoder_init_seed 可与 CNN 组做到**初始权重逐位一致**

    参数:
        latent_dim:        特征 z 维度（与 CNN 组保持一致，默认 256）
        decoder_init_seed: 解码器初始化种子（四组传同一值即严格控制变量）
    """

    def __init__(self, latent_dim: int = 256, decoder_init_seed: int = None):
        super().__init__()
        # 延迟 import：只有用到本类才要求装 timm
        import timm

        from src.model_cae import (BOTTLENECK_CHANNELS, BOTTLENECK_SIZE,
                                   build_decoder)

        self.latent_dim = latent_dim
        self.backbone = timm.create_model(
            "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
            pretrained=True, num_classes=0,
        )
        # ImageNet 归一化常数（同 MAE 类，见其注释）
        self.register_buffer(
            "in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # 投影瓶颈：token 池化向量 (B,192) -> 特征 z (B,latent_dim)
        self.proj = nn.Linear(EMBED_DIM, latent_dim)
        # CAE 同款解码器（224 版），种子固定时与 CNN 组初始权重逐位一致
        self.decoder_fc, self.decoder_conv = build_decoder(
            latent_dim, input_size=224, init_seed=decoder_init_seed
        )
        self._bottleneck_shape = (BOTTLENECK_CHANNELS, BOTTLENECK_SIZE,
                                  BOTTLENECK_SIZE)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码：图片 -> 特征向量 z

        参数:
            x: (被破坏的)图片, shape (B, 3, 224, 224), 取值 [0,1]
        返回:
            z: 特征向量, shape (B, latent_dim)
        """
        x = (x - self.in_mean) / self.in_std
        # forward_features 输出 (B,197,192)，去掉第 0 个 CLS token 后均值池化
        pooled = self.backbone.forward_features(x)[:, 1:].mean(dim=1)
        return self.proj(pooled)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码：特征向量 -> 重建图片（与 CAE.decode 完全同构）

        参数:
            z: 特征向量, shape (B, latent_dim)
        返回:
            x_hat: 重建图片, shape (B, 3, 224, 224)
        """
        feat = self.decoder_fc(z).view(-1, *self._bottleneck_shape)
        return self.decoder_conv(feat)

    def forward(self, x: torch.Tensor) -> tuple:
        """前向：破坏图 -> z -> 重建整图

        返回:
            (x_hat, z): 重建图 (B,3,224,224) 与特征 (B,latent_dim)
        """
        z = self.encode(x)
        return self.decode(z), z

    def count_parameters(self) -> int:
        """统计可训练参数总量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
