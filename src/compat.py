# -*- coding: utf-8 -*-
"""跨 torch 版本兼容模块（流程位置：多机协作的版本差异隔离层）

背景：203 机器（Tesla V100）的 NVIDIA 驱动 443.18 最高支持 CUDA 10.2，
torch 被钉死在 1.10.1+cu102；本机是 torch 2.6。两版 API 的差异统一在
本模块消化，业务代码不感知 torch 版本。
"""

import torch


def load_checkpoint(path: str, map_location):
    """加载 checkpoint（兼容 torch 1.10 ~ 2.x）

    torch>=1.13 的 load 有 weights_only 参数（2.6 起默认 True，会拒绝
    载入含配置字典的 checkpoint，必须显式传 False）；torch 1.10 没有
    该参数（行为等同 False）。先按新 API 调用，老版本 TypeError 时回退。

    参数:
        path:         checkpoint 文件路径
        map_location: 设备映射（同 torch.load 的同名参数）
    返回:
        checkpoint 字典
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # torch < 1.13：没有 weights_only 参数，默认行为即完整反序列化
        return torch.load(path, map_location=map_location)
