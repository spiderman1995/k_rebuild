# -*- coding: utf-8 -*-
"""数据加载模块（流程位置：整个训练/评估管线的第一环）

功能：读取 `分钟k线图/` 文件夹中的 448x448 RGBA PNG，完成以下预处理后
以 PyTorch 张量形式提供给 DataLoader：
    1. 丢弃 Alpha 通道（实测恒为 255，无信息量），RGBA -> RGB
    2. 像素值除以 255 归一化到 [0, 1]
    3. HWC 布局转为 PyTorch 卷积要求的 CHW 布局，输出 shape (3, 448, 448)
    4. 可选反色（1 - x）：让白色背景变 0、K线线条变高激活值

设计依据见 README.md 第二、三节。
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.logger import get_logger

# 模块级子 logger：日志带 'kline.dataset' 前缀
log = get_logger("dataset")


class KLineDataset(Dataset):
    """分钟K线图数据集

    每个样本是一张单日分钟K线图，返回 (图片张量, 文件名)。
    文件名一并返回是为了评估阶段能把重建图和原图对应起来。

    参数:
        img_dir: 图片文件夹路径，内含 `股票代码_日期.png` 格式的 PNG
        invert:  是否反色（1 - x）。默认 False。反色后背景为 0、
                 线条为高激活值，更利于卷积网络学习稀疏线条特征
        files:   可选的文件名清单。传入时只加载清单中的文件（用于
                 配合 split.py 构造训练/验证/测试子集）；不传则
                 加载文件夹内全部 PNG
        size:    图片边长，默认 448（原图）；加载 pic_to_224x224 的
                 降采样图时传 224，尺寸校验与输出 shape 随之调整
    """

    def __init__(self, img_dir: str, invert: bool = False, files=None,
                 size: int = 448):
        self.img_dir = img_dir
        self.invert = invert
        self.size = size
        if files is not None:
            # 按清单构造子集：校验文件确实存在，尽早暴露清单与磁盘不一致
            missing = [f for f in files if not os.path.isfile(os.path.join(img_dir, f))]
            if missing:
                raise FileNotFoundError(f"清单中 {len(missing)} 个文件不存在，如: {missing[:3]}")
            self.files = sorted(files)
        else:
            # sorted 保证文件顺序在不同操作系统上一致，训练可复现
            self.files = sorted(
                f for f in os.listdir(img_dir) if f.lower().endswith(".png")
            )
        if not self.files:
            raise FileNotFoundError(f"{img_dir} 中没有找到任何 PNG 图片")
        log.info("数据集加载: %s | %d 张图片 | invert=%s",
                 img_dir, len(self.files), invert)

    def __len__(self) -> int:
        """返回数据集样本总数（DataLoader 依赖此方法确定迭代范围）"""
        return len(self.files)

    def __getitem__(self, idx: int):
        """加载并预处理第 idx 张图片

        参数:
            idx: 样本下标，范围 [0, len(self)-1]
        返回:
            img:  图片张量, shape (3, 448, 448), dtype float32, 取值 [0, 1]
            name: 对应的文件名字符串, 如 '300001_2019-01-03.png'
        """
        name = self.files[idx]
        pil_img = Image.open(os.path.join(self.img_dir, name))

        expected = (self.size, self.size)
        if pil_img.size != expected:
            # 先告警再抛错：日志文件中留下异常样本的名字，方便清洗数据
            log.warning("异常图片 %s: 尺寸 %s != 预期 %s",
                        name, pil_img.size, expected)
            raise ValueError(
                f"{name} 尺寸为 {pil_img.size}，与预期 {expected} 不符"
            )

        # convert('RGB') 丢弃 Alpha 通道：4 通道 RGBA -> 3 通道 RGB
        arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0

        # PIL 读出的布局是 HWC (448, 448, 3)，PyTorch 卷积要求 CHW (3, 448, 448)。
        # transpose(2, 0, 1) 表示新维度顺序取原来的第 2、0、1 维，即把通道维提到最前
        arr = arr.transpose(2, 0, 1)

        if self.invert:
            # 反色：白底(1.0)变 0，深色线条变高值
            arr = 1.0 - arr

        # from_numpy 与 numpy 数组共享内存零拷贝转换；上游 arr 是新建数组，无副作用
        img = torch.from_numpy(arr)
        return img, name
