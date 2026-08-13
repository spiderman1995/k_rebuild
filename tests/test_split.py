# -*- coding: utf-8 -*-
"""阶段 8 测试关卡：数据划分工具

通过标准（见 开发计划.md 第二期）：
    1. 7:2:1 比例正确（36 张 -> 25/7/4）
    2. 三集合互斥且并集 = 全量（不丢样本、不重复）
    3. 同种子两次切分结果完全一致（可复现）
    4. KLineDataset 支持按文件清单构造子集
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import KLineDataset
from src.split import split_dir, split_files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "分钟k线图")


def test_split_ratios():
    """36 张按 7:2:1 -> 25/7/4（round(36*0.7)=25, round(36*0.2)=7, 剩 4）"""
    train, val, test = split_dir(IMG_DIR)
    assert (len(train), len(val), len(test)) == (25, 7, 4), (
        f"切分数量错误: {len(train)}/{len(val)}/{len(test)}"
    )


def test_split_disjoint_and_complete():
    """互斥且完整：三集合无交集，并集等于全量文件"""
    all_files = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".png"))
    train, val, test = split_files(all_files)
    s_train, s_val, s_test = set(train), set(val), set(test)
    # & 是集合交集运算符，互斥意味着两两交集为空
    assert not (s_train & s_val), "训练/验证集有重叠"
    assert not (s_train & s_test), "训练/测试集有重叠"
    assert not (s_val & s_test), "验证/测试集有重叠"
    # | 是集合并集运算符
    assert s_train | s_val | s_test == set(all_files), "切分后丢失或多出样本"


def test_split_reproducible():
    """同种子两次切分结果完全一致；不同种子结果不同"""
    a1 = split_dir(IMG_DIR, seed=42)
    a2 = split_dir(IMG_DIR, seed=42)
    assert a1 == a2, "同种子切分结果不一致，实验不可复现"
    b = split_dir(IMG_DIR, seed=7)
    assert a1 != b, "不同种子切分结果相同，随机性可疑"


def test_split_invalid_inputs():
    """非法输入：比例和不为 1、空清单，都应报错而非静默"""
    with pytest.raises(ValueError):
        split_files(["a.png"], ratios=(0.5, 0.2, 0.1))  # 和为 0.8
    with pytest.raises(ValueError):
        split_files([])


def test_dataset_from_file_list():
    """KLineDataset 按清单构造子集：数量与内容一致"""
    train, val, test = split_dir(IMG_DIR)
    ds_val = KLineDataset(IMG_DIR, files=val)
    assert len(ds_val) == len(val)
    img, name = ds_val[0]
    assert img.shape == (3, 448, 448)
    assert name in val, "子集数据集返回了清单之外的文件"


def test_dataset_file_list_missing_file():
    """清单里有不存在的文件时应立刻报错，而不是训练中途才崩"""
    with pytest.raises(FileNotFoundError):
        KLineDataset(IMG_DIR, files=["不存在的文件.png"])
