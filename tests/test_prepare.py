# -*- coding: utf-8 -*-
"""阶段 10 测试关卡：224 数据准备工具

通过标准（见 开发计划.md 第三期）：
    1. seq 模式取排序后前 N 张；random 模式同种子可复现、不同种子不同
    2. 输出图片为 224×224×3（无 alpha）
    3. 输出像素值 = 原图对应 2×2 块的均值（与 numpy 参考实现数值一致）
    4. --dst-raw 时原图被完整复制；坏图跳过不中断
"""

import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prepare_224 import downsample_one, prepare, select_files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "tests", "fixtures", "kline36")
TMP_DIR = os.path.join(ROOT, "tests", "_prepare_tmp")


def _args(**kw):
    """构造 prepare 参数（默认：本地 36 张源，输出到临时目录）"""
    base = dict(src=IMG_DIR, dst_224=os.path.join(TMP_DIR, "p224"),
                dst_raw=None, n=5, mode="seq", seed=42)
    base.update(kw)
    # ** 把字典解包成关键字参数传给 Namespace
    return argparse.Namespace(**base)


def test_select_seq_and_random():
    """seq 取排序前 N；random 同种子可复现、不同种子不同"""
    all_sorted = sorted(f for f in os.listdir(IMG_DIR) if f.endswith(".png"))
    assert select_files(IMG_DIR, 5, "seq") == all_sorted[:5], "seq 未取排序前5张"

    r1 = select_files(IMG_DIR, 10, "random", seed=42)
    r2 = select_files(IMG_DIR, 10, "random", seed=42)
    r3 = select_files(IMG_DIR, 10, "random", seed=7)
    assert r1 == r2, "random 同种子不可复现"
    assert r1 != r3, "random 不同种子结果相同，随机性可疑"
    assert len(set(r1)) == 10, "random 抽样出现重复"


def test_downsample_is_2x2_mean():
    """核心数值验证：输出像素 = 原图 2×2 块均值（对照 numpy 参考实现）"""
    name = sorted(os.listdir(IMG_DIR))[0]
    os.makedirs(TMP_DIR, exist_ok=True)
    out_path = os.path.join(TMP_DIR, "check.png")
    try:
        downsample_one(os.path.join(IMG_DIR, name), out_path)
        out = np.asarray(Image.open(out_path), dtype=np.float64)
        assert out.shape == (224, 224, 3), f"输出 shape 错误: {out.shape}"

        # numpy 参考实现：reshape 把 (448,448,3) 拆成 (224,2,224,2,3)，
        # 对两个"2"维取均值，即每个 2×2 块的平均
        src = np.asarray(
            Image.open(os.path.join(IMG_DIR, name)).convert("RGB"), dtype=np.float64
        )
        ref = src.reshape(224, 2, 224, 2, 3).mean(axis=(1, 3))
        # PIL 保存为 uint8 有舍入，允许 ±1 的量化误差
        assert np.abs(out - ref).max() <= 1.0, "输出不等于 2×2 块均值"
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


def test_prepare_full_flow_with_raw_copy():
    """全流程：5 张 seq + 复制原图，数量、尺寸、字节一致性全查"""
    try:
        args = _args(dst_raw=os.path.join(TMP_DIR, "raw"))
        result = prepare(args)
        assert result["n_224"] == 5 and result["n_raw"] == 5 and not result["errors"]

        for name in result["picked"]:
            img224 = Image.open(os.path.join(args.dst_224, name))
            assert img224.size == (224, 224) and img224.mode == "RGB"
            # 原图复制应逐字节一致：读入二进制直接比较
            src_bytes = open(os.path.join(IMG_DIR, name), "rb").read()
            raw_bytes = open(os.path.join(args.dst_raw, name), "rb").read()
            assert src_bytes == raw_bytes, f"原图复制不一致: {name}"
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


def test_prepare_skips_bad_image():
    """坏图（尺寸异常）应跳过并记入 errors，不中断整体流程"""
    bad_src = os.path.join(TMP_DIR, "bad_src")
    os.makedirs(bad_src, exist_ok=True)
    try:
        # 放 2 张好图 + 1 张 100×100 的坏图
        good = sorted(os.listdir(IMG_DIR))[:2]
        for g in good:
            shutil.copy2(os.path.join(IMG_DIR, g), os.path.join(bad_src, g))
        Image.new("RGB", (100, 100)).save(os.path.join(bad_src, "bad.png"))

        result = prepare(_args(src=bad_src, n=3))
        assert result["n_224"] == 2, "好图应全部处理成功"
        assert result["errors"] == ["bad.png"], "坏图应被记录到 errors"
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
