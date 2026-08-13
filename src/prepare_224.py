# -*- coding: utf-8 -*-
"""224 数据准备工具（流程位置：ViT 类方法的前置数据处理，两台机器通用）

功能：从源目录选取 N 张 448×448 K线图，做 2×2 均值降采样得到 224×224，
保存到输出目录；可选地把选中的原图也复制一份（远程机器落库用）。

选取模式：
    seq    —— 按文件名排序后取前 N 张（本地方案：顺序取样）
    random —— 随机抽 N 张，seed 固定可复现（远程 203 方案：随机取样）

降采样方法：**2×2 块均值（BOX/AREA 插值）**——448 恰为 224 的 2 倍，
每个输出像素 = 原图对应 2×2 块 4 个像素的平均值，是整数倍降采样中
失真最小的方法（插值对照实验结论见 开发计划.md）。

用法：
    本地（1万张原图已放入 分钟k线图/）：
        python src/prepare_224.py --src 分钟k线图 --dst-224 pic_to_224x224 --n 10000 --mode seq
    远程 203（从 207 万中随机抽 1 万并落库原图）：
        python src/prepare_224.py --src E:\\k_graphy\\分钟k线图 --dst-raw 分钟k线图 ^
               --dst-224 pic_to_224x224 --n 10000 --mode random --seed 42
"""

import argparse
import os
import random
import shutil
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger, setup_logger

log = get_logger("prepare")

# 源图与目标图的固定规格
SRC_SIZE = (448, 448)
DST_SIZE = (224, 224)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="选取并降采样K线图到 224×224")
    parser.add_argument("--src", required=True, help="源图片目录（448×448 PNG）")
    parser.add_argument("--dst-224", required=True, help="224×224 输出目录")
    parser.add_argument("--dst-raw", default=None,
                        help="可选：把选中的原图复制到此目录（远程落库用）")
    parser.add_argument("--n", type=int, default=10000, help="选取张数")
    parser.add_argument("--mode", choices=["seq", "random"], default="seq",
                        help="选取模式：seq=排序后取前N张，random=随机抽N张")
    parser.add_argument("--seed", type=int, default=42,
                        help="random 模式的随机种子（保证可复现）")
    return parser.parse_args()


def select_files(src: str, n: int, mode: str, seed: int = 42) -> list:
    """从源目录选取 N 个 PNG 文件名

    参数:
        src:  源目录
        n:    选取张数；超过可用数量时取全部并告警
        mode: 'seq' 排序取前 N；'random' 随机抽 N（种子固定可复现）
        seed: random 模式的种子
    返回:
        选中的文件名列表（random 模式下也按名称排序返回，便于断点核对）
    """
    files = sorted(f for f in os.listdir(src) if f.lower().endswith(".png"))
    if not files:
        raise FileNotFoundError(f"{src} 中没有 PNG 图片")
    if n > len(files):
        log.warning("要求 %d 张但源目录只有 %d 张，将全部选取", n, len(files))
        n = len(files)

    if mode == "seq":
        picked = files[:n]
    else:
        # random.Random(seed) 独立随机器：不污染全局随机状态，同种子结果恒定
        picked = random.Random(seed).sample(files, n)
    return sorted(picked)


def downsample_one(src_path: str, dst_path: str):
    """单张图：RGBA→RGB、2×2 均值降采样 448→224、保存 PNG

    参数:
        src_path: 源图路径（448×448）
        dst_path: 输出路径（224×224）
    """
    img = Image.open(src_path)
    if img.size != SRC_SIZE:
        # 异常尺寸直接报错跳过比静默处理好：数据问题要暴露
        raise ValueError(f"{os.path.basename(src_path)} 尺寸 {img.size} != {SRC_SIZE}")
    # convert('RGB') 丢弃 alpha 通道（实测恒为255，无信息）
    # Image.BOX 即块均值插值：缩小2倍时每个输出像素=对应2×2块的平均值
    img.convert("RGB").resize(DST_SIZE, Image.BOX).save(dst_path)


def prepare(args) -> dict:
    """执行完整数据准备流程

    返回:
        dict: {'picked': 选中文件名列表, 'n_224': 降采样成功数,
               'n_raw': 原图复制数, 'errors': 失败文件列表}
    """
    setup_logger()
    os.makedirs(args.dst_224, exist_ok=True)
    if args.dst_raw:
        os.makedirs(args.dst_raw, exist_ok=True)

    picked = select_files(args.src, args.n, args.mode, args.seed)
    log.info("数据准备启动 | 源: %s (%s模式) | 选中 %d 张 | 输出: %s",
             args.src, args.mode, len(picked), args.dst_224)

    n_224 = n_raw = 0
    errors = []
    t0 = time.time()
    for i, name in enumerate(picked, 1):
        src_path = os.path.join(args.src, name)
        try:
            downsample_one(src_path, os.path.join(args.dst_224, name))
            n_224 += 1
            if args.dst_raw:
                # copy2 保留文件时间戳等元信息
                shutil.copy2(src_path, os.path.join(args.dst_raw, name))
                n_raw += 1
        except (ValueError, OSError) as e:
            # 单张失败不中断全局（207万张里可能混入坏图），记录后继续
            errors.append(name)
            log.warning("处理失败并跳过: %s (%s)", name, e)
        if i % 1000 == 0:
            log.info("进度 %d/%d (%.0fs)", i, len(picked), time.time() - t0)

    log.info("数据准备完成 | 降采样 %d 张 -> %s | 复制原图 %d 张 | 失败 %d 张 | 耗时 %.0fs",
             n_224, args.dst_224, n_raw, len(errors), time.time() - t0)
    return {"picked": picked, "n_224": n_224, "n_raw": n_raw, "errors": errors}


if __name__ == "__main__":
    prepare(parse_args())
