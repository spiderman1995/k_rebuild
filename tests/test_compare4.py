# -*- coding: utf-8 -*-
"""阶段 11d 测试关卡：四组对照实验脚本（机制测试）

通过标准（见 开发计划.md 第三期）：
    1. 小样本（夹具36张降采样）小轮数（2 epoch）四组全部跑通
    2. 每组的 train/val/test 损失历史长度 = epoch 数，值有限且为正
    3. JSON 与 loss 曲线 PNG 文件生成
    4. 四组 checkpoint 各自保存

说明：机制测试不要求损失低（2 epoch 训不出来），指标由 11e 正式实验负责。
"""

import argparse
import json
import math
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_compare4 import METHODS, run_compare
from src.prepare_224 import downsample_one

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(ROOT, "tests", "fixtures", "kline36")
TMP_DIR = os.path.join(ROOT, "tests", "_cmp4_tmp")


def test_compare4_machinery():
    """全流程机制测试：夹具 36 张 → 降采样 → 四组 × 2 epoch"""
    img224 = os.path.join(TMP_DIR, "img224")
    os.makedirs(img224, exist_ok=True)
    try:
        # 夹具 36 张全部降采样为 224（36 张按 7:1:2 切成 25/4/7）
        for name in sorted(os.listdir(FIXTURE_DIR)):
            downsample_one(os.path.join(FIXTURE_DIR, name),
                           os.path.join(img224, name))

        epochs = 2
        args = argparse.Namespace(
            img_dir=img224, epochs=epochs, batch_size=8, lr=1e-3,
            latent_dim=64, seed=42,
            out_dir=os.path.join(TMP_DIR, "out"),
            ckpt_dir=os.path.join(TMP_DIR, "ckpt"),
            methods="cae,inpaint,mae,color",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        result = run_compare(args)

        # 关卡 1+2：请求的四组齐全，历史长度正确，损失有限且为正
        history = result["history"]
        requested = set(args.methods.split(","))
        assert requested <= set(METHODS), "测试请求了未定义的方法"
        assert set(history) == requested, f"方法不齐: {list(history)}"
        for m, h in history.items():
            for split in ("train", "val", "test"):
                assert len(h[split]) == epochs, f"{m}.{split} 长度错误"
                assert all(math.isfinite(v) and v > 0 for v in h[split]), (
                    f"{m}.{split} 损失非法: {h[split]}"
                )

        # 关卡 3：JSON 与曲线图落盘
        assert os.path.isfile(result["json_path"]), "JSON 未生成"
        assert os.path.isfile(result["plot_path"]), "loss 曲线图未生成"
        assert os.path.getsize(result["plot_path"]) > 10_000, "曲线图文件异常小"

        # 关卡 4：所跑各组 checkpoint 保存
        for m in requested:
            assert os.path.isfile(os.path.join(args.ckpt_dir, f"{m}.pth")), (
                f"{m} checkpoint 未保存"
            )
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


def test_plot_mixed_lengths():
    """混合长度曲线出图：合并 20ep 与 50ep 历史时画图不应崩溃

    绘图必须走子进程（plot_compare4.py 不 import torch）：
    本测试进程里已加载 torch，进程内直接 import matplotlib 会触发
    OpenMP 运行时冲突崩溃，见 开发计划.md 阶段 11 开发备注。
    """
    import subprocess
    os.makedirs(TMP_DIR, exist_ok=True)
    try:
        # 构造两条长度不同的假历史（3ep 与 6ep）
        history = {
            "a": {"label": "方法A", "train": [3, 2, 1], "val": [3, 2, 1],
                  "test": [3, 2, 1]},
            "b": {"label": "方法B", "train": [6, 5, 4, 3, 2, 1],
                  "val": [6, 5, 4, 3, 2, 1], "test": [6, 5, 4, 3, 2, 1]},
        }
        json_path = os.path.join(TMP_DIR, "mixed.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"epochs": 6, "history": history}, f, ensure_ascii=False)

        png_path = os.path.join(TMP_DIR, "mixed.png")
        script = os.path.join(ROOT, "src", "plot_compare4.py")
        result = subprocess.run(
            [sys.executable, script, "--json", json_path, "--out", png_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"混合长度绘图失败:\n{result.stderr}"
        assert os.path.isfile(png_path) and os.path.getsize(png_path) > 10_000
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


def test_merge_rerun():
    """合并逻辑（阶段11f）：分两次各跑一组，第二次应并入第一次的结果同图对比"""
    img224 = os.path.join(TMP_DIR, "img224")
    os.makedirs(img224, exist_ok=True)
    try:
        for name in sorted(os.listdir(FIXTURE_DIR)):
            downsample_one(os.path.join(FIXTURE_DIR, name),
                           os.path.join(img224, name))

        def make_args(methods):
            return argparse.Namespace(
                img_dir=img224, epochs=1, batch_size=8, lr=1e-3,
                latent_dim=64, seed=42,
                out_dir=os.path.join(TMP_DIR, "out"),
                ckpt_dir=os.path.join(TMP_DIR, "ckpt"),
                methods=methods,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

        run_compare(make_args("cae"))            # 第一次：只跑 cae
        result = run_compare(make_args("inpaint"))  # 第二次：只跑 inpaint

        # 第二次的 history 应同时含两组（旧结果被读入合并）
        assert set(result["history"]) == {"cae", "inpaint"}, (
            f"合并失败: {list(result['history'])}"
        )
        with open(result["json_path"], encoding="utf-8") as f:
            saved = json.load(f)
        assert set(saved["history"]) == {"cae", "inpaint"}, "JSON 未包含合并结果"
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
