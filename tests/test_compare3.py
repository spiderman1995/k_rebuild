# -*- coding: utf-8 -*-
"""阶段 13 测试关卡：三组对照实验脚本（机制测试）

通过标准（见 开发计划.md 第三期）：
    1. 小样本（夹具36张降采样）小轮数（2 epoch）三组全部跑通
    2. 每组的 train/val/test 损失历史长度 = epoch 数，值有限且为正
    3. JSON 与 loss 曲线 PNG 文件生成
    4. 三组 checkpoint 各自保存

说明：机制测试不要求损失低（2 epoch 训不出来），正式实验另行验收。
"""

import argparse
import json
import math
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.experiment_compare3 as compare3
from src.experiment_compare3 import METHODS, run_compare
from src.prepare_224 import downsample_one

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(ROOT, "tests", "fixtures", "kline36")
TMP_DIR = os.path.join(ROOT, "tests", "_cmp3_tmp")


class TinyAutoencoder(torch.nn.Module):
    """隔离外部预训练权重的轻量模型，只验证实验编排契约。"""

    def __init__(self, latent_dim: int, use_backbone: bool):
        super().__init__()
        feature_dim = 4
        if use_backbone:
            # 名称必须是 backbone.*，同时覆盖预训练模型的分组学习率路径。
            self.backbone = torch.nn.Conv2d(3, feature_dim, kernel_size=1)
        else:
            self.encoder_conv = torch.nn.Conv2d(3, feature_dim, kernel_size=1)
        self.encoder_fc = torch.nn.Linear(feature_dim, latent_dim)
        self.output = torch.nn.Linear(latent_dim, 3)
        self.latent_dim = latent_dim
        self.pre_projection_dim = latent_dim
        self.decoder_variant = "test_tiny"

    def forward(self, x):
        """返回符合正式模型契约的重建图与特征向量。"""
        encoder = self.backbone if hasattr(self, "backbone") else self.encoder_conv
        feature = encoder(x).mean(dim=(2, 3))
        z = self.encoder_fc(feature)
        rgb = torch.sigmoid(self.output(z)).view(-1, 3, 1, 1)
        return rgb.expand(-1, -1, x.shape[2], x.shape[3]), z

    def count_parameters(self):
        """返回可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def fake_build_model(method: str, latent_dim: int, seed: int):
    """为三组实验返回轻量替身；seed 保留正式构造函数签名。"""
    del seed
    return TinyAutoencoder(latent_dim, use_backbone=method.startswith("vit_"))


def test_active_methods_are_exactly_three():
    """已取消的遮挡修复/去色重彩不得继续留在训练入口。"""
    assert list(METHODS) == ["cae", "vit_recon", "vit_mask"]


def test_compare3_machinery(monkeypatch):
    """全流程机制测试：夹具 36 张 → 降采样 → 三组 × 2 epoch。

    用小型替身模型隔离预训练权重下载；真实 ViT 结构由 test_mae.py 覆盖。
    """
    monkeypatch.setattr(compare3, "build_model", fake_build_model)
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
            methods="cae,vit_recon,vit_mask",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        result = run_compare(args)

        # 关卡 1+2：请求的三组齐全，历史长度正确，损失有限且为正
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


def test_make_optimizer_by_structure():
    """开闭原则修复（12-O）：按参数名前缀而非方法名清单决定分组学习率

    含 backbone.* 参数的模型 → 两组学习率（主干 0.1x）；
    不含的 → 单组统一学习率。不依赖任何具体方法名。
    """
    from src.experiment_compare3 import make_optimizer

    class FakePretrained(torch.nn.Module):
        """带 backbone 子模块的假模型（模拟预训练微调场景）"""
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.head = torch.nn.Linear(4, 4)

    opt = make_optimizer(FakePretrained(), lr=1e-3)
    assert len(opt.param_groups) == 2, "含 backbone 应分两组"
    assert abs(opt.param_groups[0]["lr"] - 1e-4) < 1e-12, "主干应为 0.1x 学习率"
    assert abs(opt.param_groups[1]["lr"] - 1e-3) < 1e-12, "其余应为全速学习率"

    opt = make_optimizer(torch.nn.Linear(4, 4), lr=1e-3)
    assert len(opt.param_groups) == 1, "无 backbone 应单组"
    assert abs(opt.param_groups[0]["lr"] - 1e-3) < 1e-12


def test_plot_mixed_lengths():
    """混合长度曲线出图：合并 20ep 与 50ep 历史时画图不应崩溃

    绘图必须走子进程（plot_compare3.py 不 import torch）：
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
        script = os.path.join(ROOT, "src", "plot_compare3.py")
        result = subprocess.run(
            [sys.executable, script, "--json", json_path, "--out", png_path],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"混合长度绘图失败:\n{result.stderr}"
        assert os.path.isfile(png_path) and os.path.getsize(png_path) > 10_000
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)


def test_merge_rerun(monkeypatch):
    """合并逻辑：分两次各跑一组，第二次应并入第一次的结果同图对比。"""
    monkeypatch.setattr(compare3, "build_model", fake_build_model)
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

        run_compare(make_args("cae"))
        result = run_compare(make_args("vit_recon"))

        # 第二次的 history 应同时含两组（旧结果被读入合并）
        assert set(result["history"]) == {"cae", "vit_recon"}, (
            f"合并失败: {list(result['history'])}"
        )
        with open(result["json_path"], encoding="utf-8") as f:
            saved = json.load(f)
        assert set(saved["history"]) == {"cae", "vit_recon"}, "JSON 未包含合并结果"
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
