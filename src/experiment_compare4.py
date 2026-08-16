# -*- coding: utf-8 -*-
"""四组"破坏重建"对照实验（流程位置：阶段 11 主实验脚本）

实验协议（见 开发计划.md 阶段 11，由用户确定）：
    - 数据：只用 2×2 均值降采样的 224×224 图（pic_to_224x224/）
    - 划分：7:1:2 训练/验证/测试（seed 固定），四组共用同一划分
    - 每组 20 epoch，纯 MSE 损失；逐 epoch 记录训练/验证/测试三条 loss
    - 四组目标统一为"从(被破坏的)输入重建原图"，损失可横向比较：
        cae     组① 输入=原图（不破坏，压缩重建基线，CAE-224）
        inpaint 组② 输入=随机挖矩形块（CAE-224）
        mae     组③ 输入=按16×16网格遮75%（ViT-Tiny 从零）
        color   组④ 输入=灰度化（CAE-224）
    - 验证/测试的破坏图案用固定种子，保证逐 epoch 曲线平滑可比

产出：
    results/compare4/compare4.json     全部损失历史与配置
    results/compare4/loss_curves.png   每组三条曲线 + 四组验证集对比

用法（1 万张就位后，在项目根目录）：
    python src/experiment_compare4.py --img-dir pic_to_224x224 --epochs 20
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corruptions import erase_rects, mask_patches, to_grayscale
from src.dataset import KLineDataset
from src.logger import get_logger, setup_logger
from src.machine import get_machine_tag
from src.model_cae import CAE
from src.model_mae import MAE
from src.split import split_files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = get_logger("compare4")

# 结果按机器分文件夹隔离（results/<机器标识>/...），多机跑同名实验
# 也不会产生 git 冲突；标识来源见 get_machine_tag
MACHINE = get_machine_tag()

# 方法定义：名称 -> (中文标签, 破坏函数)；破坏函数签名 f(x, generator)
METHODS = {
    "cae": ("① CAE压缩重建", lambda x, g: x),  # 不破坏
    "inpaint": ("② 遮挡修复", lambda x, g: erase_rects(x, generator=g)),
    "mae": ("③ MAE(ViT从零)", lambda x, g: mask_patches(x, generator=g)),
    "color": ("④ 去色重彩", lambda x, g: to_grayscale(x)),
    # ③′：与 ③ 同样的破坏方式，但编码器换成 ImageNet 预训练 ViT-Tiny，
    # 用于对照"从零 vs 预训练"（运行前需设 HF_HOME 指向权重缓存盘）
    "mae_pre": ("③′ MAE(ViT预训练)", lambda x, g: mask_patches(x, generator=g)),
    # ③″：统一解码器版（阶段11h）——预训练 ViT 编码器 + 投影瓶颈 +
    # CAE 同款解码器，与 cae/inpaint/color 三组解码器初始权重逐位一致
    "mae_uni": ("③″ MAE(ViT预训练·统一解码器)",
                lambda x, g: mask_patches(x, generator=g)),
}
# 验证/测试破坏图案的固定种子（与训练随机破坏区分开）
EVAL_CORRUPT_SEED = 12345
# 正式对照统一使用真实 512 维接口；旧版 mae/mae_pre 固定 192 维，仅保留
# 历史复现用途，不属于当前统一维度协议。
FEATURE_DIM = 512
# 新实验采用适度增强的 GroupNorm 残差解码器；legacy 仍可加载历史模型。
SHARED_DECODER_VARIANT = "residual"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="四组破坏重建对照实验")
    parser.add_argument("--img-dir", default=os.path.join(ROOT, "pic_to_224x224"),
                        help="224×224 图片目录")
    parser.add_argument("--epochs", type=int, default=20, help="每组训练轮数")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam 学习率")
    parser.add_argument("--latent-dim", type=int, default=FEATURE_DIM,
                        help="统一特征维度（正式实验固定为512）")
    parser.add_argument("--seed", type=int, default=42, help="划分与初始化种子")
    # 默认输出按机器分文件夹（results/<机器标识>/compare4），防多机 git 冲突
    parser.add_argument("--out-dir",
                        default=os.path.join(ROOT, "results", MACHINE, "compare4"))
    parser.add_argument("--ckpt-dir",
                        default=os.path.join(ROOT, "checkpoints", MACHINE, "compare4"))
    parser.add_argument("--methods", default="cae,inpaint,color,mae_uni",
                        help="逗号分隔的方法子集（调试用）")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(method: str, latent_dim: int, seed: int) -> torch.nn.Module:
    """按方法名构造模型

    seed 用作**解码器初始化种子**：cae/inpaint/color/mae_uni 四组的解码器
    由此保证初始权重逐位一致（严格控制变量，见 build_decoder）。
    mae/mae_pre 是早期方案（线性 token 解码头），保留供历史对照。
    """
    if method == "mae":
        return MAE()
    if method == "mae_pre":
        return MAE(pretrained=True)
    if method == "mae_uni":
        from src.model_mae import MAEUnified
        return MAEUnified(
            latent_dim=latent_dim, decoder_init_seed=seed,
            decoder_variant=SHARED_DECODER_VARIANT,
        )
    return CAE(
        latent_dim=latent_dim, input_size=224, decoder_init_seed=seed,
        decoder_variant=SHARED_DECODER_VARIANT,
    )


def make_optimizer(model: torch.nn.Module, lr: float):
    """按模型结构构造优化器：含预训练主干（backbone.*）时用分组学习率

    分组依据：预训练主干已经会"看图"，用 1/10 学习率轻微调，避免大学习率
    把 ImageNet 学来的通用视觉特征冲掉（灾难性遗忘）；随机初始化的
    投影层/解码器则用全速学习率从头学。

    判断依据是**参数名前缀**而非方法名清单（开闭原则）：新增带预训练
    主干的方法时无需修改本函数，也杜绝"方法名清单漏改导致主干被全速
    学习率训练、实验结果静默失真"的风险。
    """
    backbone, rest = [], []
    for name, p in model.named_parameters():
        if name.startswith("backbone."):
            backbone.append(p)
        else:
            rest.append(p)

    if backbone:
        return torch.optim.Adam([
            {"params": backbone, "lr": lr * 0.1},
            {"params": rest, "lr": lr},
        ])
    return torch.optim.Adam(model.parameters(), lr=lr)


def run_epoch(model, loader, corrupt_fn, device, optimizer=None,
              eval_seed: int = None) -> float:
    """跑一遍数据集，返回平均 MSE 损失

    参数:
        model:      当前模型
        loader:     DataLoader
        corrupt_fn: 破坏函数 f(x, generator)
        device:     计算设备
        optimizer:  传入则为训练模式（反向传播更新），否则纯评估
        eval_seed:  评估时的破坏种子；训练时传 None（每 batch 随机破坏）
    返回:
        整个数据集的逐样本平均损失
    """
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()
    # 评估用固定种子 generator：每个 epoch 的破坏图案完全一致，曲线平滑可比
    gen = None
    if eval_seed is not None:
        gen = torch.Generator().manual_seed(eval_seed)

    total, n = 0.0, 0
    # torch.set_grad_enabled 按训练/评估开关梯度计算（评估省显存）
    with torch.set_grad_enabled(training):
        for imgs, _names in loader:
            corrupted = corrupt_fn(imgs, gen)          # 在 CPU 上破坏后再上卡
            imgs, corrupted = imgs.to(device), corrupted.to(device)
            x_hat, _z = model(corrupted)
            loss = torch.nn.functional.mse_loss(x_hat, imgs)  # 目标恒为原图
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * imgs.size(0)
            n += imgs.size(0)
    return total / n


def plot_in_subprocess(json_path: str, plot_path: str):
    """在独立子进程中出图（不 import torch，绕开 OpenMP 运行时冲突）

    本机 torch 与 anaconda MKL 各带一份 OpenMP 运行时（OMP Error #15），
    同进程混用 torch 和 matplotlib 会直接崩溃；绘图逻辑在
    src/plot_compare4.py，其进程内只有 matplotlib，不会触发冲突。

    参数:
        json_path: 已写好的 compare4.json 路径
        plot_path: 输出 PNG 路径
    """
    script = os.path.join(ROOT, "src", "plot_compare4.py")
    # sys.executable 是当前 Python 解释器路径，保证子进程用同一环境
    result = subprocess.run(
        [sys.executable, script, "--json", json_path, "--out", plot_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # 子进程失败不应吞掉：把 stderr 带进异常，实验数据(JSON)已保存不受影响
        raise RuntimeError(f"绘图子进程失败:\n{result.stderr}")


def build_loaders(args) -> tuple:
    """按 7:1:2 划分数据并构建三个 DataLoader（各方法共用同一划分）

    返回:
        (train_loader, val_loader, test_loader, split_sizes)
        split_sizes 为 {'train': n, 'val': n, 'test': n}，供结果 JSON 记录
    """
    files = sorted(f for f in os.listdir(args.img_dir) if f.endswith(".png"))
    train_f, val_f, test_f = split_files(files, ratios=(0.7, 0.1, 0.2),
                                         seed=args.seed)

    def make_loader(fs, shuffle):
        return DataLoader(KLineDataset(args.img_dir, files=fs, size=224),
                          batch_size=args.batch_size, shuffle=shuffle)

    sizes = {"train": len(train_f), "val": len(val_f), "test": len(test_f)}
    return make_loader(train_f, True), make_loader(val_f, False), \
        make_loader(test_f, False), sizes


def train_one_method(method: str, loaders: tuple, args, device) -> dict:
    """训练单个方法全部 epoch 并保存 checkpoint，返回其损失历史

    参数:
        method:  METHODS 中的方法名
        loaders: (train_loader, val_loader, test_loader)
        args:    命令行参数（本模块内部约定，与 parse_args 同源）
        device:  计算设备
    返回:
        {'label':.., 'train': [...], 'val': [...], 'test': [...]}
    """
    train_loader, val_loader, test_loader = loaders
    label, corrupt_fn = METHODS[method]
    torch.manual_seed(args.seed)  # 每组同种子初始化，控制变量
    model = build_model(method, args.latent_dim, args.seed).to(device)
    if model.latent_dim != args.latent_dim:
        log.warning(
            "[%s] 实际特征维度为 %d，不符合本次统一维度 %d；"
            "该方法仅可作历史参考，不能纳入当前公平对比",
            label, model.latent_dim, args.latent_dim,
        )
    pre_projection_dim = getattr(model, "pre_projection_dim", None)
    if pre_projection_dim is not None and pre_projection_dim < model.latent_dim:
        log.warning(
            "[%s] 输出形状虽为 %d 维，但投影前只有 %d 维；"
            "在编码器结构调整前不能视为真实 %d 维信息瓶颈",
            label, model.latent_dim, pre_projection_dim, model.latent_dim,
        )
    optimizer = make_optimizer(model, args.lr)
    log.info("[%s] 开始训练 | 参数量 %s", label, f"{model.count_parameters():,}")

    h = {
        "label": label,
        "feature_dim": model.latent_dim,
        "pre_projection_dim": getattr(model, "pre_projection_dim", None),
        "decoder_variant": getattr(model, "decoder_variant", "token_linear"),
        "train": [], "val": [], "test": [],
    }
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, corrupt_fn, device, optimizer)
        va = run_epoch(model, val_loader, corrupt_fn, device,
                       eval_seed=EVAL_CORRUPT_SEED)
        te = run_epoch(model, test_loader, corrupt_fn, device,
                       eval_seed=EVAL_CORRUPT_SEED + 1)
        h["train"].append(tr)
        h["val"].append(va)
        h["test"].append(te)
        log.info("[%s] epoch %2d/%d | 训练 %.5f | 验证 %.5f | 测试 %.5f | %ds",
                 label, ep, args.epochs, tr, va, te, time.time() - t0)

    # 每组各存一个 checkpoint，便于后续复用编码器提特征。
    # latent_dim 记录模型的**真实**特征维度（三类模型契约统一为该属性）：
    # mae/mae_pre 的维度固定为 192，与命令行 --latent-dim 无关，
    # 若记录 args.latent_dim 会误导后续加载方
    torch.save({"model_state": model.state_dict(), "method": method,
                "latent_dim": model.latent_dim, "epochs": args.epochs,
                "pre_projection_dim": getattr(model, "pre_projection_dim", None),
                "decoder_variant": getattr(model, "decoder_variant", None)},
               os.path.join(args.ckpt_dir, f"{method}.pth"))
    return h


def merge_and_save_history(json_path: str, history: dict, args,
                           split_sizes: dict) -> dict:
    """与既有结果 JSON 合并后写盘，返回合并后的完整 history

    合并规则：单独重跑某一组（如 --methods mae_uni）时，新曲线并入旧
    compare4.json，同名方法被本次结果覆盖，出图时新旧方法同框对比。
    """
    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as f:
            old = json.load(f).get("history", {})
        if old:
            log.info("检测到已有结果（%s），合并后同图对比", ",".join(old))
        # 字典解包合并：old 在前 history 在后，本次结果覆盖同名旧结果
        history = {**old, **history}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "protocol": "仅用224降采样图; 7:1:2划分; 统一MSE重建原图",
            "epochs": args.epochs, "seed": args.seed,
            "split": split_sizes,
            "history": history,
        }, f, ensure_ascii=False, indent=2)
    return history


def run_compare(args) -> dict:
    """执行完整对照实验（纯编排：数据 → 逐方法训练 → 落盘 → 出图）

    返回:
        dict: {'history': 各方法损失历史, 'json_path': .., 'plot_path': ..}
    """
    setup_logger()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader, sizes = build_loaders(args)
    log.info("对照实验启动 | 数据 %s | 划分 %d/%d/%d | epochs=%d | 设备 %s",
             args.img_dir, sizes["train"], sizes["val"], sizes["test"],
             args.epochs, device)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    history = {}
    for method in methods:
        history[method] = train_one_method(
            method, (train_loader, val_loader, test_loader), args, device)

    json_path = os.path.join(args.out_dir, "compare4.json")
    history = merge_and_save_history(json_path, history, args, sizes)

    # 文件名带最大轮数（如 loss_curves_by_split_50ep.png）：
    # 每次延长训练产生新文件，不覆盖旧图，便于前后对照
    max_ep = max(len(h["train"]) for h in history.values())
    plot_path = os.path.join(args.out_dir, f"loss_curves_by_split_{max_ep}ep.png")
    plot_in_subprocess(json_path, plot_path)
    log.info("实验完成 | 曲线图: %s | 数据: %s", plot_path, json_path)
    return {"history": history, "json_path": json_path, "plot_path": plot_path}


if __name__ == "__main__":
    run_compare(parse_args())
