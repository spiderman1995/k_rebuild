# -*- coding: utf-8 -*-
"""环境检查脚本（流程位置：新机器部署前的自检工具，203 首次部署用）

收集并打印机器配置与项目运行条件，同时写入 env_report.txt 便于回传：
    - 系统 / CPU / 内存
    - 网络：各启用网卡的 IPv4 / MAC，以及 Tailscale IP（远程连接用）
    - 各磁盘剩余空间（含 C 盘是否告急）
    - GPU（nvidia-smi）与 torch 的 CUDA 可用性
    - Python 与关键依赖包版本
    - 数据目录存在性与 PNG 数量

用法（在项目根目录）：
    python src/check_env.py
    python src/check_env.py --data-dir E:\\k_graphy\\分钟k线图
纯标准库实现（torch 等按"装了就报、没装就标记缺失"处理），无额外依赖。
"""

import argparse
import ctypes
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import uuid

REPORT_LINES = []


def say(line: str = ""):
    """打印同时缓存到报告（结束时统一写 env_report.txt）"""
    print(line)
    REPORT_LINES.append(line)


def section(title: str):
    """分节标题"""
    say(f"\n===== {title} =====")


def check_system():
    """系统 / CPU / 内存"""
    section("系统")
    say(f"操作系统: {platform.platform()}")
    say(f"处理器:   {platform.processor()}")
    say(f"CPU 核心: {os.cpu_count()}")
    if platform.system() == "Windows":
        # 用 Win32 API GlobalMemoryStatusEx 取物理内存（标准库无直接接口）
        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MemStatus()
        m.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        say(f"物理内存: {m.ullTotalPhys / 1024**3:.1f} GB "
            f"(可用 {m.ullAvailPhys / 1024**3:.1f} GB)")


def check_network():
    """网络标识：各启用网卡的 IPv4/MAC + Tailscale IP（远程 SSH 连接依据）"""
    section("网络")
    say(f"主机名: {socket.gethostname()}")

    got_detail = False
    if platform.system() == "Windows":
        # 用 PowerShell 查询网卡（属性名不受系统语言影响，比解析 ipconfig 稳）：
        # 对每个已启用(Up)网卡取 名称/MAC/该网卡上的全部 IPv4，输出为 JSON
        ps = ("Get-NetAdapter | Where-Object {$_.Status -eq 'Up'} | ForEach-Object { "
              "[PSCustomObject]@{name=$_.Name; mac=$_.MacAddress; "
              "ip=@((Get-NetIPAddress -InterfaceIndex $_.ifIndex -AddressFamily IPv4 "
              "-ErrorAction SilentlyContinue).IPAddress)} } | ConvertTo-Json")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                 capture_output=True, text=True, timeout=30)
            if out.returncode == 0 and out.stdout.strip():
                data = json.loads(out.stdout)
                # 单网卡时 ConvertTo-Json 返回对象而非数组，统一成列表
                if isinstance(data, dict):
                    data = [data]
                for nic in data:
                    ips = nic["ip"] if isinstance(nic["ip"], list) else [nic["ip"]]
                    ips = [i for i in ips if i]  # 过滤 None
                    say(f"网卡: {nic['name']:<12} MAC: {nic['mac']}  "
                        f"IPv4: {', '.join(ips) if ips else '(无)'}")
                got_detail = True
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
            pass

    if not got_detail:
        # 兜底（非 Windows 或 PowerShell 失败）：socket 拿 IP 列表，uuid 拿一个 MAC
        try:
            ips = socket.gethostbyname_ex(socket.gethostname())[2]
            say(f"IPv4: {', '.join(ips)}")
        except socket.gaierror:
            say("IPv4: 获取失败")
        # uuid.getnode() 返回 48 位整数 MAC；格式化为常见的冒号分隔十六进制
        mac = uuid.getnode()
        say("MAC(其一): " + ":".join(f"{(mac >> s) & 0xff:02x}"
                                     for s in range(40, -1, -8)))

    # Tailscale IP：装了 tailscale 的机器可用它做跨网段直连（SSH 首选地址）
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                             text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            say(f"Tailscale IP: {out.stdout.strip()}  ← 跨网段 SSH 首选地址")
        else:
            say("Tailscale: 已安装但未登录/未运行")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        say("Tailscale: 未安装")


def check_disks():
    """各盘剩余空间；C 盘紧张时给出显式警告（缓存/临时文件选盘依据）"""
    section("磁盘")
    for letter in "CDEFG":
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            usage = shutil.disk_usage(drive)
            free_gb = usage.free / 1024**3
            warn = "  ⚠ 空间告急，缓存/临时文件不要放此盘" if free_gb < 10 else ""
            say(f"{drive}  总 {usage.total / 1024**3:.0f} GB, "
                f"剩余 {free_gb:.1f} GB{warn}")


def check_gpu():
    """nvidia-smi 与 torch 的 CUDA 视角（两边都看，排查驱动/包不匹配）"""
    section("GPU")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            say(f"nvidia-smi: {out.stdout.strip()}")
        else:  # 部分驱动不支持 query 参数，退回普通输出的首行
            plain = subprocess.run(["nvidia-smi"], capture_output=True,
                                   text=True, timeout=30)
            first = plain.stdout.strip().splitlines()[:3]
            say("nvidia-smi(摘要): " + " | ".join(first))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        say("nvidia-smi: 不可用（无 NVIDIA 驱动或不在 PATH）")

    try:
        import torch
        say(f"torch: {torch.__version__} | CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            say(f"torch 设备: {torch.cuda.get_device_name(0)} | "
                f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        say("torch: 未安装")


def check_packages():
    """项目依赖包版本清单"""
    section("Python 与依赖包")
    say(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for pkg in ["torch", "torchvision", "numpy", "PIL", "matplotlib",
                "pytorch_msssim", "timm", "pytest"]:
        try:
            mod = __import__(pkg)
            say(f"{pkg}: {getattr(mod, '__version__', '已安装(无版本号)')}")
        except ImportError:
            say(f"{pkg}: ✗ 未安装")


def check_data(data_dir: str):
    """数据目录存在性与 PNG 数量（207 万级目录用 scandir 流式计数）"""
    section("数据目录")
    say(f"路径: {data_dir}")
    if not os.path.isdir(data_dir):
        say("✗ 目录不存在")
        return
    count = 0
    # scandir 迭代器逐项扫描，不一次性载入 200 万条目录项
    with os.scandir(data_dir) as it:
        for entry in it:
            if entry.name.lower().endswith(".png"):
                count += 1
    say(f"✓ PNG 数量: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="机器环境自检")
    parser.add_argument("--data-dir", default=r"E:\k_graphy\分钟k线图",
                        help="数据目录（默认 203 的路径）")
    args = parser.parse_args()

    say(f"环境检查报告 | 主机: {platform.node()}")
    check_system()
    check_disks()
    check_gpu()
    check_packages()
    check_data(args.data_dir)

    # 报告落盘，便于拷贝回传
    with open("env_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_LINES))
    say("\n报告已写入 env_report.txt")


if __name__ == "__main__":
    main()
