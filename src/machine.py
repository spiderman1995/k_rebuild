# -*- coding: utf-8 -*-
"""机器标识模块（流程位置：多机协作时实验产出目录的隔离策略）

结果按机器分文件夹（results/<机器标识>/...），两台机器跑同名实验
写不同路径，git 同步时不会互相冲突（约定见 开发计划.md）。
"""

import os
import platform


def get_machine_tag() -> str:
    """返回本机的结果目录标识

    优先取环境变量 KLINE_MACHINE（如本机=local、203机=203，简短可读）；
    未设置则退回主机名——两台机器主机名不同，结果依然天然隔离。
    """
    return os.environ.get("KLINE_MACHINE") or platform.node().lower()
