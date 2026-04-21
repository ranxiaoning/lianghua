"""
向后兼容 shim — 已迁移至 core/fetch.py
此文件仅保留供旧脚本过渡期调用，请更新 import 为 core.fetch。
"""
import warnings
warnings.warn(
    "fetch_stock.py 已迁移至 core/fetch.py，请使用 'from core.fetch import ...'",
    DeprecationWarning,
    stacklevel=2,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.fetch import fetch_daily, fetch_hourly, load_cached, DATA_DIR

__all__ = ["fetch_daily", "fetch_hourly", "load_cached", "DATA_DIR"]
