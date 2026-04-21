"""
技术指标计算模块
================
提供 A 股常用技术指标的纯函数实现，依赖仅 pandas + numpy。

指标列表：
  MA5 / MA20           简单移动平均
  EMA12 / EMA26        指数移动平均（MACD 原料）
  DIFF / DEA / MACD    MACD 系统（A股 Wind 标准：柱 = 2×(DIFF−DEA)）
  RSI(14)              相对强弱指数（Wilder 平滑）
  BB_upper/mid/lower   布林带（中轨=MA20，上下轨=MA20±2σ）
  BB_width             布林带宽（波动率代理）
"""

import numpy as np
import pandas as pd


# ─────────────────────── 基础均线 ───────────────────────

def ma(close: pd.Series, period: int) -> pd.Series:
    """简单移动平均（SMA）。"""
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    """
    指数移动平均（EMA）。
    adjust=False 与国内主流软件（同花顺/东方财富）计算结果一致。
    """
    return close.ewm(span=period, adjust=False).mean()


# ─────────────────────── MACD ───────────────────────

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD 系统（A股 Wind / 通达信标准）。

    Returns
    -------
    diff : DIFF = EMA_fast − EMA_slow
    dea  : DEA  = EMA_signal(DIFF)
    hist : MACD 柱 = 2 × (DIFF − DEA)  ← 乘 2 是国内标准
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    diff = ema_fast - ema_slow
    dea  = ema(diff, signal)
    hist = 2.0 * (diff - dea)
    return diff.round(4), dea.round(4), hist.round(4)


# ─────────────────────── RSI ───────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI（Wilder 指数平滑，alpha = 1/period）。
    与同花顺、东方财富默认算法一致。
    超买参考：>70  超卖参考：<30
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).round(2)


# ─────────────────────── 布林带 ───────────────────────

def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    布林带（Bollinger Bands）。

    中轨 = MA₂₀
    上轨 = MA₂₀ + std_mult × σ₂₀
    下轨 = MA₂₀ − std_mult × σ₂₀
    σ 使用总体标准差（ddof=0），与通达信一致。

    Returns
    -------
    upper, mid, lower
    """
    mid   = close.rolling(window=period, min_periods=period).mean()
    std   = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper.round(4), mid.round(4), lower.round(4)


# ─────────────────────── 一键计算 ───────────────────────

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    对含 'close' 列的 DataFrame 一次性附加所有技术指标列。
    返回新 DataFrame，不修改原始对象。

    新增列（见 CLAUDE.md 技术指标规范）：
      ma5, ma20
      ema12, ema26
      diff, dea, macd
      rsi14
      bb_upper, bb_mid, bb_lower, bb_width
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame 缺少 'close' 列")

    out   = df.copy()
    close = out['close']

    # ── 均线 ──────────────────────────────────────────
    out['ma5']  = ma(close, 5)
    out['ma20'] = ma(close, 20)

    # ── EMA ──────────────────────────────────────────
    out['ema12'] = ema(close, 12)
    out['ema26'] = ema(close, 26)

    # ── MACD ─────────────────────────────────────────
    out['diff'], out['dea'], out['macd'] = macd(close, 12, 26, 9)

    # ── RSI ──────────────────────────────────────────
    out['rsi14'] = rsi(close, 14)

    # ── 布林带 ───────────────────────────────────────
    out['bb_upper'], out['bb_mid'], out['bb_lower'] = bollinger_bands(close, 20, 2.0)
    # 带宽衡量波动率收缩/扩张程度
    out['bb_width'] = ((out['bb_upper'] - out['bb_lower']) / out['bb_mid']).round(4)

    return out


# ─────────────────────── 信号解读（辅助） ───────────────────────

def latest_signals(df: pd.DataFrame) -> dict:
    """
    读取最新一行指标，返回可读的信号字典。
    用于快速打印当前信号状态。
    """
    if df.empty:
        return {}
    row = df.iloc[-1]

    def _safe(key: str, fmt: str = '.2f') -> str:
        v = row.get(key)
        return f'{v:{fmt}}' if v is not None and not (isinstance(v, float) and np.isnan(v)) else '—'

    macd_trend = '金叉区间' if row.get('diff', 0) > row.get('dea', 0) else '死叉区间'
    rsi_zone   = ('超买' if row.get('rsi14', 50) > 70
                  else '超卖' if row.get('rsi14', 50) < 30
                  else '中性')
    bb_pos     = '上轨附近' if row.get('close', 0) >= row.get('bb_upper', 0) * 0.99 else \
                 '下轨附近' if row.get('close', 0) <= row.get('bb_lower', 0) * 1.01 else '轨道内'

    return {
        '日期':     str(row.get('date', ''))[:10],
        '收盘价':   _safe('close'),
        'MA5':      _safe('ma5'),
        'MA20':     _safe('ma20'),
        'EMA12':    _safe('ema12'),
        'EMA26':    _safe('ema26'),
        'DIFF':     _safe('diff'),
        'DEA':      _safe('dea'),
        'MACD柱':   _safe('macd'),
        'MACD形态': macd_trend,
        'RSI14':    _safe('rsi14'),
        'RSI信号':  rsi_zone,
        '布林上轨': _safe('bb_upper'),
        '布林中轨': _safe('bb_mid'),
        '布林下轨': _safe('bb_lower'),
        '布林带宽': _safe('bb_width'),
        '价格位置': bb_pos,
    }
