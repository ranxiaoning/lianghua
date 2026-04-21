#!/usr/bin/env python3
"""
数据分析主流程
==============
拉取行情 → 计算全套技术指标 → 三联图输出 → 保存带指标数据

用法:
  python core/pipeline.py --symbol 601288 --start 20200101 --end 20241231
  python core/pipeline.py --symbol 600519 --start 20180101 --end 20241231 --no-plot
"""

import argparse
import sys
from pathlib import Path

# 确保从项目根目录运行时 core/ 包可被找到
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from core.fetch import fetch_daily
from core.indicators import compute_all, latest_signals

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHARTS_DIR = ROOT / "charts"
DATA_DIR   = ROOT / "data"
CHARTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════
# 图表
# ══════════════════════════════════════════════════════

def plot_analysis(df: pd.DataFrame, symbol: str,
                  start: str, end: str) -> Path:
    """
    三联技术分析图：
      ① 价格 + MA5 + MA20 + EMA12/26 + 布林带
      ② MACD 柱状图 + DIFF + DEA 线
      ③ RSI(14) 折线 + 超买/超卖水平线
    """
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs  = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[5, 2, 2],
        hspace=0.06,
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    dates = df["date"]

    # ── ① 价格面板 ─────────────────────────────────────
    ax1.fill_between(dates, df["bb_lower"], df["bb_upper"],
                     alpha=0.08, color="#1976D2", label="布林通道")
    ax1.plot(dates, df["bb_upper"],  color="#1976D2", lw=0.7, ls="--", alpha=0.6)
    ax1.plot(dates, df["bb_mid"],    color="#1976D2", lw=0.9, ls="-",  alpha=0.7, label="布林中轨 MA20")
    ax1.plot(dates, df["bb_lower"],  color="#1976D2", lw=0.7, ls="--", alpha=0.6)

    ax1.plot(dates, df["close"], color="#212121", lw=1.2, label="收盘价", zorder=4)
    ax1.plot(dates, df["ma5"],   color="#F57C00", lw=1.0, label="MA5",  alpha=0.85)
    ax1.plot(dates, df["ema12"], color="#E53935", lw=0.9, label="EMA12", ls="-.", alpha=0.8)
    ax1.plot(dates, df["ema26"], color="#8E24AA", lw=0.9, label="EMA26", ls="-.", alpha=0.8)

    ax1.set_ylabel("价格（元）", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.8)
    ax1.grid(True, alpha=0.18)
    ax1.set_title(
        f"{symbol}  技术分析图  {start[:4]}-{start[4:6]}-{start[6:]} ~ "
        f"{end[:4]}-{end[4:6]}-{end[6:]}",
        fontsize=13, fontweight="bold", pad=10,
    )
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── ② MACD 面板 ────────────────────────────────────
    hist      = df["macd"]
    colors_bar = ["#c0392b" if v >= 0 else "#27ae60" for v in hist]
    ax2.bar(dates, hist, color=colors_bar, width=1.0, alpha=0.75, label="MACD 柱")
    ax2.plot(dates, df["diff"], color="#E53935", lw=1.1, label="DIFF")
    ax2.plot(dates, df["dea"],  color="#1976D2", lw=1.1, label="DEA")
    ax2.axhline(0, color="#888888", lw=0.6, ls="--")

    ax2.set_ylabel("MACD", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.8)
    ax2.grid(True, alpha=0.18)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── ③ RSI 面板 ─────────────────────────────────────
    ax3.plot(dates, df["rsi14"], color="#8E24AA", lw=1.2, label="RSI(14)")
    ax3.axhline(70, color="#c0392b", lw=0.9, ls="--", alpha=0.8, label="超买 70")
    ax3.axhline(50, color="#888888", lw=0.7, ls=":",  alpha=0.6)
    ax3.axhline(30, color="#27ae60", lw=0.9, ls="--", alpha=0.8, label="超卖 30")
    ax3.fill_between(dates, df["rsi14"], 70,
                     where=df["rsi14"] >= 70, alpha=0.15, color="#c0392b")
    ax3.fill_between(dates, df["rsi14"], 30,
                     where=df["rsi14"] <= 30, alpha=0.15, color="#27ae60")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI(14)", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.8)
    ax3.grid(True, alpha=0.18)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30)

    # 年份分界线（三图共享）
    years = pd.date_range(
        f"{df['date'].dt.year.min()}-01-01",
        f"{df['date'].dt.year.max()}-12-31",
        freq="YS",
    )
    for yr in years:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(yr, color="#dddddd", lw=0.8, zorder=0)

    out = CHARTS_DIR / f"{symbol}_analysis_{start}_{end}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图表] → {out}")
    return out


# ══════════════════════════════════════════════════════
# 摘要打印
# ══════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, symbol: str) -> None:
    sigs = latest_signals(df)
    sep  = "─" * 55

    print(f"\n{sep}")
    print(f"  {symbol}  最新技术指标摘要（{sigs.get('日期', '')}）")
    print(sep)

    groups = [
        ("价格",   ["收盘价", "MA5", "MA20", "EMA12", "EMA26"]),
        ("MACD",   ["DIFF", "DEA", "MACD柱", "MACD形态"]),
        ("RSI",    ["RSI14", "RSI信号"]),
        ("布林带", ["布林上轨", "布林中轨", "布林下轨", "布林带宽", "价格位置"]),
    ]
    for group_name, keys in groups:
        print(f"\n  【{group_name}】")
        for k in keys:
            v = sigs.get(k, "—")
            print(f"    {k:<10}: {v}")

    # 近期涨跌
    print(f"\n  【近期表现】")
    for n, label in [(5, "近5日"), (20, "近20日"), (60, "近60日")]:
        if len(df) > n:
            ret = (df["close"].iloc[-1] / df["close"].iloc[-1 - n] - 1) * 100
            sign = "+" if ret >= 0 else ""
            print(f"    {label:<6}: {sign}{ret:.2f}%")

    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════
# 数据保存
# ══════════════════════════════════════════════════════

def save_enriched(df: pd.DataFrame, symbol: str,
                  start: str, end: str) -> Path:
    """保存带指标的完整数据到 data/ 目录（parquet + CSV 双轨）。"""
    stem = f"{symbol}_indicators_{start}_{end}"

    out_pq  = DATA_DIR / f"{stem}.parquet"
    out_csv = DATA_DIR / f"{stem}.csv"

    df.to_parquet(out_pq, index=False)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[数据] parquet → {out_pq}")
    print(f"[数据] CSV     → {out_csv}  ({len(df)} 行 × {len(df.columns)} 列)")
    return out_pq


# ══════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════

def run(symbol: str, start: str, end: str,
        no_plot: bool = False) -> pd.DataFrame:
    """
    完整数据分析流程：
      1. 拉取 / 读取行情数据
      2. 计算全套技术指标
      3. 打印最新指标摘要
      4. 保存带指标数据（parquet）
      5. 生成三联技术分析图
    """
    print(f"\n{'═'*55}")
    print(f"  数据分析 Pipeline — {symbol}  {start}~{end}")
    print(f"{'═'*55}")

    # Step 1: 拉取数据
    df = fetch_daily(symbol, start, end)
    print(f"  行情数据: {len(df)} 行  "
          f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")

    # Step 2: 指标计算
    df = compute_all(df)
    print(f"  指标计算: 完成（{len(df.columns)} 列）")

    # Step 3: 摘要
    print_summary(df, symbol)

    # Step 4: 保存
    save_enriched(df, symbol, start, end)

    # Step 5: 图表
    if not no_plot:
        plot_analysis(df, symbol, start, end)

    return df


# ══════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A股技术分析 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbol",   required=True, help="股票代码，如 601288")
    parser.add_argument("--start",    required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end",      required=True, help="结束日期 YYYYMMDD")
    parser.add_argument("--no-plot",  action="store_true", help="不生成图表")
    args = parser.parse_args()

    try:
        run(args.symbol, args.start, args.end, no_plot=args.no_plot)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
