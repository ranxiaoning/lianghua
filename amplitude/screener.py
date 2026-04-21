#!/usr/bin/env python3
"""
多股票选股 Pipeline
===================
对每支股票运行振幅策略参数扫描，横向对比后推荐最优股票

策略: 从峰值回调 buy_pct% 买入，涨至入场峰值 sell_pct% 卖出

输入方式:
  1. 离散代码:  --symbols 601288,600519,000001,000858
  2. 连续范围:  --code-range 600000 600050   (生成 600000~600050 全部代码)
  3. 混合模式:  --symbols 601288,600519 --code-range 600000 600010

输出:
  ① 控制台: 股票综合排名表 + 年度明细对比
  ② backtest/ 目录:
     - screener_ranking_YYYYMMDD.png   : 综合排名横向对比图
     - screener_equity_YYYYMMDD.png    : Top N 股票净值曲线叠加
     - screener_yearly_YYYYMMDD.png    : 股票 × 年份 盈亏热力图
     - screener_report_YYYYMMDD.md     : Markdown 综合报告
     - screener_results_YYYYMMDD.csv   : 全量结果 CSV

用法示例:
  python stock_screener_pipeline.py \\
      --symbols 601288,600519,000001,000858,600036 \\
      --start 20150101 --end 20241231

  python stock_screener_pipeline.py \\
      --code-range 600000 600030 \\
      --start 20180101 --end 20241231 \\
      --buy-min 88 --buy-max 98 --sell-min 102 --sell-max 120 --step 2

  python stock_screener_pipeline.py \\
      --symbols 601288,600519 --code-range 000001 000010 \\
      --start 20150101 --end 20241231 --top-n 5
"""

import argparse
import os
import sys
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# 复用已有模块
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from amplitude.sweep import (
    load_or_fetch, _sweep_one, parameter_sweep, score_and_top10,
    DEFAULT_WEIGHTS,
)
from amplitude.backtest import run_backtest, calc_metrics, DATA_DIR, BACKTEST_DIR

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

TODAY = datetime.now().strftime('%Y%m%d')


# ═══════════════════════════════════════════════════════════════════
# 1. 股票代码解析
# ═══════════════════════════════════════════════════════════════════

def parse_symbol_range(start_code: str, end_code: str) -> list[str]:
    """
    生成从 start_code 到 end_code 的股票代码列表。
    自动补零至6位，保留前缀格式（如 600000~600050）。
    """
    try:
        s = int(start_code)
        e = int(end_code)
    except ValueError:
        raise ValueError(f"代码范围格式错误: {start_code}~{end_code}，请使用纯数字")
    if s > e:
        s, e = e, s
    prefix_len = max(len(start_code), len(end_code), 6)
    return [str(i).zfill(prefix_len) for i in range(s, e + 1)]


def build_symbol_list(symbols_str: str, range_args: list[str] | None) -> list[str]:
    """合并离散代码 + 连续范围，去重保序。"""
    seen   = set()
    result = []

    def _add(sym: str):
        sym = sym.strip().zfill(6)
        if sym and sym not in seen:
            seen.add(sym)
            result.append(sym)

    if symbols_str:
        for s in symbols_str.split(','):
            _add(s)

    if range_args and len(range_args) == 2:
        for s in parse_symbol_range(range_args[0], range_args[1]):
            _add(s)

    return result


# ═══════════════════════════════════════════════════════════════════
# 2. 单股扫描（支持静默模式，可在子进程中运行）
# ═══════════════════════════════════════════════════════════════════

def screen_one_stock(symbol: str, start: str, end: str,
                     buy_min: float, buy_max: float,
                     sell_min: float, sell_max: float,
                     step: float, cash: float,
                     min_trades: float, min_year_win: float,
                     min_data_years: float,
                     quiet: bool = False) -> dict:
    """
    对单支股票运行参数扫描，返回该股最优参数的指标字典。
    quiet=True 时不打印任何内容（并行子进程使用）。
    数据不足、拉取失败、无有效组合时返回带 _skip=True 的字典。
    """
    try:
        df = load_or_fetch(symbol, start, end, quiet=quiet)
    except Exception as e:
        return {'symbol': symbol, '_error': str(e), '_skip': True}

    if df.empty:
        return {'symbol': symbol, '_error': '数据为空', '_skip': True}

    actual_years = len(df) / 252
    if actual_years < min_data_years:
        return {'symbol': symbol,
                '_error': f'数据不足({actual_years:.1f}年<{min_data_years}年)',
                '_skip': True}

    try:
        sweep_df = parameter_sweep(
            df, buy_min, buy_max, sell_min, sell_max, step, cash,
            verbose=False,
        )
    except Exception as e:
        return {'symbol': symbol, '_error': f'扫描失败: {e}', '_skip': True}

    if sweep_df.empty:
        return {'symbol': symbol, '_error': '无有效回测结果', '_skip': True}

    stable, top1 = score_and_top10(
        sweep_df,
        min_trades_per_year=min_trades,
        min_year_win_rate=min_year_win,
        weights=DEFAULT_WEIGHTS,
        top_n=1,
    )

    if top1.empty:
        best_row = sweep_df.sort_values('ann_ret', ascending=False).iloc[0].copy()
        best_row['composite_score'] = 0.0
        best_row['_relaxed'] = True
    else:
        best_row = top1.iloc[0].copy()
        best_row['_relaxed'] = False

    result = best_row.to_dict()
    result['symbol']      = symbol
    result['data_years']  = round(actual_years, 1)
    result['n_combos']    = len(sweep_df)
    result['_skip']       = False
    result['_error']      = ''
    return result


# ── 顶层可 pickle 的 worker 函数（ProcessPoolExecutor 必须） ──────
def _parallel_worker(packed: tuple) -> dict:
    """
    ProcessPoolExecutor 工作函数，顶层定义保证可 pickle。
    packed 是 screen_one_stock 所有参数的元组（含 quiet=True）。
    """
    (symbol, start, end,
     buy_min, buy_max, sell_min, sell_max,
     step, cash, min_trades, min_year_win, min_data_years) = packed
    return screen_one_stock(
        symbol, start, end,
        buy_min, buy_max, sell_min, sell_max,
        step, cash, min_trades, min_year_win, min_data_years,
        quiet=True,           # 子进程静默，输出由主进程统一打印
    )


# ═══════════════════════════════════════════════════════════════════
# 3. 多股票筛选主循环（支持并行 / 顺序两种模式）
# ═══════════════════════════════════════════════════════════════════

def _print_result_line(done: int, total: int, sym: str,
                       result: dict, elapsed: float,
                       lock: threading.Lock) -> None:
    """线程安全地打印单股结果行。"""
    if result.get('_skip'):
        err  = result.get('_error', '未知错误')
        line = f"  ✗ [{done:>3}/{total}] {sym:<8}  跳过  ({err})  [{elapsed:.1f}s]"
    else:
        tag  = '⚠放宽' if result.get('_relaxed') else '✓'
        line = (f"  ✓ [{done:>3}/{total}] {sym:<8}  "
                f"buy={result['buy_pct']:.0f}%/sell={result['sell_pct']:.0f}%  "
                f"年化={result['ann_ret']:>6.1f}%  "
                f"年度胜率={result['year_win_rate']:.0f}%  {tag}  [{elapsed:.1f}s]")
    with lock:
        print(line, flush=True)


def _progress_bar(done: int, total: int, workers: int,
                  elapsed: float, lock: threading.Lock) -> None:
    """在同一行刷新进度条（并行模式专用）。"""
    pct     = done / total * 100
    bar_len = 32
    filled  = int(bar_len * done / total)
    bar     = '█' * filled + '░' * (bar_len - filled)
    eta     = (elapsed / done * (total - done)) if done else 0
    line    = (f"\r  [{bar}] {pct:5.1f}%  {done}/{total}  "
               f"{workers}进程  耗时 {elapsed:.0f}s  预计剩余 {eta:.0f}s    ")
    with lock:
        print(line, end='', flush=True)


def screen_all_stocks(symbols: list[str], start: str, end: str,
                      buy_min: float, buy_max: float,
                      sell_min: float, sell_max: float,
                      step: float, cash: float,
                      min_trades: float, min_year_win: float,
                      min_data_years: float,
                      max_workers: int = 1) -> tuple[pd.DataFrame, list[str]]:
    """
    遍历所有股票，返回 (results_df, skipped_list)。
    max_workers=1  → 顺序模式（默认，兼容性最好）
    max_workers>1  → ProcessPoolExecutor 并行模式
    results_df 已按 composite_score 降序排列。
    """
    total   = len(symbols)
    rows    = []
    errors  = []
    t_all   = time.time()
    # 确定实际并行数（不超过股票数量）
    workers = min(max_workers, total)

    mode_str = f"{workers} 进程并行" if workers > 1 else "顺序执行"
    print(f"\n{'═'*70}")
    print(f"  开始扫描 — {total} 支股票  [{mode_str}]")
    print(f"  参数范围: buy {buy_min:.0f}%~{buy_max:.0f}%  "
          f"sell {sell_min:.0f}%~{sell_max:.0f}%  step {step}%")
    print(f"{'═'*70}\n")

    lock = threading.Lock()   # 保护 stdout，防止多线程输出乱序

    # ── 公共参数包（不含 symbol）────────────────────────────────
    common = (start, end,
              buy_min, buy_max, sell_min, sell_max,
              step, cash, min_trades, min_year_win, min_data_years)

    if workers <= 1:
        # ── 顺序模式 ─────────────────────────────────────────────
        for idx, sym in enumerate(symbols, 1):
            t1     = time.time()
            result = screen_one_stock(sym, *common, quiet=False)
            elapsed = time.time() - t1
            done    = idx
            if result.get('_skip'):
                errors.append(f"{sym}: {result.get('_error', '未知错误')}")
            else:
                rows.append(result)
            _print_result_line(done, total, sym, result, elapsed, lock)
    else:
        # ── 并行模式（ProcessPoolExecutor）────────────────────────
        packed_args = [(sym,) + common for sym in symbols]
        start_times = {}        # future → 提交时间
        done_count  = 0

        print(f"  启动进程池（最多 {workers} 进程）...\n")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_sym = {}
            for sym, args in zip(symbols, packed_args):
                fut = executor.submit(_parallel_worker, args)
                future_to_sym[fut] = sym
                start_times[fut]   = time.time()

            for fut in as_completed(future_to_sym):
                sym     = future_to_sym[fut]
                elapsed = time.time() - start_times[fut]
                done_count += 1

                try:
                    result = fut.result()
                except Exception as exc:
                    result = {'symbol': sym, '_error': str(exc), '_skip': True}

                if result.get('_skip'):
                    errors.append(f"{sym}: {result.get('_error', '未知错误')}")
                else:
                    rows.append(result)

                _print_result_line(done_count, total, sym, result, elapsed, lock)
                _progress_bar(done_count, total, workers,
                              time.time() - t_all, lock)

        print()   # 结束进度条行

    total_elapsed = time.time() - t_all
    print(f"\n  扫描完成: {len(rows)} 支有效 / {len(errors)} 支跳过  "
          f"总耗时 {total_elapsed:.1f}s"
          f"  平均 {total_elapsed/total:.1f}s/股\n")

    if not rows:
        raise ValueError("所有股票均无有效结果，请检查代码和日期范围")

    df = pd.DataFrame(rows)
    df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    return df, errors


# ═══════════════════════════════════════════════════════════════════
# 4. 可视化
# ═══════════════════════════════════════════════════════════════════

def plot_ranking_bars(results: pd.DataFrame, top_n: int) -> Path:
    """
    横向对比条形图：年化收益 / 最大回撤 / 夏普比率 / 年度胜率。
    高亮 Top N，灰显其余。
    """
    n   = min(len(results), 30)          # 最多显示30支
    sub = results.head(n).copy()
    sub['label'] = sub['symbol'] + '\n' + sub.apply(
        lambda r: f"b={r['buy_pct']:.0f}/s={r['sell_pct']:.0f}", axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'股票综合排名对比  (扫描 {len(results)} 支，展示前 {n} 支)',
                 fontsize=13, fontweight='bold')

    panels = [
        ('ann_ret',       '年化收益率 (%)', '#27ae60', '#a8d5b5'),
        ('max_dd',        '最大回撤 (%)',   '#c0392b', '#f5a9a9'),
        ('sharpe',        '夏普比率',       '#2980b9', '#a8c8f0'),
        ('year_win_rate', '年度胜率 (%)',   '#8e44ad', '#d7bde2'),
    ]

    for ax, (col, title, top_color, rest_color) in zip(axes.flat, panels):
        vals   = sub[col].values
        labels = sub['label'].values
        colors = [top_color if i < top_n else rest_color for i in range(len(sub))]

        bars = ax.barh(range(len(sub)), vals, color=colors, edgecolor='white',
                       linewidth=0.5)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.25)
        ax.axvline(0, color='black', lw=0.5)

        # 数值标注
        for i, (bar, v) in enumerate(zip(bars, vals)):
            rank_label = f'#{i+1}' if i < top_n else ''
            ax.text(v + abs(vals.max()) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.1f}  {rank_label}', va='center', fontsize=7.5,
                    fontweight='bold' if i < top_n else 'normal',
                    color='black')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = BACKTEST_DIR / f"screener_ranking_{TODAY}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表] 排名对比图   → {out}")
    return out


def plot_equity_comparison(results: pd.DataFrame, start: str, end: str,
                           top_n: int, cash: float) -> Path:
    """Top N 股票各自最优参数的净值曲线叠加对比。"""
    top = results.head(top_n)
    fig, ax = plt.subplots(figsize=(16, 8))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(top)))

    for i, (_, row) in enumerate(top.iterrows()):
        sym = row['symbol']
        try:
            df  = load_or_fetch(sym, start, end)
            res = run_backtest(df, row['buy_pct'], row['sell_pct'], cash)
            eq  = res['equity_df'].set_index('date')['equity']
            eq.index = pd.to_datetime(eq.index)
            nav = eq / cash

            label = (f"#{i+1} {sym}  buy={row['buy_pct']:.0f}%/sell={row['sell_pct']:.0f}%  "
                     f"年化={row['ann_ret']:.1f}%  回撤={row['max_dd']:.1f}%")
            ax.plot(nav.index, nav.values, color=cmap[i], lw=1.8,
                    label=label, alpha=0.88)
        except Exception as e:
            print(f"  [警告] {sym} 净值曲线绘制失败: {e}")
            continue

    ax.axhline(1.0, color='#aaaaaa', ls=':', lw=1)
    years = pd.date_range(f'{start[:4]}-01-01', f'{end[:4]}-12-31', freq='YS')
    for yr in years:
        ax.axvline(yr, color='#eeeeee', lw=0.8)

    ax.set_title(f'Top {top_n} 股票最优策略 净值曲线对比  {start[:4]}~{end[:4]}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('净值（初始=1）', fontsize=11)
    ax.set_xlabel('日期', fontsize=10)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()

    plt.tight_layout()
    out = BACKTEST_DIR / f"screener_equity_{TODAY}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表] 净值曲线图   → {out}")
    return out


def plot_yearly_matrix(results: pd.DataFrame, top_n: int) -> Path:
    """
    股票 × 年份 盈亏热力图。
    每格显示年度收益率，绿=盈，红=亏。
    """
    top = results.head(min(top_n * 2, len(results)))   # 最多展示 top_n*2 支

    mat_rows = []
    for _, row in top.iterrows():
        yr_dict = row.get('yearly_rets', {})
        if not yr_dict:
            continue
        entry = {'股票': row['symbol']}
        for yr, v in sorted(yr_dict.items()):
            entry[str(yr)] = round(v, 1)
        mat_rows.append(entry)

    if not mat_rows:
        return BACKTEST_DIR / 'screener_yearly_empty.png'

    mat_df = pd.DataFrame(mat_rows).set_index('股票')
    mat_df = mat_df.reindex(sorted(mat_df.columns), axis=1)

    # 行注解：添加综合评分
    row_labels = []
    for sym in mat_df.index:
        r = results[results['symbol'] == sym]
        if r.empty:
            row_labels.append(sym)
        else:
            r = r.iloc[0]
            row_labels.append(
                f"{sym}\nb={r['buy_pct']:.0f}/s={r['sell_pct']:.0f}\n"
                f"{r['ann_ret']:.1f}%/yr  胜率{r['year_win_rate']:.0f}%"
            )
    mat_df.index = row_labels

    cmap = LinearSegmentedColormap.from_list('rg', ['#b71c1c', '#ffffff', '#1b5e20'])
    vabs = max(abs(mat_df.values[~pd.isna(mat_df.values)].max()), 1)

    cell_h = max(0.6, 8 / len(mat_df))
    fig_h  = max(6, len(mat_df) * cell_h + 2)
    fig, ax = plt.subplots(figsize=(max(10, len(mat_df.columns) * 1.1), fig_h))

    sns.heatmap(
        mat_df.astype(float), ax=ax, cmap=cmap,
        center=0, vmin=-vabs, vmax=vabs,
        annot=True, fmt='.1f', linewidths=0.5, linecolor='#dddddd',
        annot_kws={'size': 9},
        cbar_kws={'label': '年度收益率 (%)', 'shrink': 0.75},
    )
    ax.set_title(f'股票 × 年份  年度收益热力图\n绿=盈利  红=亏损  数字单位: %',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('年份', fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    out = BACKTEST_DIR / f"screener_yearly_{TODAY}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表] 年度热力图   → {out}")
    return out


def plot_scatter_overview(results: pd.DataFrame, top_n: int) -> Path:
    """
    散点图总览：横轴=年化收益，纵轴=最大回撤，气泡大小=年度胜率，颜色=综合评分。
    Top N 用红星标注。
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    sc = ax.scatter(
        results['ann_ret'], results['max_dd'],
        s=results['year_win_rate'] * 3,
        c=results['composite_score'],
        cmap='RdYlGn', alpha=0.7, edgecolors='white', linewidth=0.5,
        zorder=3,
    )
    plt.colorbar(sc, ax=ax, label='综合评分', shrink=0.8)

    # 标注 Top N
    for i, (_, row) in enumerate(results.head(top_n).iterrows()):
        ax.scatter(row['ann_ret'], row['max_dd'],
                   s=200, marker='*', c='#c0392b', zorder=5, edgecolors='black')
        ax.annotate(
            f"#{i+1} {row['symbol']}\nb={row['buy_pct']:.0f}/s={row['sell_pct']:.0f}",
            (row['ann_ret'], row['max_dd']),
            textcoords='offset points', xytext=(8, 4),
            fontsize=7.5, color='#c0392b', fontweight='bold',
        )

    ax.set_xlabel('年化收益率 (%)', fontsize=11)
    ax.set_ylabel('最大回撤 (%)', fontsize=11)
    ax.set_title(f'股票综合选优散点图  (★=Top {top_n}  气泡大小=年度胜率)',
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()       # 回撤越小越好 → 越上方
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = BACKTEST_DIR / f"screener_scatter_{TODAY}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表] 散点总览图   → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════
# 5. 控制台报告
# ═══════════════════════════════════════════════════════════════════

def print_ranking_report(results: pd.DataFrame, errors: list[str],
                         top_n: int, start: str, end: str,
                         buy_min: float, buy_max: float,
                         sell_min: float, sell_max: float,
                         step: float, min_year_win: float) -> None:
    sep = '═' * 105
    print(f"\n{sep}")
    print(f"  股票综合选优排名  ——  振幅策略参数扫描")
    print(sep)
    print(f"  回测区间 : {start}~{end}")
    print(f"  参数范围 : buy {buy_min:.0f}%~{buy_max:.0f}%  "
          f"sell {sell_min:.0f}%~{sell_max:.0f}%  step {step}%")
    print(f"  稳定性过滤: 年度胜率≥{min_year_win:.0f}%")
    print(f"  有效股票 : {len(results)} 支  跳过: {len(errors)} 支")
    print(sep)

    hdr = (f"  {'#':>4} {'代码':>8} {'buy%':>5} {'sell%':>6} "
           f"{'年化%':>7} {'最大回撤':>8} {'夏普':>6} {'卡玛':>6} "
           f"{'年度胜率':>8} {'盈利年':>7} {'综合评分':>9} {'备注'}")
    print(hdr)
    print(f"  {'-'*100}")

    for rank, (_, row) in enumerate(results.iterrows(), 1):
        note = '⚠放宽' if row.get('_relaxed') else ''
        star = '★' if rank <= top_n else ' '
        print(f"  {star}{rank:>3} {row['symbol']:>8} {row['buy_pct']:>5.1f} {row['sell_pct']:>6.1f} "
              f"{row['ann_ret']:>7.2f} {row['max_dd']:>8.2f} "
              f"{row['sharpe']:>6.3f} {row['calmar']:>6.3f} "
              f"{row['year_win_rate']:>7.0f}% "
              f"{row['n_profit_years']:>4.0f}/{row['n_total_years']:>2.0f}年"
              f"  {row['composite_score']:>9.4f}  {note}")

    # 年度收益对比（Top N）
    top = results.head(top_n)
    all_years = sorted(set(
        yr for yrs in top['yearly_rets'] for yr in yrs.keys()
        if isinstance(yrs, dict)
    ))

    if all_years:
        print(f"\n  ── Top {top_n} 年度收益对比 (%) ──")
        yr_hdr = f"  {'#':>3} {'代码':>8} {'参数':>13} " + \
                 ''.join([f"{str(y):>7}" for y in all_years])
        print(yr_hdr)
        print(f"  {'-'*95}")
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            yr   = row.get('yearly_rets', {}) or {}
            param = f"b={row['buy_pct']:.0f}/s={row['sell_pct']:.0f}"
            cells = []
            for y in all_years:
                v = yr.get(y, yr.get(str(y), None))
                if v is None:
                    cells.append(f"{'—':>7}")
                else:
                    sign = '+' if v >= 0 else ''
                    cells.append(f"{sign}{v:>6.1f}")
            print(f"  {rank:>3} {row['symbol']:>8} {param:>13} " + ''.join(cells))

    # 跳过列表
    if errors:
        print(f"\n  ── 跳过列表 ──")
        for e in errors[:20]:
            print(f"    ✗ {e}")
        if len(errors) > 20:
            print(f"    ... 共 {len(errors)} 支")

    print(f"\n  ⚠ 风险提示: 回测结果不代表未来收益，A股受政策影响显著。")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════════
# 6. Markdown 报告 + CSV
# ═══════════════════════════════════════════════════════════════════

def save_markdown(results: pd.DataFrame, errors: list[str],
                  top_n: int, start: str, end: str,
                  buy_min: float, buy_max: float,
                  sell_min: float, sell_max: float,
                  step: float, min_year_win: float) -> Path:

    top = results.head(top_n)
    all_years = sorted(set(
        yr for yrs in top['yearly_rets'] for yr in yrs.keys()
        if isinstance(yrs, dict)
    ))

    lines = [
        "# 多股票振幅策略 选股综合报告",
        "",
        f"> **回测区间**: {start[:4]}-{start[4:6]}-{start[6:]} ~ "
        f"{end[:4]}-{end[4:6]}-{end[6:]}",
        f"> **参数范围**: buy {buy_min:.0f}%~{buy_max:.0f}%  |  "
        f"sell {sell_min:.0f}%~{sell_max:.0f}%  |  step {step}%",
        f"> **稳定性过滤**: 年度胜率 ≥ {min_year_win:.0f}%",
        f"> **有效股票**: {len(results)} 支  |  **跳过**: {len(errors)} 支",
        f"> **生成时间**: {TODAY[:4]}-{TODAY[4:6]}-{TODAY[6:]}",
        "",
        "---",
        "",
        "## 综合排名（最优参数 × 综合评分）",
        "",
        "| # | 代码 | buy% | sell% | 年化% | 最大回撤% | 夏普 | 卡玛 | 年度胜率 | 盈利年 | 综合评分 |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, (_, row) in enumerate(results.iterrows(), 1):
        star = '**★**' if rank <= top_n else str(rank)
        note = ' ⚠' if row.get('_relaxed') else ''
        lines.append(
            f"| {star} | **{row['symbol']}** | {row['buy_pct']:.1f} | {row['sell_pct']:.1f} "
            f"| **{row['ann_ret']:.2f}** | {row['max_dd']:.2f} "
            f"| {row['sharpe']:.3f} | {row['calmar']:.3f} "
            f"| **{row['year_win_rate']:.0f}%** "
            f"| {row['n_profit_years']:.0f}/{row['n_total_years']:.0f} "
            f"| **{row['composite_score']:.4f}**{note} |"
        )

    # 年度对比
    if all_years:
        yr_hdr = "| 代码 | 参数 |" + "".join([f" {y} |" for y in all_years]) + " 年度胜率 |"
        yr_sep = "|:---|:---|" + "---:|" * len(all_years) + "---:|"
        lines += ["", "---", "", "## Top N 年度收益明细 (%)", "", yr_hdr, yr_sep]

        for _, row in top.iterrows():
            yr    = row.get('yearly_rets', {}) or {}
            param = f"b={row['buy_pct']:.0f}/s={row['sell_pct']:.0f}"
            cells = [f"**{row['symbol']}**", param]
            for y in all_years:
                v = yr.get(y, yr.get(str(y), None))
                if v is None:
                    cells.append('—')
                elif v >= 0:
                    cells.append(f"**+{v:.1f}**")
                else:
                    cells.append(f"~~{v:.1f}~~")
            cells.append(f"**{row['year_win_rate']:.0f}%**")
            lines.append("| " + " | ".join(cells) + " |")

    # 跳过列表
    if errors:
        lines += ["", "---", "", "## 跳过股票", ""]
        for e in errors:
            lines.append(f"- ✗ `{e}`")

    lines += [
        "", "---", "",
        "> **风险提示**: 回测结果不代表未来收益，A股受政策影响显著，请谨慎参考。", "",
    ]

    out = BACKTEST_DIR / f"screener_report_{TODAY}.md"
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"[报告MD] → {out}")
    return out


def save_csv(results: pd.DataFrame) -> Path:
    drop_cols = ['yearly_rets', '_skip', '_error', '_relaxed']
    out = BACKTEST_DIR / f"screener_results_{TODAY}.csv"
    results.drop(columns=drop_cols, errors='ignore').to_csv(
        out, index=False, encoding='utf-8-sig')
    print(f"[CSV]    → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════
# 7. 主入口
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='多股票选股 Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grp = parser.add_argument_group('股票代码输入（至少选一种）')
    grp.add_argument('--symbols',    default='',
                     help='离散股票代码，逗号分隔，如 601288,600519,000001')
    grp.add_argument('--code-range', nargs=2, metavar=('FROM', 'TO'),
                     help='连续代码范围，如 --code-range 600000 600050')

    parser.add_argument('--start', required=True, help='回测开始 YYYYMMDD')
    parser.add_argument('--end',   required=True, help='回测结束 YYYYMMDD')

    parser.add_argument('--buy-min',  type=float, default=85.0)
    parser.add_argument('--buy-max',  type=float, default=98.0)
    parser.add_argument('--sell-min', type=float, default=102.0)
    parser.add_argument('--sell-max', type=float, default=120.0)
    parser.add_argument('--step',     type=float, default=2.0,
                        help='参数步长，默认2.0（多股模式建议≥2，节省时间）')

    parser.add_argument('--min-year-win',   type=float, default=70.0,
                        help='年度胜率过滤阈值 %% (默认 70)')
    parser.add_argument('--min-trades',     type=float, default=1.0,
                        help='年均最低交易数 (默认 1.0)')
    parser.add_argument('--min-data-years', type=float, default=3.0,
                        help='最少数据年限，不足则跳过 (默认 3.0)')
    parser.add_argument('--cash',           type=float, default=100_000.0)
    parser.add_argument('--top-n',          type=int,   default=10,
                        help='推荐 Top N 支股票 (默认 10)')
    parser.add_argument('--workers',        type=int,   default=1,
                        help=('并行进程数 (默认 1=顺序)。'
                              '建议设为 CPU核数-1，如 --workers 4。'
                              '注意: Windows 并行模式须在终端直接运行，不可在 IDLE/Jupyter 中使用'))

    args = parser.parse_args()

    # 自动将 workers=0 解释为 "CPU核数-1"
    if args.workers == 0:
        args.workers = max(1, (os.cpu_count() or 4) - 1)

    symbols = build_symbol_list(args.symbols, args.code_range)
    if not symbols:
        parser.error('请至少提供 --symbols 或 --code-range 之一')

    print(f"\n{'═'*70}")
    print(f"  多股票选股 Pipeline")
    print(f"  输入股票: {len(symbols)} 支  "
          f"{'  '.join(symbols[:8])}{'...' if len(symbols) > 8 else ''}")
    print(f"  并行模式: {'顺序' if args.workers == 1 else f'{args.workers} 进程'}")
    print(f"{'═'*70}")

    # ── 扫描 ──
    results, errors = screen_all_stocks(
        symbols, args.start, args.end,
        args.buy_min, args.buy_max,
        args.sell_min, args.sell_max,
        args.step, args.cash,
        args.min_trades, args.min_year_win,
        args.min_data_years,
        max_workers=args.workers,
    )

    # ── 控制台报告 ──
    print_ranking_report(
        results, errors, args.top_n,
        args.start, args.end,
        args.buy_min, args.buy_max,
        args.sell_min, args.sell_max,
        args.step, args.min_year_win,
    )

    # ── 可视化 ──
    print("[Step] 生成图表 ...")
    plot_ranking_bars(results, args.top_n)
    plot_scatter_overview(results, args.top_n)
    plot_equity_comparison(results, args.start, args.end,
                           min(args.top_n, len(results)), args.cash)
    plot_yearly_matrix(results, args.top_n)

    # ── 保存文件 ──
    print("[Step] 保存报告与数据 ...")
    save_markdown(results, errors, args.top_n,
                  args.start, args.end,
                  args.buy_min, args.buy_max,
                  args.sell_min, args.sell_max,
                  args.step, args.min_year_win)
    save_csv(results)

    print(f"\n{'═'*70}")
    print(f"  完成！输出目录: {BACKTEST_DIR}")
    print(f"{'═'*70}\n")


if __name__ == '__main__':
    main()
