#!/usr/bin/env python3
"""
振幅策略 大型参数扫描 Pipeline
====================================
策略: 从峰值回调 buy_pct% 买入，涨至入场峰值 sell_pct% 卖出

全面扫描 buy_pct × sell_pct 参数空间，综合评分后推荐：
  ① 连年不亏损（年度胜率高，稳定性优先）
  ② 综合收益率最高
的 Top 10 参数组合，并输出年度明细 + 可视化报告

用法:
  # 默认扫描 (buy 80~99, sell 101~130)
  python param_sweep_pipeline.py --symbol 601288 --start 20150101 --end 20241231

  # 自定义范围
  python param_sweep_pipeline.py --symbol 601288 --start 20150101 --end 20241231 \\
      --buy-min 85 --buy-max 98 --sell-min 102 --sell-max 120 --step 1

  # 多股票（等权合成后扫描）
  python param_sweep_pipeline.py --symbols 601288,600519,000001 --start 20150101 --end 20241231
"""

import argparse
import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import seaborn as sns

# 复用已有模块
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from amplitude.backtest import run_backtest, calc_metrics, DATA_DIR, BACKTEST_DIR
from core.fetch import fetch_daily

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BUY_FEE  = 0.001
SELL_FEE = 0.002


# ═══════════════════════════════════════════════════════════════════
# 1. 数据层
# ═══════════════════════════════════════════════════════════════════

def load_or_fetch(symbol: str, start: str, end: str,
                  quiet: bool = False) -> pd.DataFrame:
    """优先读取本地 data/ 缓存，否则通过 akshare 拉取。quiet=True 时不打印。"""
    start_dt = pd.to_datetime(start, format='%Y%m%d')
    end_dt   = pd.to_datetime(end,   format='%Y%m%d')
    best_df  = None
    for fpath in sorted(DATA_DIR.glob(f"{symbol}_daily_*_*_qfq.csv")):
        df  = pd.read_csv(fpath, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'])
        sub = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
        if not sub.empty and (best_df is None or len(sub) > len(best_df)):
            best_df = sub
    if best_df is not None:
        best_df = best_df.sort_values('date').reset_index(drop=True)
        if not quiet:
            print(f"[数据] {symbol} 本地缓存 {len(best_df)} 行  "
                  f"{best_df['date'].iloc[0].date()} ~ {best_df['date'].iloc[-1].date()}")
        return best_df
    if not quiet:
        print(f"[数据] {symbol} 本地无缓存，自动拉取 ...")
    df    = fetch_daily(symbol, start, end, 'qfq')
    fpath = DATA_DIR / f"{symbol}_daily_{start}_{end}_qfq.csv"
    df.to_csv(fpath, index=False, encoding='utf-8-sig')
    if not quiet:
        print(f"[数据] 已保存 {len(df)} 行 → {fpath}")
    return df


def build_composite_price(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """
    多股等权合成：各股归一化后取算术平均，
    合成价格序列作为单股传入回测引擎。
    """
    frames = {}
    for sym in symbols:
        try:
            df = load_or_fetch(sym, start, end)
            df = df.sort_values('date').reset_index(drop=True)
            frames[sym] = df.set_index('date')[['open','high','low','close','volume']]
        except Exception as e:
            print(f"  [警告] {sym} 加载失败: {e}，跳过")

    if not frames:
        raise ValueError("所有股票加载失败")

    if len(frames) == 1:
        sym = list(frames.keys())[0]
        df  = frames[sym].reset_index()
        return df

    # 以最短序列为准对齐
    common_idx = None
    for df in frames.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)

    normed = {}
    for sym, df in frames.items():
        sub = df.loc[common_idx].copy()
        base = sub['close'].iloc[0]
        normed[sym] = sub / base  # 归一化

    # 等权平均各列
    combined = sum(normed.values()) / len(normed)
    combined = combined.reset_index()
    combined['date'] = pd.to_datetime(combined['date'])
    return combined.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# 2. 单次扫描（轻量，含年度明细）
# ═══════════════════════════════════════════════════════════════════

def _sweep_one(df: pd.DataFrame, buy: float, sell: float,
               cash: float = 100_000.0) -> dict:
    """
    对单个 (buy_pct, sell_pct) 运行回测，返回完整指标 + 年度明细。
    与 amplitude_pipeline._run_one 相比新增：
      - yearly_rets   : {year: ret%}
      - year_win_rate : 年度盈利比例
      - n_profit_years: 盈利年数
      - n_total_years : 总年数
    """
    result = run_backtest(df, buy, sell, cash)
    m      = calc_metrics(result, df)

    sells_df = result['trades_df']
    sells_df = sells_df[sells_df['action'] == 'SELL'] if not sells_df.empty else sells_df
    if len(sells_df) > 0:
        tw  = sells_df.loc[sells_df['profit_pct'] > 0, 'profit_pct'].sum()
        tl  = abs(sells_df.loc[sells_df['profit_pct'] <= 0, 'profit_pct'].sum())
        pf  = round(tw / tl, 3) if tl > 0 else float('inf')
    else:
        pf = 0.0

    yr      = m['yearly_rets']            # {year: ret%}
    yr_vals = list(yr.values())
    n_total  = len(yr_vals)
    n_profit = sum(1 for v in yr_vals if v > 0)
    yr_win   = round(n_profit / n_total * 100, 1) if n_total > 0 else 0.0

    return {
        'buy_pct':        buy,
        'sell_pct':       sell,
        'ann_ret':        m['ann_ret'],
        'total_ret':      m['total_ret'],
        'max_dd':         m['max_dd'],
        'sharpe':         m['sharpe'],
        'calmar':         m['calmar'],
        'win_rate':       m['win_rate'],
        'avg_profit':     m['avg_profit'],
        'n_trades':       m['n_trades'],
        'ann_trades':     round(m['n_trades'] / max(m['years'], 0.01), 2),
        'years':          m['years'],
        'bh_ret':         m['bh_ret'],
        'bh_ann':         m['bh_ann'],
        'profit_factor':  pf,
        'yearly_rets':    yr,
        'year_win_rate':  yr_win,
        'n_profit_years': n_profit,
        'n_total_years':  n_total,
    }


# ═══════════════════════════════════════════════════════════════════
# 3. 参数扫描主循环
# ═══════════════════════════════════════════════════════════════════

def parameter_sweep(df: pd.DataFrame,
                    buy_min: float, buy_max: float,
                    sell_min: float, sell_max: float,
                    step: float,
                    cash: float,
                    verbose: bool = True) -> pd.DataFrame:
    """
    遍历所有 (buy_pct, sell_pct) 组合，返回完整结果 DataFrame。
    自动跳过 buy_pct >= sell_pct 的无效组合。
    """
    buy_vals  = [round(v, 2) for v in np.arange(buy_min,  buy_max  + step/2, step)]
    sell_vals = [round(v, 2) for v in np.arange(sell_min, sell_max + step/2, step)]
    combos    = [(b, s) for b, s in product(buy_vals, sell_vals) if b < s]
    total     = len(combos)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  参数扫描: buy {buy_min}%~{buy_max}%  sell {sell_min}%~{sell_max}%  step {step}%")
        print(f"  共 {len(buy_vals)} × {len(sell_vals)} = {total} 个有效组合")
        print(f"{'='*70}")

    rows = []
    t0   = time.time()
    mile = max(1, total // 40)

    for i, (buy, sell) in enumerate(combos, 1):
        rows.append(_sweep_one(df, buy, sell, cash))
        if verbose and (i % mile == 0 or i == total):
            elapsed  = time.time() - t0
            eta      = elapsed / i * (total - i)
            pct      = i / total * 100
            bar_len  = 30
            done     = int(bar_len * i / total)
            bar      = '█' * done + '░' * (bar_len - done)
            print(f"\r  [{bar}] {pct:5.1f}%  {i}/{total}  "
                  f"耗时 {elapsed:.0f}s  预计剩余 {eta:.0f}s    ",
                  end='', flush=True)

    if verbose:
        print(f"\n\n  扫描完成！共 {len(rows)} 个结果，用时 {time.time()-t0:.1f}s\n")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# 4. 综合评分与 Top 10 筛选
# ═══════════════════════════════════════════════════════════════════

def score_and_top10(results: pd.DataFrame,
                    min_trades_per_year: float = 1.0,
                    min_year_win_rate: float = 70.0,
                    weights: dict | None = None,
                    top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    两步筛选：
      Step A (硬性过滤):
        - 年均交易数 >= min_trades_per_year
        - 年度胜率   >= min_year_win_rate%  (即至少 X% 的年份为正收益)
      Step B (综合评分 Min-Max 归一化后加权):
        - 年化收益率  (35%)
        - 年度胜率    (30%)  ← 稳定性核心
        - 卡玛比率    (20%)
        - 夏普比率    (15%)

    返回: (stable_df, top10_df)  前者是过滤后全集，后者是 Top N
    """
    if weights is None:
        weights = {
            'ann_ret':       0.35,
            'year_win_rate': 0.30,
            'calmar':        0.20,
            'sharpe':        0.15,
        }

    # ── Step A: 硬性过滤 ──
    stable = results[
        (results['ann_trades']    >= min_trades_per_year) &
        (results['year_win_rate'] >= min_year_win_rate)
    ].copy()

    if stable.empty:
        # 放宽：只保年度胜率条件
        print(f"  [警告] 年均交易数≥{min_trades_per_year} 且 年度胜率≥{min_year_win_rate}% 无满足组合")
        print(f"         自动放宽：仅保留年度胜率≥{min_year_win_rate}% 的组合")
        stable = results[results['year_win_rate'] >= min_year_win_rate].copy()

    if stable.empty:
        best_yr  = results['year_win_rate'].max()
        fallback = max(best_yr - 10, 0)
        print(f"         再次放宽：年度胜率≥{fallback:.0f}%（当前最高={best_yr:.1f}%）")
        stable = results[results['year_win_rate'] >= fallback].copy()

    if stable.empty:
        print(f"         全部放宽，取全量结果")
        stable = results.copy()

    # ── Step B: 归一化评分 ──
    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)

    stable['_s_ann']  = _norm(stable['ann_ret'])
    stable['_s_yr']   = _norm(stable['year_win_rate'])
    stable['_s_cal']  = _norm(stable['calmar'])
    stable['_s_shr']  = _norm(stable['sharpe'])

    stable['composite_score'] = (
        weights['ann_ret']       * stable['_s_ann'] +
        weights['year_win_rate'] * stable['_s_yr']  +
        weights['calmar']        * stable['_s_cal'] +
        weights['sharpe']        * stable['_s_shr']
    ).round(4)

    stable = stable.drop(columns=['_s_ann','_s_yr','_s_cal','_s_shr'])
    stable = stable.sort_values('composite_score', ascending=False).reset_index(drop=True)
    top10  = stable.head(top_n).copy()

    return stable, top10


# ═══════════════════════════════════════════════════════════════════
# 5. 可视化
# ═══════════════════════════════════════════════════════════════════

def _equity_from_row(df: pd.DataFrame, row: pd.Series, cash: float) -> pd.Series:
    res = run_backtest(df, row['buy_pct'], row['sell_pct'], cash)
    eq  = res['equity_df'].set_index('date')['equity']
    eq.index = pd.to_datetime(eq.index)
    return eq / cash  # 净值


def plot_top10_equity(df: pd.DataFrame, top10: pd.DataFrame,
                      symbol: str, start: str, end: str,
                      cash: float = 100_000.0) -> Path:
    """Top 10 策略净值曲线 + 买持基准对比。"""
    bh_close = df.set_index('date')['close']
    bh_close.index = pd.to_datetime(bh_close.index)
    bh_nav   = bh_close / float(bh_close.iloc[0])

    fig, ax = plt.subplots(figsize=(16, 8))
    cmap    = plt.cm.tab10(np.linspace(0, 1, len(top10)))

    for i, (_, row) in enumerate(top10.iterrows()):
        nav   = _equity_from_row(df, row, cash)
        label = (f"#{i+1} buy={row['buy_pct']:.0f}% sell={row['sell_pct']:.0f}%  "
                 f"年化={row['ann_ret']:.1f}%  回撤={row['max_dd']:.1f}%  "
                 f"年胜率={row['year_win_rate']:.0f}%")
        ax.plot(nav.index, nav.values, color=cmap[i], lw=1.6,
                label=label, alpha=0.85, zorder=3)

    ax.plot(bh_nav.index, bh_nav.values, 'k--', lw=2.2, label='买持基准', alpha=0.6, zorder=2)
    ax.axhline(1.0, color='#aaaaaa', ls=':', lw=1)

    # 标注年份边界
    years = pd.date_range(f'{start[:4]}-01-01', f'{end[:4]}-12-31', freq='YS')
    for yr in years:
        ax.axvline(yr, color='#dddddd', lw=0.8, zorder=1)

    ax.set_title(f'{symbol}  振幅策略 Top 10 净值曲线  {start[:4]}~{end[:4]}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('净值（初始=1）', fontsize=11)
    ax.set_xlabel('日期', fontsize=10)
    ax.legend(loc='upper left', fontsize=8, ncol=1, framealpha=0.85)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()

    out = BACKTEST_DIR / f"{symbol}_top10_equity_{start}_{end}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表] Top10 净值曲线 → {out}")
    return out


def plot_yearly_heatmap(top10: pd.DataFrame, symbol: str,
                        start: str, end: str) -> Path:
    """Top 10 各策略 年度收益热力图（绿=盈利 红=亏损）。"""
    # 整理年度收益矩阵
    yearly_rows = []
    for i, (_, row) in enumerate(top10.iterrows()):
        yr_dict = row['yearly_rets']
        label   = f"#{i+1} b={row['buy_pct']:.0f}/s={row['sell_pct']:.0f}"
        entry   = {'策略': label}
        for yr, ret in sorted(yr_dict.items()):
            entry[str(yr)] = round(ret, 1)
        yearly_rows.append(entry)

    mat_df = pd.DataFrame(yearly_rows).set_index('策略')
    mat_df = mat_df[sorted(mat_df.columns)]          # 年份升序

    cmap = LinearSegmentedColormap.from_list('rg', ['#b71c1c', '#ffffff', '#1b5e20'])
    vmax = max(abs(mat_df.values[~np.isnan(mat_df.values.astype(float))].max()), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(mat_df.columns) * 1.0), 6))
    sns.heatmap(
        mat_df.astype(float), ax=ax, cmap=cmap,
        center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt='.1f', linewidths=0.6,
        linecolor='#eeeeee',
        annot_kws={'size': 9},
        cbar_kws={'label': '年度收益率 (%)', 'shrink': 0.8},
    )
    ax.set_title(f'{symbol}  Top 10 参数  年度收益热力图  {start[:4]}~{end[:4]}\n'
                 f'绿=盈利  红=亏损  数字单位: %',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('年份', fontsize=10)
    ax.set_ylabel('策略参数', fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    out = BACKTEST_DIR / f"{symbol}_top10_yearly_{start}_{end}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表] 年度热力图   → {out}")
    return out


def plot_param_heatmap(results: pd.DataFrame,
                       top10: pd.DataFrame,
                       symbol: str, start: str, end: str) -> Path:
    """
    4 宫格参数热力图：年化收益 / 年度胜率 / 卡玛比率 / 综合评分。
    Top 10 位置用 ★ 标注。
    """
    buy_vals  = sorted(results['buy_pct'].unique())
    sell_vals = sorted(results['sell_pct'].unique())

    top10_pairs = set(zip(top10['buy_pct'], top10['sell_pct']))

    panels = [
        ('ann_ret',        '年化收益率 (%)',   'RdYlGn',   '{:.1f}'),
        ('year_win_rate',  '年度胜率 (%)',     'RdYlGn',   '{:.0f}'),
        ('calmar',         '卡玛比率',         'RdYlGn',   '{:.2f}'),
        ('composite_score','综合评分',         'RdYlGn',   '{:.3f}'),
    ]
    # composite_score 仅存在于 stable_df 子集，对 full results 补 NaN
    if 'composite_score' not in results.columns:
        results = results.copy()
        results['composite_score'] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{symbol}  振幅策略参数热力图  {start[:4]}~{end[:4]}\n'
                 f'★ = Top 10 推荐参数', fontsize=13, fontweight='bold')

    for ax, (col, title, cmap_name, fmt) in zip(axes.flat, panels):
        mat = np.full((len(buy_vals), len(sell_vals)), np.nan)
        for i, b in enumerate(buy_vals):
            for j, s in enumerate(sell_vals):
                sub = results[(results['buy_pct'] == b) & (results['sell_pct'] == s)]
                if not sub.empty and col in sub.columns:
                    mat[i, j] = sub.iloc[0][col]

        im = ax.imshow(mat[::-1], cmap=cmap_name, aspect='auto',
                       vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_xticks(range(len(sell_vals)))
        ax.set_yticks(range(len(buy_vals)))
        ax.set_xticklabels([f'{v:.0f}' for v in sell_vals],
                           fontsize=7, rotation=45)
        ax.set_yticklabels([f'{v:.0f}' for v in buy_vals[::-1]], fontsize=7)
        ax.set_xlabel('sell_pct (%)', fontsize=9)
        ax.set_ylabel('buy_pct (%)', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

        # 单元格文字 + ★ 标注
        for i, b in enumerate(buy_vals[::-1]):
            for j, s in enumerate(sell_vals):
                sub = results[(results['buy_pct'] == b) & (results['sell_pct'] == s)]
                if sub.empty:
                    continue
                val = sub.iloc[0][col]
                if np.isnan(val):
                    continue
                is_top = (b, s) in top10_pairs
                txt  = ('★' if is_top else '') + fmt.format(val)
                fc   = 'black'
                fw   = 'bold' if is_top else 'normal'
                fs   = 7 if len(buy_vals) <= 20 else 6
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=fs, color=fc, fontweight=fw,
                        alpha=1.0 if is_top else 0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = BACKTEST_DIR / f"{symbol}_sweep_heatmap_{start}_{end}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表] 参数热力图   → {out}")
    return out


def plot_best_detail(df: pd.DataFrame, best_row: pd.Series,
                     symbol: str, start: str, end: str,
                     cash: float = 100_000.0) -> Path:
    """最优参数详细分析图：净值 + 回撤 + 月度分布 + 年度柱状图。"""
    buy, sell = best_row['buy_pct'], best_row['sell_pct']
    result    = run_backtest(df, buy, sell, cash)
    m         = calc_metrics(result, df)
    eq        = m['eq_series']
    nav       = eq / cash

    bh_close  = df.set_index('date')['close']
    bh_close.index = pd.to_datetime(bh_close.index)
    bh_nav    = bh_close / float(bh_close.iloc[0])

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 净值曲线 ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(nav.index, nav.values, '#c0392b', lw=2, label=f'振幅策略  年化={m["ann_ret"]:.1f}%')
    ax1.plot(bh_nav.index, bh_nav.values, '#2980b9', lw=1.5, ls='--',
             label=f'买持基准  年化={m["bh_ann"]:.1f}%', alpha=0.8)
    ax1.fill_between(nav.index, nav.values, bh_nav.values,
                     where=nav.values >= bh_nav.values,
                     alpha=0.12, color='green', label='超额')
    ax1.fill_between(nav.index, nav.values, bh_nav.values,
                     where=nav.values < bh_nav.values,
                     alpha=0.12, color='red', label='跑输')
    ax1.axhline(1.0, color='#aaaaaa', ls=':', lw=0.8)
    ax1.set_title(f'[最优参数] {symbol}  buy={buy:.1f}%  sell={sell:.1f}%  '
                  f'年化={m["ann_ret"]:.1f}%  回撤={m["max_dd"]:.1f}%  '
                  f'夏普={m["sharpe"]:.2f}  年度胜率={best_row["year_win_rate"]:.0f}%',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('净值')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()

    # ── 回撤曲线 ──
    ax2 = fig.add_subplot(gs[1, 0])
    dd  = m['dd_series'] * 100
    ax2.fill_between(dd.index, dd.values, 0, color='#c0392b', alpha=0.65)
    ax2.set_title(f'水下曲线  最大回撤: {m["max_dd"]:.1f}%', fontsize=10)
    ax2.set_ylabel('回撤 (%)')
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ── 月度收益分布直方图 ──
    ax3 = fig.add_subplot(gs[1, 1])
    mr  = pd.Series(list(m['monthly_rets'].values()))
    ax3.hist(mr, bins=28, color='#2980b9', alpha=0.75, edgecolor='white')
    ax3.axvline(0,      color='#c0392b', ls='--', lw=1.5, label='零线')
    ax3.axvline(mr.mean(), color='#27ae60', ls='--', lw=1.5,
                label=f'均值={mr.mean():.2f}%')
    ax3.set_title(f'月度收益分布  胜率={len(mr[mr>0])}/{len(mr)}月 ({len(mr[mr>0])/len(mr)*100:.0f}%)',
                  fontsize=10)
    ax3.set_xlabel('月收益率 (%)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # ── 年度收益柱状图 ──
    ax4 = fig.add_subplot(gs[2, 0])
    yr  = m['yearly_rets']
    yrs = [str(k) for k in sorted(yr.keys())]
    vals= [yr[int(k)] if int(k) in yr else yr[k] for k in sorted(yr.keys())]
    colors_bar = ['#27ae60' if v >= 0 else '#c0392b' for v in vals]
    bars = ax4.bar(yrs, vals, color=colors_bar, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3 * np.sign(v),
                 f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top',
                 fontsize=8)
    ax4.axhline(0, color='black', lw=0.8)
    ax4.set_title(f'年度收益  年度胜率={best_row["n_profit_years"]:.0f}/{best_row["n_total_years"]:.0f}年',
                  fontsize=10)
    ax4.set_ylabel('年度收益率 (%)')
    ax4.tick_params(axis='x', rotation=30)
    ax4.grid(True, alpha=0.2, axis='y')

    # ── 绩效摘要表 ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    rows_tbl = [
        ['指标',        '策略值',                  '买持基准'],
        ['年化收益率',  f"{m['ann_ret']:.2f}%",    f"{m['bh_ann']:.2f}%"],
        ['总收益率',    f"{m['total_ret']:.2f}%",  f"{m['bh_ret']:.2f}%"],
        ['最大回撤',    f"{m['max_dd']:.2f}%",     '—'],
        ['夏普比率',    f"{m['sharpe']:.3f}",      '—'],
        ['卡玛比率',    f"{m['calmar']:.3f}",      '—'],
        ['胜率 (交易)', f"{m['win_rate']:.1f}%",   '—'],
        ['年度胜率',    f"{best_row['year_win_rate']:.0f}%", '—'],
        ['年均交易',    f"{best_row['ann_trades']:.1f}笔",  '—'],
        ['综合评分',    f"{best_row['composite_score']:.4f}", '—'],
    ]
    tbl = ax5.table(cellText=rows_tbl[1:], colLabels=rows_tbl[0],
                    loc='center', cellLoc='center', colWidths=[0.38, 0.31, 0.31])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.1, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f2f2f2')
    ax5.set_title('绩效摘要', fontsize=11, fontweight='bold', pad=8)

    out = BACKTEST_DIR / f"{symbol}_best_detail_{buy}_{sell}_{start}_{end}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表] 最优策略详情 → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════
# 6. 报告输出
# ═══════════════════════════════════════════════════════════════════

def print_console_report(stable: pd.DataFrame, top10: pd.DataFrame,
                         results: pd.DataFrame,
                         symbol: str, start: str, end: str,
                         buy_min: float, buy_max: float,
                         sell_min: float, sell_max: float,
                         step: float, min_year_win: float,
                         min_trades: float) -> None:
    sep = '═' * 100
    print(f"\n{sep}")
    print(f"  TOP 10 推荐参数  ——  {symbol}  振幅策略大型参数扫描")
    print(sep)
    print(f"  扫描范围 : buy {buy_min:.0f}%~{buy_max:.0f}%  sell {sell_min:.0f}%~{sell_max:.0f}%  step {step}%")
    print(f"  回测区间 : {start}~{end}   "
          f"买持基准年化 = {results['bh_ann'].iloc[0]:.2f}%")
    print(f"  稳定性过滤: 年度胜率≥{min_year_win:.0f}%  年均交易数≥{min_trades:.1f}")
    print(f"  全量组合 : {len(results)}  过滤后 : {len(stable)}")
    print(sep)

    hdr = (f"  {'#':>3} {'buy%':>5} {'sell%':>6} "
           f"{'年化%':>7} {'最大回撤':>8} {'夏普':>6} {'卡玛':>6} "
           f"{'年度胜率':>8} {'交易次数':>8} {'盈亏比':>7} {'综合评分':>9}")
    print(hdr)
    print(f"  {'-'*97}")

    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        pf = row['profit_factor']
        pf_str = f"{pf:>7.2f}" if pf != float('inf') else "    inf"
        print(f"  {rank:>3} {row['buy_pct']:>5.1f} {row['sell_pct']:>6.1f} "
              f"{row['ann_ret']:>7.2f} {row['max_dd']:>8.2f} "
              f"{row['sharpe']:>6.3f} {row['calmar']:>6.3f} "
              f"{row['year_win_rate']:>7.0f}% "
              f"{row['n_trades']:>7.0f} "
              f"{pf_str} {row['composite_score']:>9.4f}")

    best = top10.iloc[0]
    print(f"\n{sep}")
    print(f"  ★★ 最优推荐: buy={best['buy_pct']:.1f}%  sell={best['sell_pct']:.1f}%  "
          f"年化={best['ann_ret']:.2f}%  回撤={best['max_dd']:.2f}%  "
          f"年度胜率={best['year_win_rate']:.0f}%  综合评分={best['composite_score']:.4f}")
    print(sep)

    # 年度明细展示（Top 10 的年份对比）
    print(f"\n  ── 年度收益对比 (%) ──")
    all_years = sorted(set(
        yr for row in top10['yearly_rets'] for yr in row.keys()
    ))
    yr_header = f"  {'#':>3} {'参数':>14} " + ' '.join([f"{str(y):>7}" for y in all_years])
    print(yr_header)
    print(f"  {'-'*90}")
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        yr = row['yearly_rets']
        yr_cells = []
        for y in all_years:
            v = yr.get(y, yr.get(str(y), None))
            if v is None:
                yr_cells.append(f"{'—':>7}")
            else:
                sign = '+' if v >= 0 else ''
                yr_cells.append(f"{sign}{v:>6.1f}")
        param = f"b={row['buy_pct']:.0f}/s={row['sell_pct']:.0f}"
        print(f"  {rank:>3} {param:>14} " + ' '.join(yr_cells))
    print(f"\n  ⚠ 风险提示: 回测结果不代表未来收益，A股受政策影响显著。\n")


def save_markdown_report(stable: pd.DataFrame, top10: pd.DataFrame,
                         results: pd.DataFrame,
                         symbol: str, start: str, end: str,
                         buy_min: float, buy_max: float,
                         sell_min: float, sell_max: float,
                         step: float, min_year_win: float,
                         min_trades: float,
                         weights: dict) -> Path:
    """生成完整的 Markdown 报告。"""
    bh_ann = results['bh_ann'].iloc[0] if not results.empty else 0
    all_years = sorted(set(
        yr for row in top10['yearly_rets'] for yr in row.keys()
    ))

    lines = [
        f"# {symbol} 振幅策略 大型参数扫描报告",
        "",
        f"> **扫描范围**: buy {buy_min:.0f}%~{buy_max:.0f}%  |  sell {sell_min:.0f}%~{sell_max:.0f}%  |  step {step}%",
        f"> **回测区间**: {start[:4]}-{start[4:6]}-{start[6:]} ~ {end[:4]}-{end[4:6]}-{end[6:]}",
        f"> **稳定性过滤**: 年度胜率 ≥ {min_year_win:.0f}%  |  年均交易数 ≥ {min_trades:.1f}",
        f"> **评分权重**: 年化收益 {weights['ann_ret']:.0%} / 年度胜率 {weights['year_win_rate']:.0%} / 卡玛 {weights['calmar']:.0%} / 夏普 {weights['sharpe']:.0%}",
        f"> **全量组合**: {len(results)} 组  |  **过滤后**: {len(stable)} 组  |  **买持基准年化**: {bh_ann:.2f}%",
        "",
        "---",
        "",
        "## Top 10 推荐参数（稳定性优先 + 收益最高）",
        "",
        "| # | buy% | sell% | 年化% | 最大回撤% | 夏普 | 卡玛 | 年度胜率 | 盈利年/总年 | 交易数 | 综合评分 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        pf_str = f"{row['profit_factor']:.2f}" if row['profit_factor'] != float('inf') else "inf"
        star   = '**★**' if rank == 1 else str(rank)
        lines.append(
            f"| {star} | **{row['buy_pct']:.1f}** | **{row['sell_pct']:.1f}** "
            f"| **{row['ann_ret']:.2f}** | {row['max_dd']:.2f} "
            f"| {row['sharpe']:.3f} | {row['calmar']:.3f} "
            f"| **{row['year_win_rate']:.0f}%** "
            f"| {row['n_profit_years']:.0f}/{row['n_total_years']:.0f} "
            f"| {row['n_trades']:.0f} | **{row['composite_score']:.4f}** |"
        )

    # 年度收益对比表
    yr_hdr = "| 策略 |" + "".join([f" {y} |" for y in all_years]) + " 年度胜率 |"
    yr_sep = "|:---|" + "---:|" * len(all_years) + "---:|"
    lines += ["", "---", "", "## 年度收益明细对比 (%)", "", yr_hdr, yr_sep]

    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        yr    = row['yearly_rets']
        cells = [f"#{rank} b={row['buy_pct']:.0f}/s={row['sell_pct']:.0f}"]
        for y in all_years:
            v = yr.get(y, yr.get(str(y), None))
            if v is None:
                cells.append('—')
            else:
                if v >= 0:
                    cells.append(f"**+{v:.1f}**")
                else:
                    cells.append(f"~~{v:.1f}~~")
        cells.append(f"**{row['year_win_rate']:.0f}%**")
        lines.append("| " + " | ".join(cells) + " |")

    # 全量扫描统计
    lines += [
        "", "---", "",
        "## 全量扫描统计概览", "",
        "| 指标 | 均值 | 中位数 | 最大 | 最小 |",
        "|:---|---:|---:|---:|---:|",
    ]
    for label, col in [('年化收益率(%)', 'ann_ret'), ('最大回撤(%)', 'max_dd'),
                        ('夏普比率', 'sharpe'), ('卡玛比率', 'calmar'),
                        ('年度胜率(%)', 'year_win_rate'), ('年均交易数', 'ann_trades')]:
        s = results[col]
        lines.append(f"| {label} | {s.mean():.2f} | {s.median():.2f} | {s.max():.2f} | {s.min():.2f} |")

    profitable = (results['ann_ret'] > 0).sum()
    beat_bh    = (results['ann_ret'] > results['bh_ann']).sum()
    lines += [
        "",
        f"- **盈利组合**: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)",
        f"- **跑赢买持**: {beat_bh}/{len(results)} ({beat_bh/len(results)*100:.0f}%)",
        "", "---", "",
        "> **风险提示**: 以上回测结果不代表未来收益，A股受政策影响显著，请谨慎参考。",
        "",
    ]

    out = BACKTEST_DIR / f"{symbol}_sweep_top10_{start}_{end}.md"
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"[报告MD] → {out}")
    return out


def save_csv(stable: pd.DataFrame, top10: pd.DataFrame,
             symbol: str, start: str, end: str) -> None:
    """保存 Top10 和全量稳定组合 CSV，剔除不可序列化的列。"""
    drop_cols = ['yearly_rets']

    top10_path  = BACKTEST_DIR / f"{symbol}_top10_{start}_{end}.csv"
    stable_path = BACKTEST_DIR / f"{symbol}_stable_all_{start}_{end}.csv"

    top10.drop(columns=drop_cols, errors='ignore').to_csv(
        top10_path, index=False, encoding='utf-8-sig')
    stable.drop(columns=drop_cols, errors='ignore').to_csv(
        stable_path, index=False, encoding='utf-8-sig')
    print(f"[CSV] Top10  → {top10_path}")
    print(f"[CSV] 全量稳定 → {stable_path}")


# ═══════════════════════════════════════════════════════════════════
# 7. 主入口
# ═══════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    'ann_ret':       0.35,
    'year_win_rate': 0.30,
    'calmar':        0.20,
    'sharpe':        0.15,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='振幅策略 大型参数扫描 Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--symbol',  default='', help='单股代码，如 601288')
    parser.add_argument('--symbols', default='',
                        help='多股代码（逗号分隔），如 601288,600519,000001')
    parser.add_argument('--start',   required=True, help='回测开始 YYYYMMDD')
    parser.add_argument('--end',     required=True, help='回测结束 YYYYMMDD')

    parser.add_argument('--buy-min',  type=float, default=80.0,  help='buy_pct 最小值 (默认 80)')
    parser.add_argument('--buy-max',  type=float, default=99.0,  help='buy_pct 最大值 (默认 99)')
    parser.add_argument('--sell-min', type=float, default=101.0, help='sell_pct 最小值 (默认 101)')
    parser.add_argument('--sell-max', type=float, default=130.0, help='sell_pct 最大值 (默认 130)')
    parser.add_argument('--step',     type=float, default=1.0,   help='参数步长 (默认 1.0)')

    parser.add_argument('--min-year-win', type=float, default=70.0,
                        help='年度胜率过滤阈值 %% (默认 70)')
    parser.add_argument('--min-trades',   type=float, default=1.0,
                        help='年均最低交易数 (默认 1.0)')
    parser.add_argument('--cash',         type=float, default=100_000.0,
                        help='初始资金 (默认 100000)')
    parser.add_argument('--top-n',        type=int,   default=10,
                        help='推荐 Top N (默认 10)')

    args = parser.parse_args()

    # 确定股票列表
    if args.symbols:
        sym_list = [s.strip() for s in args.symbols.split(',') if s.strip()]
        label    = '+'.join(sym_list)
    elif args.symbol:
        sym_list = [args.symbol.strip()]
        label    = args.symbol.strip()
    else:
        print("[ERROR] 请提供 --symbol 或 --symbols", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  振幅策略 大型参数扫描 Pipeline")
    print(f"  股票: {label}   区间: {args.start}~{args.end}")
    print(f"{'='*70}\n")

    # ── Step 1: 数据准备 ──
    print("[Step 1] 准备行情数据 ...")
    df = build_composite_price(sym_list, args.start, args.end)
    print(f"  数据行数: {len(df)}  "
          f"日期: {pd.to_datetime(df['date'].iloc[0]).date()} ~ "
          f"{pd.to_datetime(df['date'].iloc[-1]).date()}")

    # ── Step 2: 参数扫描 ──
    print(f"\n[Step 2] 参数扫描 ...")
    results = parameter_sweep(
        df,
        buy_min=args.buy_min,   buy_max=args.buy_max,
        sell_min=args.sell_min, sell_max=args.sell_max,
        step=args.step,
        cash=args.cash,
    )

    # 保存全量结果（不含 yearly_rets dict）
    full_csv = BACKTEST_DIR / f"{label}_sweep_full_{args.start}_{args.end}.csv"
    results.drop(columns=['yearly_rets'], errors='ignore').to_csv(
        full_csv, index=False, encoding='utf-8-sig')
    print(f"[CSV] 全量扫描结果 → {full_csv}")

    # ── Step 3: 评分排名 ──
    print(f"\n[Step 3] 综合评分与 Top {args.top_n} 筛选 ...")
    stable, top10 = score_and_top10(
        results,
        min_trades_per_year=args.min_trades,
        min_year_win_rate=args.min_year_win,
        weights=DEFAULT_WEIGHTS,
        top_n=args.top_n,
    )
    print(f"  稳定组合: {len(stable)} 个  Top {args.top_n} 已选出")

    # ── Step 4: 控制台报告 ──
    print_console_report(
        stable, top10, results, label,
        args.start, args.end,
        args.buy_min, args.buy_max,
        args.sell_min, args.sell_max,
        args.step, args.min_year_win, args.min_trades,
    )

    # ── Step 5: 可视化 ──
    print(f"[Step 5] 生成图表 ...")

    # 将 composite_score 合并回 results（用于热力图）
    score_map = {(r['buy_pct'], r['sell_pct']): r['composite_score']
                 for _, r in stable.iterrows()}
    results['composite_score'] = results.apply(
        lambda r: score_map.get((r['buy_pct'], r['sell_pct']), np.nan), axis=1)

    plot_param_heatmap(results, top10, label, args.start, args.end)
    plot_top10_equity(df, top10, label, args.start, args.end, args.cash)
    plot_yearly_heatmap(top10, label, args.start, args.end)
    plot_best_detail(df, top10.iloc[0], label, args.start, args.end, args.cash)

    # ── Step 6: 保存报告 ──
    print(f"\n[Step 6] 保存报告与 CSV ...")
    save_markdown_report(
        stable, top10, results, label,
        args.start, args.end,
        args.buy_min, args.buy_max,
        args.sell_min, args.sell_max,
        args.step, args.min_year_win, args.min_trades,
        DEFAULT_WEIGHTS,
    )
    save_csv(stable, top10, label, args.start, args.end)

    print(f"\n{'='*70}")
    print(f"  Pipeline 完成！输出目录: {BACKTEST_DIR}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
