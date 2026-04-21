#!/usr/bin/env python3
"""
振幅策略一体化流水线 - 自动拉数据 + 回测 + 网格学习

【单次回测】
  python amplitude_pipeline.py --symbol 601288 --start 20200101 --end 20251231
  python amplitude_pipeline.py --symbol 601288 --start 20200101 --end 20251231 --buy-pct 97 --sell-pct 103

【网格学习】(自动推荐最优参数)
  python amplitude_pipeline.py --symbol 601288 --start 20200101 --end 20251231 --grid
  python amplitude_pipeline.py --symbol 601288 --start 20200101 --end 20251231 \\
      --grid --min-trades 5 --buy-range 90 99 --sell-range 101 115 --step 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from core.fetch import fetch_daily
from amplitude.backtest import (
    run_backtest, calc_metrics, print_report, plot_result,
    DATA_DIR, BACKTEST_DIR,
)


# ─────────────────────────── 数据层（自动拉取）───────────────────────────

def get_data(symbol: str, start: str, end: str, adjust: str = 'qfq') -> pd.DataFrame:
    """优先读 data/ 本地缓存，找不到则自动拉取并保存。"""
    start_dt = pd.to_datetime(start, format='%Y%m%d')
    end_dt   = pd.to_datetime(end,   format='%Y%m%d')
    best_df  = None
    for fpath in sorted(DATA_DIR.glob(f"{symbol}_daily_*_*_{adjust}.csv")):
        df  = pd.read_csv(fpath, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'])
        sub = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
        if not sub.empty and (best_df is None or len(sub) > len(best_df)):
            best_df = sub
    if best_df is not None:
        best_df = best_df.sort_values('date').reset_index(drop=True)
        print(f"[数据] 本地缓存  {len(best_df)} 行  "
              f"{best_df['date'].iloc[0].date()} ~ {best_df['date'].iloc[-1].date()}")
        return best_df
    print(f"[数据] 本地无缓存，自动拉取 {symbol}  {start} ~ {end} ...")
    df    = fetch_daily(symbol, start, end, adjust)
    fpath = DATA_DIR / f"{symbol}_daily_{start}_{end}_{adjust}.csv"
    df.to_csv(fpath, index=False, encoding='utf-8-sig')
    print(f"[数据] 已保存 {len(df)} 行 → {fpath}")
    return df


# ─────────────────────────── 单次回测 ───────────────────────────

def single_run(df: pd.DataFrame, symbol: str, start: str, end: str,
               buy_pct: float, sell_pct: float, cash: float) -> None:
    result  = run_backtest(df, buy_pct, sell_pct, cash)
    metrics = calc_metrics(result, df)

    txt_path = BACKTEST_DIR / f"{symbol}_report_{buy_pct}_{sell_pct}_{start}_{end}.txt"
    print_report(metrics, buy_pct, sell_pct, symbol, txt_path)
    plot_result(result, metrics, df, buy_pct, sell_pct, symbol, start, end)

    mr_path = BACKTEST_DIR / f"{symbol}_monthly_{buy_pct}_{sell_pct}_{start}_{end}.csv"
    pd.DataFrame([
        {'月份': k, '收益率(%)': v} for k, v in metrics['monthly_rets'].items()
    ]).to_csv(mr_path, index=False, encoding='utf-8-sig')
    print(f"[月度收益] → {mr_path}")

    if not result['trades_df'].empty:
        tr_path = BACKTEST_DIR / f"{symbol}_trades_{buy_pct}_{sell_pct}_{start}_{end}.csv"
        result['trades_df'].to_csv(tr_path, index=False, encoding='utf-8-sig')
        print(f"[交易记录] {len(result['trades_df'])} 条 → {tr_path}")


# ─────────────────────────── 网格学习 ───────────────────────────

def _run_one(df: pd.DataFrame, buy: float, sell: float, cash: float) -> dict:
    """
    轻量运行单个参数组合，只返回统计数字，不保存图表/文件。

    返回指标包括：
      - buy_pct / sell_pct : 参数
      - ann_ret / total_ret : 年化收益率 / 总收益率
      - max_dd : 最大回撤
      - sharpe / calmar : 风险调整收益指标
      - win_rate / avg_profit : 胜率 / 平均单笔盈亏
      - n_trades / ann_trades : 交易笔数 / 年均交易笔数
      - years : 回测年限
      - bh_ret / bh_ann : 买持基准收益 / 买持基准年化
      - profit_factor : 盈亏比 (总盈利 / 总亏损 的绝对值)
    """
    result = run_backtest(df, buy, sell, cash)
    m      = calc_metrics(result, df)

    # ── 计算盈亏比 (Profit Factor) ──
    # 盈亏比 = 所有盈利交易的利润之和 / 所有亏损交易的亏损之和 (取绝对值)
    sells = result['trades_df']
    sells = sells[sells['action'] == 'SELL'] if not sells.empty else sells
    if len(sells) > 0:
        total_win  = sells.loc[sells['profit_pct'] > 0, 'profit_pct'].sum()
        total_loss = abs(sells.loc[sells['profit_pct'] <= 0, 'profit_pct'].sum())
        profit_factor = round(total_win / total_loss, 3) if total_loss > 0 else float('inf')
    else:
        profit_factor = 0.0

    return {
        'buy_pct':       buy,
        'sell_pct':      sell,
        'ann_ret':       m['ann_ret'],
        'total_ret':     m['total_ret'],
        'max_dd':        m['max_dd'],
        'sharpe':        m['sharpe'],
        'calmar':        m['calmar'],
        'win_rate':      m['win_rate'],
        'avg_profit':    m['avg_profit'],
        'n_trades':      m['n_trades'],
        'ann_trades':    round(m['n_trades'] / max(m['years'], 0.01), 2),
        'years':         m['years'],
        'bh_ret':        m['bh_ret'],
        'bh_ann':        m['bh_ann'],
        'profit_factor': profit_factor,
    }


def grid_search(df: pd.DataFrame,
                buy_range: tuple[float, float],
                sell_range: tuple[float, float],
                step: float,
                min_trades_per_year: float,
                cash: float,
                symbol: str,
                start: str,
                end: str) -> None:
    buy_vals  = [round(v, 2) for v in
                 np.arange(buy_range[0], buy_range[1] + step / 2, step)]
    sell_vals = [round(v, 2) for v in
                 np.arange(sell_range[0], sell_range[1] + step / 2, step)]
    combos = [(b, s) for b in buy_vals for s in sell_vals if b < s]
    total  = len(combos)

    print(f"\n[网格] 扫描 {total} 个参数组合  "
          f"buy {buy_range[0]}~{buy_range[1]}  "
          f"sell {sell_range[0]}~{sell_range[1]}  step {step}")

    rows = []
    for i, (buy, sell) in enumerate(combos, 1):
        rows.append(_run_one(df, buy, sell, cash))
        print(f"\r  进度: {i:>4}/{total}  buy={buy:.1f}  sell={sell:.1f}   ",
              end='', flush=True)
    print()

    results = pd.DataFrame(rows)

    # 保存完整网格 CSV
    grid_csv = BACKTEST_DIR / f"{symbol}_grid_{start}_{end}.csv"
    results.to_csv(grid_csv, index=False, encoding='utf-8-sig')
    print(f"[网格CSV] {len(results)} 组 → {grid_csv}")

    # 将 buy_range / sell_range / step 传给 _analyze_and_recommend
    # 用于在综合报告中展示扫描范围信息
    _analyze_and_recommend(
        results, df, symbol, start, end, cash, min_trades_per_year,
        buy_range=buy_range, sell_range=sell_range, step=step,
    )


def _analyze_and_recommend(results: pd.DataFrame, df: pd.DataFrame,
                            symbol: str, start: str, end: str,
                            cash: float, min_trades: float,
                            buy_range: tuple, sell_range: tuple, step: float) -> None:
    valid = results[results['ann_trades'] >= min_trades].copy()
    if valid.empty:
        max_t = results['ann_trades'].max()
        print(f"\n[警告] 无年均交易数 >= {min_trades} 的组合，"
              f"当前最高年均交易数 {max_t:.1f}，已展示全量结果")
        valid = results.copy()

    bp_row = valid.loc[valid['ann_ret'].idxmax()]
    bs_row = valid.loc[valid['calmar'].idxmax()]

    # ① 控制台：完整表格
    _print_full_grid_table(results, valid, min_trades, bp_row, bs_row)

    # ② 热力图 PNG
    _plot_heatmap(results, min_trades, symbol, start, end)

    # ③ 综合报告 MD
    report_md = _build_comprehensive_md(
        results, valid, min_trades, bp_row, bs_row,
        symbol, start, end, cash, buy_range, sell_range, step)
    rpt_path = BACKTEST_DIR / f"{symbol}_grid_report_{start}_{end}.md"
    rpt_path.write_text(report_md, encoding='utf-8')
    print(f"[综合报告] → {rpt_path}")

    # ④ 推荐参数完整回测输出
    same = (bp_row['buy_pct'] == bs_row['buy_pct'] and
            bp_row['sell_pct'] == bs_row['sell_pct'])
    pairs = [('profit', '收益最高', bp_row)]
    if not same:
        pairs.append(('safest', '最稳妥', bs_row))

    for tag, label, row in pairs:
        buy, sell = row['buy_pct'], row['sell_pct']
        print(f"\n{'='*64}")
        print(f"  完整报告 [{label}]  buy={buy:.1f}%  sell={sell:.1f}%")
        print(f"{'='*64}")
        result  = run_backtest(df, buy, sell, cash)
        metrics = calc_metrics(result, df)

        md_path = BACKTEST_DIR / f"{symbol}_grid_{tag}_{buy}_{sell}_{start}_{end}.md"
        print_report(metrics, buy, sell, symbol, md_path)
        plot_result(result, metrics, df, buy, sell, symbol, start, end)

        mr_path = BACKTEST_DIR / f"{symbol}_grid_{tag}_monthly_{buy}_{sell}_{start}_{end}.csv"
        pd.DataFrame([
            {'月份': k, '收益率(%)': v} for k, v in metrics['monthly_rets'].items()
        ]).to_csv(mr_path, index=False, encoding='utf-8-sig')
        print(f"[月度收益] → {mr_path}")

        if not result['trades_df'].empty:
            tr_path = BACKTEST_DIR / f"{symbol}_grid_{tag}_trades_{buy}_{sell}_{start}_{end}.csv"
            result['trades_df'].to_csv(tr_path, index=False, encoding='utf-8-sig')
            print(f"[交易记录] {len(result['trades_df'])} 条 → {tr_path}")


# ── 控制台全量表格 ──────────────────────────────────────────────

def _print_full_grid_table(results: pd.DataFrame, valid: pd.DataFrame,
                            min_trades: float, bp: pd.Series, bs: pd.Series) -> None:
    """
    在控制台打印完整的网格搜索结果表格。

    包含三个部分：
      1. 概览统计：全量/有效组合数 + 各指标平均值/中位数/极值
      2. 全量排序表：按年化收益降序，标注是否达标及推荐标签
      3. 有效参数排序表：仅保留达标组合，按卡玛比率降序
      4. 推荐摘要：最终推荐的两个最优参数组合
    """
    sep = "=" * 100
    hdr = (f"  {'#':>4} {'buy%':>5} {'sell%':>6} {'年化%':>7} {'总收益%':>8} "
           f"{'回撤%':>7} {'夏普':>6} {'卡玛':>6} {'胜率%':>6} {'盈亏比':>6} {'年均交易':>8} {'备注'}")
    div = f"  {'-'*97}"

    print(f"\n{sep}")
    print(f"  网格学习结果  过滤: 年均交易数 >= {min_trades}  "
          f"扫描: {len(results)} 组  有效: {len(valid)} 组")
    print(sep)

    # ── 概览统计 ──
    print(f"\n  ── 统计概览 ──")
    print(f"  {'-'*60}")
    for label, data_src in [('全量', results), ('有效', valid)]:
        if data_src.empty:
            continue
        print(f"  [{label}]  年化收益: 平均={data_src['ann_ret'].mean():.2f}%  "
              f"中位数={data_src['ann_ret'].median():.2f}%  "
              f"最高={data_src['ann_ret'].max():.2f}%  最低={data_src['ann_ret'].min():.2f}%")
        print(f"  [{label}]  最大回撤: 平均={data_src['max_dd'].mean():.2f}%  "
              f"中位数={data_src['max_dd'].median():.2f}%  "
              f"最小={data_src['max_dd'].min():.2f}%  最大={data_src['max_dd'].max():.2f}%")
        print(f"  [{label}]  夏普比率: 平均={data_src['sharpe'].mean():.3f}  "
              f"中位数={data_src['sharpe'].median():.3f}  "
              f"最高={data_src['sharpe'].max():.3f}  最低={data_src['sharpe'].min():.3f}")
        print(f"  [{label}]  胜率:     平均={data_src['win_rate'].mean():.1f}%  "
              f"年均交易: 平均={data_src['ann_trades'].mean():.1f}笔")
        # 盈利组合占比
        profitable = (data_src['ann_ret'] > 0).sum()
        print(f"  [{label}]  盈利组合: {profitable}/{len(data_src)} "
              f"({profitable/len(data_src)*100:.0f}%)  "
              f"买持基准年化: {data_src['bh_ann'].iloc[0]:.2f}%")
        # 跑赢基准占比
        beat_bh = (data_src['ann_ret'] > data_src['bh_ann']).sum()
        print(f"  [{label}]  跑赢买持: {beat_bh}/{len(data_src)} "
              f"({beat_bh/len(data_src)*100:.0f}%)")
        print()

    # ── 全量排序表 ──
    print(f"  ── 全量结果 (按年化收益排序，✗=未达年均交易数要求) ──")
    print(hdr); print(div)
    for rank, (_, r) in enumerate(
            results.sort_values('ann_ret', ascending=False).iterrows(), 1):
        ok  = r['ann_trades'] >= min_trades
        tag = ''
        if r['buy_pct'] == bp['buy_pct'] and r['sell_pct'] == bp['sell_pct']:
            tag = ' ★收益'
        elif r['buy_pct'] == bs['buy_pct'] and r['sell_pct'] == bs['sell_pct']:
            tag = ' ★稳妥'
        flag = ('✓' if ok else '✗') + tag
        pf = r.get('profit_factor', 0)
        pf_str = f"{pf:>6.2f}" if pf != float('inf') else "   inf"
        print(f"  {rank:>4} {r['buy_pct']:>5.1f} {r['sell_pct']:>6.1f} "
              f"{r['ann_ret']:>7.2f} {r['total_ret']:>8.2f} {r['max_dd']:>7.2f} "
              f"{r['sharpe']:>6.3f} {r['calmar']:>6.3f} {r['win_rate']:>6.1f} "
              f"{pf_str} {r['ann_trades']:>8.1f}  {flag}")

    # ── 有效参数排序表 ──
    print(f"\n  ── 有效参数 (年均交易数 >= {min_trades}，按卡玛比率排序) ──")
    print(hdr); print(div)
    for rank, (_, r) in enumerate(
            valid.sort_values('calmar', ascending=False).iterrows(), 1):
        tag = ''
        if r['buy_pct'] == bp['buy_pct'] and r['sell_pct'] == bp['sell_pct']:
            tag = ' ★收益最高'
        if r['buy_pct'] == bs['buy_pct'] and r['sell_pct'] == bs['sell_pct']:
            tag += ' ★最稳妥'
        pf = r.get('profit_factor', 0)
        pf_str = f"{pf:>6.2f}" if pf != float('inf') else "   inf"
        print(f"  {rank:>4} {r['buy_pct']:>5.1f} {r['sell_pct']:>6.1f} "
              f"{r['ann_ret']:>7.2f} {r['total_ret']:>8.2f} {r['max_dd']:>7.2f} "
              f"{r['sharpe']:>6.3f} {r['calmar']:>6.3f} {r['win_rate']:>6.1f} "
              f"{pf_str} {r['ann_trades']:>8.1f}{tag}")

    # ── 推荐摘要 ──
    print(f"\n{sep}")
    print(f"  ★ 收益最高: buy={bp['buy_pct']:.1f}%  sell={bp['sell_pct']:.1f}%  "
          f"年化={bp['ann_ret']:.2f}%  总收益={bp['total_ret']:.2f}%  "
          f"回撤={bp['max_dd']:.2f}%  夏普={bp['sharpe']:.3f}  "
          f"胜率={bp['win_rate']:.1f}%  年均={bp['ann_trades']:.1f}笔")
    print(f"  ★ 最稳妥:   buy={bs['buy_pct']:.1f}%  sell={bs['sell_pct']:.1f}%  "
          f"年化={bs['ann_ret']:.2f}%  总收益={bs['total_ret']:.2f}%  "
          f"回撤={bs['max_dd']:.2f}%  卡玛={bs['calmar']:.3f}  "
          f"胜率={bs['win_rate']:.1f}%  年均={bs['ann_trades']:.1f}笔")
    print(f"{sep}\n")


# ── 热力图 ──────────────────────────────────────────────────────

def _plot_heatmap(results: pd.DataFrame, min_trades: float,
                  symbol: str, start: str, end: str) -> Path:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    buy_vals  = sorted(results['buy_pct'].unique())
    sell_vals = sorted(results['sell_pct'].unique())

    panels = [
        ('ann_ret',    '年化收益率 (%)',  'RdYlGn',   False),
        ('calmar',     '卡玛比率',        'RdYlGn',   False),
        ('max_dd',     '最大回撤 (%)',    'RdYlGn_r', False),
        ('ann_trades', '年均交易数',      'YlOrBr',   False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'{symbol} 振幅策略网格热力图  {start[:4]}-{end[:4]}'
                 f'  (加粗=年均交易数 >= {min_trades})', fontsize=13)

    for ax, (col, title, cmap, _) in zip(axes.flat, panels):
        mat = np.full((len(buy_vals), len(sell_vals)), np.nan)
        for i, buy in enumerate(buy_vals):
            for j, sell in enumerate(sell_vals):
                row = results[(results['buy_pct'] == buy) & (results['sell_pct'] == sell)]
                if not row.empty:
                    mat[i, j] = row.iloc[0][col]

        im = ax.imshow(mat[::-1], cmap=cmap, aspect='auto',
                       vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_xticks(range(len(sell_vals)))
        ax.set_xticklabels([f'{v:.0f}' for v in sell_vals], fontsize=8)
        ax.set_yticks(range(len(buy_vals)))
        ax.set_yticklabels([f'{v:.0f}' for v in buy_vals[::-1]], fontsize=8)
        ax.set_xlabel('sell_pct (%)', fontsize=9)
        ax.set_ylabel('buy_pct (%)', fontsize=9)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85)

        for i, buy in enumerate(buy_vals[::-1]):
            for j, sell in enumerate(sell_vals):
                row = results[(results['buy_pct'] == buy) & (results['sell_pct'] == sell)]
                if row.empty:
                    continue
                val = row.iloc[0][col]
                ok  = row.iloc[0]['ann_trades'] >= min_trades
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=7.5, color='black',
                        fontweight='bold' if ok else 'normal',
                        alpha=1.0 if ok else 0.55)

    plt.tight_layout()
    out = BACKTEST_DIR / f"{symbol}_grid_heatmap_{start}_{end}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[热力图]   → {out}")
    return out


# ── 综合报告 Markdown ──────────────────────────────────────────

def _build_comprehensive_md(results: pd.DataFrame, valid: pd.DataFrame,
                             min_trades: float, bp: pd.Series, bs: pd.Series,
                             symbol: str, start: str, end: str, cash: float,
                             buy_range: tuple, sell_range: tuple, step: float) -> str:
    """
    构建详尽的综合报告 Markdown 文本。
    """
    def _pf(val):
        return f"{val:.2f}" if val != float('inf') else "inf"

    lines = [
        f"# {symbol} 振幅策略网格学习 ————— 综合报告",
        "",
        f"> **扫描范围**: buy {buy_range[0]}%~{buy_range[1]}%  |  sell {sell_range[0]}%~{sell_range[1]}%  |  step {step}%",
        f"> **回测区间**: {start[:4]}-{start[4:6]}-{start[6:]} ~ {end[:4]}-{end[4:6]}-{end[6:]} ({results['years'].iloc[0]:.2f} 年)" if not results.empty else "> **回测区间**: 无数据",
        f"> **过滤条件**: 年均交易数 >= {min_trades}  |  初始资金: {cash:,.0f} 元",
        f"> **扫描结果**: {len(results)} 组  |  **有效**: {len(valid)} 组  |  **有效率**: {len(valid)/len(results)*100:.0f}%" if len(results)>0 else "> **扫描结果**: 0 组",
        f"> **买持基准**: 年化 = **{results['bh_ann'].iloc[0]:.2f}%**  |  总收益 = **{results['bh_ret'].iloc[0]:.2f}%**" if not results.empty else "",
        "",
        "## 【零】全局概览统计",
        "",
        "| 指标 | 均值 | 中位数 | 标准差 | 最小值 | 最大值 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |"
    ]

    if not results.empty:
        metrics = [
            ('年化收益率 (%)', 'ann_ret'),
            ('总收益率 (%)', 'total_ret'),
            ('最大回撤 (%)', 'max_dd'),
            ('夏普比率', 'sharpe'),
            ('卡玛比率', 'calmar'),
            ('胜率 (%)', 'win_rate'),
            ('平均单笔盈亏(%)', 'avg_profit'),
            ('年均交易数', 'ann_trades'),
        ]
        for label, col in metrics:
            if col in results.columns:
                s = results[col]
                lines.append(f"| {label} | {s.mean():.2f} | {s.median():.2f} | {s.std():.2f} | {s.min():.2f} | {s.max():.2f} |")
        
        profit_c = (results['ann_ret'] > 0).sum()
        lines += [
            "",
            f"- **盈利组合**: {profit_c}/{len(results)} ({profit_c/len(results)*100:.0f}%)",
            f"- **亏损组合**: {len(results)-profit_c}/{len(results)} ({(len(results)-profit_c)/len(results)*100:.0f}%)"
        ]
        beat_c = (results['ann_ret'] > results['bh_ann']).sum()
        lines.append(f"- **跑赢买持**: {beat_c}/{len(results)} ({beat_c/len(results)*100:.0f}%)")
        lines.append("")

    hdr_md = "| # | buy% | sell% | 年化% | 总收益% | 回撤% | 夏普 | 卡玛 | 胜率% | 盈亏比 | 年均交易 | 备注 |"
    sep_md = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|"

    lines += ["## 【一】全量扫描结果 (按年化收益降序)", "", hdr_md, sep_md]
    for rank, (_, r) in enumerate(results.sort_values('ann_ret', ascending=False).iterrows(), 1):
        ok = r['ann_trades'] >= min_trades
        tag = []
        if r['buy_pct'] == bp['buy_pct'] and r['sell_pct'] == bp['sell_pct']: tag.append('**★收益最高**')
        if r['buy_pct'] == bs['buy_pct'] and r['sell_pct'] == bs['sell_pct']: tag.append('**★最稳妥**')
        flag_str = ' '.join(tag) if ok else '~~未达标~~ '+' '.join(tag)
        row_str = f"| {rank} | {r['buy_pct']:.1f} | {r['sell_pct']:.1f} | **{r['ann_ret']:.2f}** | {r['total_ret']:.2f} | {r['max_dd']:.2f} | {r['sharpe']:.3f} | {r['calmar']:.3f} | {r['win_rate']:.1f} | {_pf(r.get('profit_factor',0))} | {r['ann_trades']:.1f} | {flag_str} |"
        lines.append(row_str)

    lines += ["", "## 【二】有效参数详细 (年均交易数达标，按卡玛比率降序)", "", hdr_md, sep_md]
    for rank, (_, r) in enumerate(valid.sort_values('calmar', ascending=False).iterrows(), 1):
        tag = []
        if r['buy_pct'] == bp['buy_pct'] and r['sell_pct'] == bp['sell_pct']: tag.append('**★收益最高**')
        if r['buy_pct'] == bs['buy_pct'] and r['sell_pct'] == bs['sell_pct']: tag.append('**★最稳妥**')
        row_str = f"| {rank} | {r['buy_pct']:.1f} | {r['sell_pct']:.1f} | **{r['ann_ret']:.2f}** | {r['total_ret']:.2f} | {r['max_dd']:.2f} | {r['sharpe']:.3f} | {r['calmar']:.3f} | {r['win_rate']:.1f} | {_pf(r.get('profit_factor',0))} | {r['ann_trades']:.1f} | {' '.join(tag)} |"
        lines.append(row_str)

    buy_vals = sorted(results['buy_pct'].unique())
    sell_vals = sorted(results['sell_pct'].unique())

    def _build_matrix(title, col, fmt="{:.2f}"):
        res = ["", f"## 【三】参数敏感性矩阵 — {title} (加*为年均交易达标)", ""]
        h = "| buy\\\\sell | " + " | ".join([f"{v:.1f}" for v in sell_vals]) + " |"
        s = "|---:|" + "---:|" * len(sell_vals)
        res.extend([h, s])
        for b in buy_vals:
            row_cells = [f"**{b:.1f}**"]
            for s_val in sell_vals:
                cdf = results[(results['buy_pct']==b) & (results['sell_pct']==s_val)]
                if cdf.empty:
                    row_cells.append(" - ")
                else:
                    v = cdf.iloc[0]
                    val_str = fmt.format(v[col])
                    if v['ann_trades'] >= min_trades:
                        row_cells.append(f"**{val_str}\\***")
                    else:
                        row_cells.append(val_str)
            res.append("| " + " | ".join(row_cells) + " |")
        return res

    lines += _build_matrix("年化收益率 (%)", 'ann_ret')
    lines += _build_matrix("卡玛比率", 'calmar', "{:.3f}")
    lines += _build_matrix("最大回撤 (%)", 'max_dd')
    lines += _build_matrix("夏普比率", 'sharpe', "{:.3f}")
    lines += _build_matrix("胜率 (%)", 'win_rate', "{:.1f}")
    lines += _build_matrix("年均交易数", 'ann_trades', "{:.1f}")

    def _top_md(title, sort_col, asc=False, n=10, df_src=results):
        res = ["", f"### {title}", "", hdr_md, sep_md]
        for rank, (_, r) in enumerate(df_src.sort_values(sort_col, ascending=asc).head(n).iterrows(), 1):
            ok = r['ann_trades'] >= min_trades
            res.append(f"| {rank} | {r['buy_pct']:.1f} | {r['sell_pct']:.1f} | **{r['ann_ret']:.2f}** | {r['total_ret']:.2f} | {r['max_dd']:.2f} | {r['sharpe']:.3f} | {r['calmar']:.3f} | {r['win_rate']:.1f} | {_pf(r.get('profit_factor',0))} | {r['ann_trades']:.1f} | {'' if ok else '~~未达标~~'} |")
        return res

    lines += ["", "## 【四】各项指标排行榜 (Top/Bottom 10)", ""]
    lines += _top_md("年化收益率 Top 10", 'ann_ret')
    lines += _top_md("年化收益率 Bottom 10 (最差)", 'ann_ret', True)
    lines += _top_md("卡玛比率 Top 10", 'calmar')
    lines += _top_md("夏普比率 Top 10", 'sharpe')
    lines += _top_md("最大回撤(小) Top 10 (最安全)", 'max_dd', True)
    lines += _top_md("最大回撤(大) Bottom 10 (最危险)", 'max_dd')
    lines += _top_md("胜率 Top 10", 'win_rate')
    
    df_pf = results.assign(pf_clean=results['profit_factor'].replace([np.inf, -np.inf], 999999).fillna(0))
    lines += _top_md("盈亏比 Top 10", 'pf_clean', False, 10, df_pf)

    lines += ["", "## 【五】稳定性分析 — 低回撤+高胜率组合", ""]
    med_dd = valid['max_dd'].median()
    med_wr = valid['win_rate'].median()
    lines += [
        f"> 筛选条件: **年均交易数达标** 且 **回撤 <= {med_dd:.2f}%** 且 **胜率 >= {med_wr:.1f}%**",
        ""
    ]
    stable = valid[(valid['max_dd'] <= med_dd) & (valid['win_rate'] >= med_wr)]
    lines += [f"符合条件的组合数: **{len(stable)} 组**", "", hdr_md, sep_md]
    for rank, (_, r) in enumerate(stable.sort_values('calmar', ascending=False).iterrows(), 1):
        lines.append(f"| {rank} | {r['buy_pct']:.1f} | {r['sell_pct']:.1f} | **{r['ann_ret']:.2f}** | {r['total_ret']:.2f} | {r['max_dd']:.2f} | {r['sharpe']:.3f} | {r['calmar']:.3f} | {r['win_rate']:.1f} | {_pf(r.get('profit_factor',0))} | {r['ann_trades']:.1f} | |")

    lines += ["", "## 【六】参数边际效应分析", "", "### (a) buy_pct 边际效应 (各买入阈值的平均表现)", ""]
    lines += ["| buy_pct | 平均年化% | 平均回撤% | 平均夏普 | 平均卡玛 | 平均胜率% | 平均年交易 | 组合数 |",
              "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for b in buy_vals:
        sub = results[results['buy_pct']==b]
        if not sub.empty:
            lines.append(f"| **{b:.1f}** | {sub['ann_ret'].mean():.2f} | {sub['max_dd'].mean():.2f} | {sub['sharpe'].mean():.3f} | {sub['calmar'].mean():.3f} | {sub['win_rate'].mean():.1f} | {sub['ann_trades'].mean():.1f} | {len(sub)} |")

    lines += ["", "### (b) sell_pct 边际效应 (各卖出阈值的平均表现)", ""]
    lines += ["| sell_pct | 平均年化% | 平均回撤% | 平均夏普 | 平均卡玛 | 平均胜率% | 平均年交易 | 组合数 |",
              "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s_val in sell_vals:
        sub = results[results['sell_pct']==s_val]
        if not sub.empty:
            lines.append(f"| **{s_val:.1f}** | {sub['ann_ret'].mean():.2f} | {sub['max_dd'].mean():.2f} | {sub['sharpe'].mean():.3f} | {sub['calmar'].mean():.3f} | {sub['win_rate'].mean():.1f} | {sub['ann_trades'].mean():.1f} | {len(sub)} |")

    lines += ["", "## 【七】跑赢买持基准分析", ""]
    bh_ann = results['bh_ann'].iloc[0] if not results.empty else 0
    lines.append(f"> **买持基准**: 年化={bh_ann:.2f}%  |  跑赢策略: **{beat_c}/{len(results)}** ({beat_c/len(results)*100:.0f}%)")
    results['excess'] = results['ann_ret'] - results['bh_ann']
    
    lines += [
        "",
        "| 超额指标 | 数值 |",
        "| :--- | ---: |",
        f"| 平均超额 | {results['excess'].mean():.2f}% |",
        f"| 中位数超额 | {results['excess'].median():.2f}% |",
        f"| 最大超额 | {results['excess'].max():.2f}% |",
        f"| 最小超额 | {results['excess'].min():.2f}% |",
        ""
    ]
    vh = valid[valid['ann_ret'] > valid['bh_ann']]
    lines += [f"### 有效且跑赢基准 ({len(vh)} 组)", "", hdr_md, sep_md]
    for rank, (_, r) in enumerate(vh.sort_values('ann_ret', ascending=False).iterrows(),1):
        lines.append(f"| {rank} | {r['buy_pct']:.1f} | {r['sell_pct']:.1f} | **{r['ann_ret']:.2f}** | {r['total_ret']:.2f} | {r['max_dd']:.2f} | {r['sharpe']:.3f} | {r['calmar']:.3f} | {r['win_rate']:.1f} | {_pf(r.get('profit_factor',0))} | {r['ann_trades']:.1f} | |")

    lines += ["", "## 【八】风险收益象限分析", ""]
    med_ret = results['ann_ret'].median()
    med_rd = results['max_dd'].median()
    q1 = valid[(valid['ann_ret'] >= med_ret) & (valid['max_dd'] <= med_rd)]
    q2 = valid[(valid['ann_ret'] >= med_ret) & (valid['max_dd'] > med_rd)]
    q3 = valid[(valid['ann_ret'] < med_ret) & (valid['max_dd'] <= med_rd)]
    q4 = valid[(valid['ann_ret'] < med_ret) & (valid['max_dd'] > med_rd)]

    lines += [
        f"> **分界线**: 年化收益 `{med_ret:.2f}%`  |  最大回撤 `{med_rd:.2f}%`",
        "",
        "| 象限 | 描述 | 数量 |",
        "| :--- | :--- | ---: |",
        f"| **Q1** | **高收益 + 低风险 (★最佳)** | {len(q1)} 组 |",
        f"| **Q2** | 高收益 + 高风险 | {len(q2)} 组 |",
        f"| **Q3** | 低收益 + 低风险 | {len(q3)} 组 |",
        f"| **Q4** | 低收益 + 高风险 (最差) | {len(q4)} 组 |",
        "",
        "### Q1 最佳象限详情",
        "", hdr_md, sep_md
    ]
    for rank, (_, r) in enumerate(q1.sort_values('ann_ret', ascending=False).iterrows(), 1):
        lines.append(f"| {rank} | {r['buy_pct']:.1f} | {r['sell_pct']:.1f} | **{r['ann_ret']:.2f}** | {r['total_ret']:.2f} | {r['max_dd']:.2f} | {r['sharpe']:.3f} | {r['calmar']:.3f} | {r['win_rate']:.1f} | {_pf(r.get('profit_factor',0))} | {r['ann_trades']:.1f} | |")

    lines += ["", "## 【九】参数组合综合评分排行 (仅有效组合)", ""]
    valid_c = valid.copy()
    if not valid_c.empty:
        for c in ['ann_ret', 'calmar', 'sharpe', 'win_rate']:
            c_min, c_max = valid_c[c].min(), valid_c[c].max()
            if c_max > c_min:
                valid_c[c+'_norm'] = (valid_c[c] - c_min) / (c_max - c_min)
            else:
                valid_c[c+'_norm'] = 1.0
        
        md_min, md_max = valid_c['max_dd'].min(), valid_c['max_dd'].max()
        if md_max > md_min:
            valid_c['mdd_norm'] = (md_max - valid_c['max_dd']) / (md_max - md_min)
        else:
            valid_c['mdd_norm'] = 1.0

        valid_c['score'] = (valid_c['ann_ret_norm'] * 0.30 +
                            valid_c['calmar_norm'] * 0.25 +
                            valid_c['sharpe_norm'] * 0.20 +
                            valid_c['win_rate_norm'] * 0.15 +
                            valid_c['mdd_norm'] * 0.10) * 100

        lines += [
            "> **评分权重**: 年化(30%) + 卡玛(25%) + 夏普(20%) + 胜率(15%) + 低回撤(10%)",
            "",
            "| # | buy% | sell% | 综合分 | 年化% | 总收益% | 回撤% | 夏普 | 卡玛 | 胜率% | 年均交易 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        ]
        for rank, (_, r) in enumerate(valid_c.sort_values('score', ascending=False).iterrows(), 1):
            lines.append(f"| {rank} | {r['buy_pct']:.1f} | {r['sell_pct']:.1f} | **{r['score']:.1f}** | {r['ann_ret']:.2f} | {r['total_ret']:.2f} | {r['max_dd']:.2f} | {r['sharpe']:.3f} | {r['calmar']:.3f} | {r['win_rate']:.1f} | {r['ann_trades']:.1f} |")

    lines += ["", "## 【十】最终推荐", ""]
    
    lines.append(f"> [!TIP]")
    lines.append(f"> **★ 收益最高优先**<br>")
    lines.append(f"> **buy = {bp['buy_pct']:.1f}% | sell = {bp['sell_pct']:.1f}%**<br>")
    lines.append(f"> - **核心**: 年化收益 `{bp['ann_ret']:.2f}%` (总计 `{bp['total_ret']:.2f}%`)<br>")
    lines.append(f"> - **风险**: 最大回撤 `{bp['max_dd']:.2f}%`<br>")
    lines.append(f"> - **质量**: 夏普 `{bp['sharpe']:.3f}` | 卡玛 `{bp['calmar']:.3f}` | 胜率 `{bp['win_rate']:.1f}%`<br>")
    lines.append(f"> - **详情**: 平均单笔盈亏 `{bp['avg_profit']:.2f}%` | 盈亏比 `{_pf(bp.get('profit_factor',0))}` | 年均交易 `{bp['ann_trades']:.1f}` 笔")
    lines.append("")
    lines.append(f"> [!NOTE]")
    lines.append(f"> **★ 最稳妥优先 (追求最低风险下的稳健收益)**<br>")
    lines.append(f"> **buy = {bs['buy_pct']:.1f}% | sell = {bs['sell_pct']:.1f}%**<br>")
    lines.append(f"> - **核心**: 年化收益 `{bs['ann_ret']:.2f}%` (总计 `{bs['total_ret']:.2f}%`)<br>")
    lines.append(f"> - **风险**: 最大回撤 `{bs['max_dd']:.2f}%`<br>")
    lines.append(f"> - **质量**: 夏普 `{bs['sharpe']:.3f}` | 卡玛 `{bs['calmar']:.3f}` | 胜率 `{bs['win_rate']:.1f}%`<br>")
    lines.append(f"> - **详情**: 平均单笔盈亏 `{bs['avg_profit']:.2f}%` | 盈亏比 `{_pf(bs.get('profit_factor',0))}` | 年均交易 `{bs['ann_trades']:.1f}` 笔")

    lines += [
        "", "---", "",
        "## 附录：核心字段与术语解释",
        "",
        "| 字段名 | 解释说明 |",
        "| :--- | :--- |",
        "| **buy_pct / sell_pct** | 策略的进出场阈值。buy=价格跌至近期峰值一定比例时抄底；sell=价格反弹至买入峰值一定比例时止盈。 |",
        "| **年化收益率 (ann_ret)** | 整个测试空间按复利折算的年均增长率，最直观衡量策略的盈利速度。 |",
        "| **最大回撤 (max_dd)** | 账户净值从历史顶峰跌至最低谷的最大跌幅，数值越小风险抗压性越好。 |",
        "| **夏普比率 (sharpe)** | 承担每一单位预期风险所能获取的超额收益，一般 >1 为较优策略。 |",
        "| **卡玛比率 (calmar)** | 年化收益率与最大回撤的比值 (收益/最高极限风险)。数值越大说明“性价比”越高。 |",
        "| **胜率 (win_rate)** | 盈利的平仓交易次数 / 总交易次数，衡量每一次入场的成功概率。 |",
        "| **盈亏比** | (盈利交易总收益/亏损交易总亏损)，如果从未出现亏损单，则该值为 inf。 |",
        "| **年均交易数** | 平均每年产生的完整交易闭环次数，策略如果频率过低说明很少遇到建仓机会。 |",
        ""
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description='振幅策略流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 通用参数
    p.add_argument('--symbol',  required=True, help='股票代码，如 601288')
    p.add_argument('--start',   required=True, help='开始日期 YYYYMMDD')
    p.add_argument('--end',     required=True, help='结束日期 YYYYMMDD')
    p.add_argument('--adjust',  default='qfq', choices=['qfq', 'hfq', 'none'],
                   help='复权方式 (默认 qfq)')
    p.add_argument('--cash',    type=float, default=100_000.0,
                   help='初始资金 (默认 100000)')

    # 单次模式参数
    p.add_argument('--buy-pct',  type=float, default=97.0,
                   help='买入触发比例 [单次模式] (默认 97)')
    p.add_argument('--sell-pct', type=float, default=103.0,
                   help='卖出触发比例 [单次模式] (默认 103)')

    # 网格模式参数
    p.add_argument('--grid', action='store_true', help='启用网格学习模式')
    p.add_argument('--buy-range',  nargs=2, type=float, default=[88.0, 99.0],
                   metavar=('MIN', 'MAX'), help='buy_pct 扫描范围 (默认 88 99)')
    p.add_argument('--sell-range', nargs=2, type=float, default=[101.0, 115.0],
                   metavar=('MIN', 'MAX'), help='sell_pct 扫描范围 (默认 101 115)')
    p.add_argument('--step',       type=float, default=1.0,
                   help='网格步长 (默认 1)')
    p.add_argument('--min-trades', type=float, default=3.0,
                   help='年均最小交易数过滤 (默认 3)')

    args = p.parse_args()

    try:
        df = get_data(args.symbol, args.start, args.end, args.adjust)

        if args.grid:
            grid_search(
                df=df,
                buy_range=tuple(args.buy_range),
                sell_range=tuple(args.sell_range),
                step=args.step,
                min_trades_per_year=args.min_trades,
                cash=args.cash,
                symbol=args.symbol,
                start=args.start,
                end=args.end,
            )
        else:
            if args.buy_pct >= args.sell_pct:
                sys.exit(f"[ERROR] --buy-pct({args.buy_pct}) 必须小于 --sell-pct({args.sell_pct})")
            single_run(df, args.symbol, args.start, args.end,
                       args.buy_pct, args.sell_pct, args.cash)

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
