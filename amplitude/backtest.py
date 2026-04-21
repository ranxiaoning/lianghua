#!/usr/bin/env python3
"""
振幅回测策略 - 从峰值回调 X% 买入，恢复至入场峰值 Y% 卖出

策略逻辑:
  无持仓: 当日最低 <= 当前追踪峰值 * (buy_pct/100) → 以该触发价买入全仓
  有持仓: 当日最高 >= 入场峰值   * (sell_pct/100) → 以该触发价卖出全仓
  T+1  : 买入次日才能卖出
  手续费: 买入 0.1%，卖出 0.2%(含印花税 0.1%)

参数说明:
  buy_pct=90  表示当价格跌至峰值 90% (即跌 10%) 时买入，成交价 = 峰值 × 90%
  sell_pct=105 表示当价格涨至入场峰值 105% 时卖出，成交价 = 入场峰值 × 105%

用法示例:
  python backtest_amplitude.py --symbol 601288 --start 20150101 --end 20241231 --buy-pct 90 --sell-pct 105
  python backtest_amplitude.py --symbol 601288 --start 20200101 --end 20241231 --buy-pct 85 --sell-pct 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_DIR     = Path(__file__).parent.parent / "data"
BACKTEST_DIR = Path(__file__).parent.parent / "backtest"
BACKTEST_DIR.mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BUY_FEE  = 0.001   # 买入手续费率
SELL_FEE = 0.002   # 卖出手续费率 + 印花税


# ─────────────────────────── 数据加载 ───────────────────────────

def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start, format='%Y%m%d')
    end_dt   = pd.to_datetime(end,   format='%Y%m%d')
    best_df  = None
    for fpath in sorted(DATA_DIR.glob(f"{symbol}_daily_*_*_qfq.csv")):
        df = pd.read_csv(fpath, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'])
        sub = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)].copy()
        if not sub.empty and (best_df is None or len(sub) > len(best_df)):
            best_df = sub
    if best_df is None:
        raise FileNotFoundError(
            f"找不到 {symbol} 在 {start}~{end} 的日线数据，请先运行:\n"
            f"  python fetch_stock.py --symbol {symbol} --start {start} --end {end}"
        )
    return best_df.sort_values('date').reset_index(drop=True)


# ─────────────────────────── 回测引擎 ───────────────────────────

def run_backtest(df: pd.DataFrame, buy_pct: float, sell_pct: float,
                 init_cash: float) -> dict:
    """
    peak      : 无持仓阶段持续追踪的历史最高价（卖出后从当日收盘重置）
    entry_peak: 本次买入时记录的峰值，决定卖出目标价，持仓期间不变
    buy_date  : 买入日期，用于 T+1 判断
    """
    buy_r  = buy_pct  / 100.0
    sell_r = sell_pct / 100.0
    cash   = init_cash
    shares = 0
    peak   = float(df.loc[0, 'high'])
    entry_peak = 0.0
    buy_date   = None
    trades, equity_list = [], []

    for idx in range(len(df)):
        row   = df.iloc[idx]
        date  = row['date']
        high  = float(row['high'])
        low   = float(row['low'])
        close = float(row['close'])

        # ── 卖出（T+1：买入次日才能卖）──
        if shares > 0 and buy_date is not None and date > buy_date:
            sell_target = entry_peak * sell_r
            if high >= sell_target:
                revenue = shares * sell_target * (1 - SELL_FEE)
                cash += revenue
                profit_pct = (sell_target / (entry_peak * buy_r) - 1) * 100
                trades.append({
                    'date': date, 'action': 'SELL',
                    'price': round(sell_target, 4),
                    'shares': shares,
                    'peak': round(entry_peak, 4),
                    'profit_pct': round(profit_pct, 2),
                })
                shares = 0
                entry_peak = 0.0
                buy_date = None
                peak = close  # 卖出后从当日收盘重新追踪峰值

        # ── 买入（无持仓）──
        if shares == 0:
            peak = max(peak, high)
            buy_target = peak * buy_r
            if low <= buy_target and buy_target > 0:
                n = int(cash / (buy_target * (1 + BUY_FEE)) / 100) * 100
                if n >= 100:
                    cash -= n * buy_target * (1 + BUY_FEE)
                    shares = n
                    entry_peak = peak
                    buy_date = date
                    trades.append({
                        'date': date, 'action': 'BUY',
                        'price': round(buy_target, 4),
                        'shares': n,
                        'peak': round(peak, 4),
                        'profit_pct': 0.0,
                    })

        equity_list.append({'date': date, 'equity': cash + shares * close})

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=['date', 'action', 'price', 'shares', 'peak', 'profit_pct'])
    equity_df = pd.DataFrame(equity_list)
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    return {
        'equity_df': equity_df,
        'trades_df': trades_df,
        'final_equity': cash + shares * float(df.iloc[-1]['close']),
        'init_cash': init_cash,
    }


# ─────────────────────────── 绩效计算 ───────────────────────────

def calc_metrics(result: dict, df: pd.DataFrame) -> dict:
    eq    = result['equity_df'].set_index('date')['equity']
    init  = result['init_cash']
    final = result['final_equity']
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)

    total_ret = (final / init - 1) * 100
    ann_ret   = ((final / init) ** (1 / years) - 1) * 100

    rm    = eq.cummax()
    dd    = (eq - rm) / rm
    max_dd = abs(dd.min()) * 100

    dr     = eq.pct_change().dropna()
    excess = dr - 0.03 / 252
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
    calmar = ann_ret / max_dd if max_dd > 0 else 0.0

    sells = result['trades_df']
    sells = sells[sells['action'] == 'SELL'] if not sells.empty else sells
    win_rate   = float((sells['profit_pct'] > 0).mean() * 100) if len(sells) > 0 else 0.0
    avg_profit = float(sells['profit_pct'].mean())             if len(sells) > 0 else 0.0

    # 月度收益（以初始资金为第一个月的起始值）
    month_end = eq.resample('ME').last()
    monthly_rets: dict[str, float] = {}
    prev = init
    for ts, val in month_end.items():
        monthly_rets[ts.strftime('%Y-%m')] = round((val / prev - 1) * 100, 2)
        prev = val

    # 年度收益
    year_end = eq.resample('YE').last()
    yearly_rets: dict[int, float] = {}
    prev = init
    for ts, val in year_end.items():
        yearly_rets[ts.year] = round((val / prev - 1) * 100, 2)
        prev = val

    # 买持基准
    bh_final = init * (float(df['close'].iloc[-1]) / float(df['close'].iloc[0]))
    bh_ret   = (bh_final / init - 1) * 100
    bh_ann   = ((bh_final / init) ** (1 / years) - 1) * 100

    return {
        'total_ret':    round(total_ret,   2),
        'ann_ret':      round(ann_ret,     2),
        'max_dd':       round(max_dd,      2),
        'sharpe':       round(sharpe,      3),
        'calmar':       round(calmar,      3),
        'win_rate':     round(win_rate,    2),
        'avg_profit':   round(avg_profit,  2),
        'n_trades':     len(sells),
        'years':        round(years,       2),
        'monthly_rets': monthly_rets,
        'yearly_rets':  yearly_rets,
        'bh_ret':       round(bh_ret,      2),
        'bh_ann':       round(bh_ann,      2),
        'dd_series':    dd,
        'eq_series':    eq,
    }


# ─────────────────────────── 报告输出 ───────────────────────────

def _build_report(m: dict, buy_pct: float, sell_pct: float, symbol: str) -> str:
    """
    构建 Markdown 格式的单次回测报告。

    包含：核心绩效表、年度收益表、月度收益热力表、风险提示。
    """
    lines = [
        f"# {symbol} 振幅策略回测报告",
        "",
        f"> 买入触发 = **{buy_pct}%**  |  卖出触发 = **{sell_pct}%**  |  回测年限 = **{m['years']:.2f} 年**",
        "",
        "---",
        "",
        "## 核心绩效",
        "",
        "| 指标 | 策略 | 买持基准 | 说明 |",
        "| :--- | ---: | ---: | :--- |",
        f"| **总收益率** | **{m['total_ret']:.2f}%** | {m['bh_ret']:.2f}% | 整个回测期间的累计收益 |",
        f"| **年化收益率** | **{m['ann_ret']:.2f}%** | {m['bh_ann']:.2f}% | 复利折算的年均收益率 |",
        f"| **最大回撤** | **{m['max_dd']:.2f}%** | - | 净值从峰值到谷底的最大跌幅 |",
        f"| **夏普比率** | **{m['sharpe']:.3f}** | - | 每单位波动风险的超额回报 (>1为优) |",
        f"| **卡玛比率** | **{m['calmar']:.3f}** | - | 年化收益 / 最大回撤 (越大越好) |",
        f"| **胜率** | **{m['win_rate']:.2f}%** | - | 盈利交易占总交易的比例 ({m['n_trades']}笔) |",
        f"| **平均单笔盈亏** | **{m['avg_profit']:.2f}%** | - | 每笔完整交易的平均收益率 |",
        "",
        "---",
        "",
        "## 年度收益率",
        "",
        "| 年份 | 收益率 | 趋势 |",
        "| :--- | ---: | :---: |",
    ]
    yr_vals = list(m['yearly_rets'].values())
    for yr, ret in m['yearly_rets'].items():
        trend = '📈' if ret >= 0 else '📉'
        lines.append(f"| {yr} | {ret:.2f}% | {trend} |")
    avg_yr = sum(yr_vals) / len(yr_vals) if yr_vals else 0
    lines.append(f"| **平均** | **{avg_yr:.2f}%** | - |")

    # 月度收益表
    lines += ["", "---", "", "## 月度收益率 (%)"]
    months_map: dict[int, dict[int, float]] = {}
    for ym, ret in m['monthly_rets'].items():
        y, mo = int(ym[:4]), int(ym[5:])
        months_map.setdefault(y, {})[mo] = ret

    # 表头
    mo_hdr = "| 年份 |" + "|".join(f" {str(mo).zfill(2)} " for mo in range(1, 13)) + "| 全年 |"
    mo_sep = "|:---|" + "|".join(["---:"] * 12) + "|---:|"
    lines += ["", mo_hdr, mo_sep]
    for yr in sorted(months_map):
        cells = [f" {yr} "]
        for mo in range(1, 13):
            v = months_map[yr].get(mo)
            cells.append(f" {v:.1f} " if v is not None else " - ")
        ann_v = m['yearly_rets'].get(yr)
        cells.append(f" **{ann_v:.1f}** " if ann_v is not None else " - ")
        lines.append("|" + "|".join(cells) + "|")

    all_mr = list(m['monthly_rets'].values())
    if all_mr:
        good = sum(1 for v in all_mr if v > 0)
        lines += [
            "",
            f"> 月均收益: **{sum(all_mr)/len(all_mr):.2f}%**  |  "
            f"盈利月份: **{good}/{len(all_mr)}** ({good/len(all_mr)*100:.0f}%)",
        ]

    lines += [
        "", "---", "",
        "> **风险提示**: 回测结果不代表未来收益，A股受政策影响大。",
        "",
    ]
    return "\n".join(lines)


def print_report(m: dict, buy_pct: float, sell_pct: float, symbol: str,
                 txt_path: Path | None = None) -> None:
    """打印并保存回测报告（Markdown 格式）。"""
    report = _build_report(m, buy_pct, sell_pct, symbol)
    print(report)
    if txt_path is not None:
        # 自动将 .txt 后缀改为 .md
        md_path = txt_path.with_suffix('.md')
        md_path.write_text(report, encoding='utf-8')
        print(f"[报告MD]   -> {md_path}")


# ─────────────────────────── 绘图 ───────────────────────────

def plot_result(result: dict, m: dict, df: pd.DataFrame,
                buy_pct: float, sell_pct: float,
                symbol: str, start: str, end: str) -> Path:
    eq        = m['eq_series']
    init      = result['init_cash']
    strat_nav = eq / init
    bh_close  = df.set_index('date')['close']
    bh_close.index = pd.to_datetime(bh_close.index)
    bh_nav    = bh_close / float(bh_close.iloc[0])
    dd_pct    = m['dd_series'] * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(strat_nav.index, strat_nav.values,
             label=f'振幅策略 ({m["ann_ret"]:.1f}%/yr)', color='#c0392b', lw=1.8)
    ax1.plot(bh_nav.index, bh_nav.values,
             label=f'买持基准 ({m["bh_ann"]:.1f}%/yr)', color='#2980b9', lw=1.2, alpha=0.75)
    ax1.axhline(1.0, color='#95a5a6', ls='--', lw=0.8)

    trades_df = result['trades_df'].copy()
    if not trades_df.empty:
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        for _, t in trades_df[trades_df['action'] == 'BUY'].iterrows():
            if t['date'] in strat_nav.index:
                ax1.scatter(t['date'], strat_nav.loc[t['date']],
                            marker='^', c='#c0392b', s=35, zorder=5)
        for _, t in trades_df[trades_df['action'] == 'SELL'].iterrows():
            if t['date'] in strat_nav.index:
                ax1.scatter(t['date'], strat_nav.loc[t['date']],
                            marker='v', c='#27ae60', s=35, zorder=5)

    ax1.set_title(
        f'{symbol}  振幅策略  买入={buy_pct}% / 卖出={sell_pct}%\n'
        f'年化 {m["ann_ret"]:.1f}%  最大回撤 {m["max_dd"]:.1f}%  '
        f'夏普 {m["sharpe"]:.2f}  胜率 {m["win_rate"]:.0f}%  '
        f'交易 {m["n_trades"]} 笔',
        fontsize=11)
    ax1.set_ylabel('净值')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    ax2.fill_between(dd_pct.index, dd_pct.values, 0,
                     alpha=0.6, color='#c0392b')
    ax2.set_ylabel('回撤 (%)')
    ax2.set_xlabel('日期')
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    out = BACKTEST_DIR / f"{symbol}_amplitude_{buy_pct}_{sell_pct}_{start}_{end}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表]    → {out}")
    return out


# ─────────────────────────── CLI ───────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='振幅回测策略',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--symbol',   required=True, help='股票代码，如 601288')
    parser.add_argument('--start',    required=True, help='回测开始 YYYYMMDD')
    parser.add_argument('--end',      required=True, help='回测结束 YYYYMMDD')
    parser.add_argument('--buy-pct',  type=float, default=90.0,
                        help='买入触发比例：峰值的 X%%, 如 90 = 跌至峰值 90%% 买入 (默认 90)')
    parser.add_argument('--sell-pct', type=float, default=105.0,
                        help='卖出触发比例：入场峰值的 Y%%, 如 105 = 涨至峰值 105%% 卖出 (默认 105)')
    parser.add_argument('--cash', type=float, default=100_000.0,
                        help='初始资金，默认 100000')
    args = parser.parse_args()

    if args.buy_pct >= args.sell_pct:
        print(f"[ERROR] --buy-pct({args.buy_pct}) 必须小于 --sell-pct({args.sell_pct})",
              file=sys.stderr)
        sys.exit(1)

    try:
        df = load_data(args.symbol, args.start, args.end)
        print(f"[数据]    {len(df)} 行  "
              f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")

        result  = run_backtest(df, args.buy_pct, args.sell_pct, args.cash)
        metrics = calc_metrics(result, df)

        txt_path = (BACKTEST_DIR /
                    f"{args.symbol}_report_{args.buy_pct}_{args.sell_pct}"
                    f"_{args.start}_{args.end}.txt")
        print_report(metrics, args.buy_pct, args.sell_pct, args.symbol, txt_path)
        plot_result(result, metrics, df, args.buy_pct, args.sell_pct,
                    args.symbol, args.start, args.end)

        # 月度收益 CSV
        mr_path = (BACKTEST_DIR /
                   f"{args.symbol}_monthly_{args.buy_pct}_{args.sell_pct}"
                   f"_{args.start}_{args.end}.csv")
        pd.DataFrame([
            {'月份': k, '收益率(%)': v}
            for k, v in metrics['monthly_rets'].items()
        ]).to_csv(mr_path, index=False, encoding='utf-8-sig')
        print(f"[月度收益] → {mr_path}")

        # 交易记录 CSV
        if not result['trades_df'].empty:
            tr_path = (BACKTEST_DIR /
                       f"{args.symbol}_trades_{args.buy_pct}_{args.sell_pct}"
                       f"_{args.start}_{args.end}.csv")
            result['trades_df'].to_csv(tr_path, index=False, encoding='utf-8-sig')
            print(f"[交易记录] {len(result['trades_df'])} 条 → {tr_path}")

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
