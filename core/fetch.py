"""
行情数据拉取模块
================
封装 akshare（主路径）+ baostock（备用路径）的日线数据拉取。
拉取结果只含原始 OHLCV，技术指标由 core/indicators.py 统一计算。

用法：
  from core.fetch import fetch_daily, load_cached
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# 东方财富接口绕过代理（防止企业/家用代理导致拉取失败）
_em_domains = (
    "push2his.eastmoney.com,push2.eastmoney.com,"
    "datacenter.eastmoney.com,data.eastmoney.com"
)
_existing = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
os.environ["NO_PROXY"] = f"{_existing},{_em_domains}".lstrip(",")
os.environ["no_proxy"] = os.environ["NO_PROXY"]

import pandas as pd
import akshare as ak

try:
    import baostock as bs
    _HAS_BS = True
except ImportError:
    _HAS_BS = False

# 项目根目录（lianghua/）下的 data/ 文件夹
ROOT    = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════

def load_cached(symbol: str, start: str, end: str,
                adjust: str = "qfq") -> pd.DataFrame | None:
    """
    尝试从 data/ 读取本地缓存的日线数据。
    找到则返回 DataFrame，否则返回 None。
    缓存文件名格式：{symbol}_daily_{start}_{end}_{adjust}.csv
    """
    start_dt = pd.to_datetime(start, format="%Y%m%d")
    end_dt   = pd.to_datetime(end,   format="%Y%m%d")
    best: pd.DataFrame | None = None

    for fpath in sorted(DATA_DIR.glob(f"{symbol}_daily_*_*_{adjust}.csv")):
        try:
            df  = pd.read_csv(fpath, encoding="utf-8-sig")
            df["date"] = pd.to_datetime(df["date"])
            sub = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
            if not sub.empty and (best is None or len(sub) > len(best)):
                best = sub
        except Exception:
            continue

    if best is not None:
        return best.sort_values("date").reset_index(drop=True)
    return None


def fetch_daily(symbol: str, start: str, end: str,
                adjust: str = "qfq",
                quiet: bool = False) -> pd.DataFrame:
    """
    拉取日线 OHLCV 数据（前复权为默认）。
    优先读本地缓存；无缓存时依次尝试 akshare → baostock。
    拉取后自动保存到 data/ 目录。

    Parameters
    ----------
    symbol : 股票代码，如 "601288"
    start  : 开始日期 "YYYYMMDD"
    end    : 结束日期 "YYYYMMDD"
    adjust : 复权方式 qfq/hfq/none
    quiet  : True 时不打印日志（并行子进程使用）
    """
    cached = load_cached(symbol, start, end, adjust)
    if cached is not None:
        if not quiet:
            print(f"[fetch] {symbol} 本地缓存 {len(cached)} 行  "
                  f"{cached['date'].iloc[0].date()} ~ {cached['date'].iloc[-1].date()}")
        return cached

    if not quiet:
        print(f"[fetch] {symbol} 无缓存，拉取 {start}~{end} ...")

    df = _fetch_akshare(symbol, start, end, adjust, quiet)
    if df is None:
        df = _fetch_baostock(symbol, start, end, adjust, quiet)

    df = _clean(df)
    _save(df, symbol, start, end, adjust, quiet)
    return df


# ══════════════════════════════════════════════════════
# 内部实现
# ══════════════════════════════════════════════════════

def _fetch_akshare(symbol: str, start: str, end: str,
                   adjust: str, quiet: bool) -> pd.DataFrame | None:
    adj_map = {"qfq": "qfq", "hfq": "hfq", "none": ""}
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start, end_date=end,
            adjust=adj_map.get(adjust, "qfq"),
        )
        if df.empty:
            raise ValueError("空数据")
        if not quiet:
            print(f"[fetch] akshare 成功 {len(df)} 行")
        return df
    except Exception as e:
        if not quiet:
            print(f"[fetch] akshare 失败: {e}，尝试 baostock ...")
        return None


def _fetch_baostock(symbol: str, start: str, end: str,
                    adjust: str, quiet: bool) -> pd.DataFrame:
    if not _HAS_BS:
        raise RuntimeError("akshare 失败且未安装 baostock，请: pip install baostock")

    prefix  = "sh" if symbol.startswith(("6", "9")) else "sz"
    bs_code = f"{prefix}.{symbol}"
    adj_flag = {"qfq": "2", "hfq": "1", "none": "3"}.get(adjust, "2")
    s = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    e = f"{end[:4]}-{end[4:6]}-{end[6:]}"

    if not quiet:
        print(f"[fetch] baostock 拉取 {bs_code} ...")
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,turn,pctChg",
            start_date=s, end_date=e,
            frequency="d", adjustflag=adj_flag,
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()

    if not rows:
        raise ValueError(f"baostock 无数据：{symbol}")

    df = pd.DataFrame(rows, columns=rs.fields)
    df = df.rename(columns={
        "date": "日期", "open": "开盘", "close": "收盘",
        "high": "最高", "low": "最低", "volume": "成交量",
        "amount": "成交额", "turn": "换手率", "pctChg": "涨跌幅",
    })
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名、类型、排序。"""
    col_map = {
        "日期": "date",   "开盘": "open",   "收盘": "close",
        "最高": "high",   "最低": "low",    "成交量": "volume",
        "成交额": "amount", "振幅": "amplitude_pct",
        "涨跌幅": "chg_pct", "涨跌额": "chg_amt", "换手率": "turnover_pct",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    num_cols = ["open", "close", "high", "low", "volume", "amount",
                "amplitude_pct", "chg_pct", "chg_amt", "turnover_pct"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def _save(df: pd.DataFrame, symbol: str, start: str, end: str,
          adjust: str, quiet: bool) -> None:
    fname = f"{symbol}_daily_{start}_{end}_{adjust}.csv"
    path  = DATA_DIR / fname
    df.to_csv(path, index=False, encoding="utf-8-sig")
    if not quiet:
        print(f"[fetch] 已保存 → {path}")


# ══════════════════════════════════════════════════════
# 小时线
# ══════════════════════════════════════════════════════

def fetch_hourly(symbol: str, start: str, end: str,
                 adjust: str = "qfq",
                 quiet: bool = False) -> pd.DataFrame:
    """
    拉取60分钟K线。akshare 对历史深度有限制，按季度分批拉取并合并。

    Parameters
    ----------
    symbol : 股票代码，如 "601288"
    start  : 开始日期 "YYYYMMDD"
    end    : 结束日期 "YYYYMMDD"
    adjust : 复权方式 qfq/hfq/none
    quiet  : True 时不打印日志
    """
    from datetime import datetime, timedelta

    adj_map = {"qfq": "qfq", "hfq": "hfq", "none": ""}
    adj_val = adj_map.get(adjust, "qfq")

    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt   = datetime.strptime(end,   "%Y%m%d")

    import time as _time
    chunks = []
    cur = start_dt
    batch_days = 90

    while cur <= end_dt:
        chunk_end = min(cur + timedelta(days=batch_days), end_dt)
        s = cur.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        if not quiet:
            print(f"[fetch] 小时线 {symbol}  {s} → {e}")
        try:
            df_chunk = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period="60",
                start_date=s + " 09:00:00",
                end_date=e   + " 15:30:00",
                adjust=adj_val,
            )
            if not df_chunk.empty:
                chunks.append(df_chunk)
        except Exception as exc:
            if not quiet:
                print(f"  [WARN] {s}~{e} 拉取失败: {exc}，跳过")
        cur = chunk_end + timedelta(days=1)
        _time.sleep(0.3)

    if not chunks:
        raise ValueError(f"未获取到任何小时线数据：{symbol}")

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates()
    return _clean_hourly(df)


def _clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "时间": "datetime", "开盘": "open",  "收盘": "close",
        "最高": "high",     "最低": "low",   "成交量": "volume",
        "成交额": "amount", "涨跌幅": "chg_pct", "涨跌额": "chg_amt",
        "振幅": "amplitude_pct", "换手率": "turnover_pct",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    num_cols = ["open", "close", "high", "low", "volume", "amount",
                "chg_pct", "chg_amt", "amplitude_pct", "turnover_pct"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
