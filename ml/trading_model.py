#!/usr/bin/env python3
"""
端到端 LSTM 交易模型 v2
========================
输入  : 历史 OHLCV + 技术指标（昨日全量）+ 今日开盘价（仅此一项）
输出  : 今日目标仓位 ∈ [0, 1]  （正数做多，0 空仓；A股不做空）
         正向变化 = 今日买入，负向变化 = 今日卖出

关键设计：
  ① 仓位约束 [0, 1]：Sigmoid 输出，天然满足长仓限制
  ② walk-forward 训练：每隔 retrain_freq 天，用当前预测日前所有历史重训
  ③ 序列级 Sharpe 损失：在连续窗口内模拟交易 P&L，直接最大化夏普代理
  ④ 信息隔离：预测日 t 的特征向量里 close[t]/high[t]/low[t] 均不可见
  ⑤ 手续费千分之一：按仓位变化量双边扣收

用法：
  # 标准 walk-forward（推荐）
  python ml/trading_model.py --symbol 002837 --start 20200101

  # 自定义超参数
  python ml/trading_model.py --symbol 002837 --start 20200101 \\
      --epochs 80 --hidden 128 --seq-len 40 --retrain-freq 20

  # 关闭 walk-forward（快速测试）
  python ml/trading_model.py --symbol 002837 --start 20200101 \\
      --no-walk-forward --train-ratio 0.7 --epochs 50
"""

import argparse
import sys
from pathlib import Path
from datetime import date

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.fetch import fetch_daily
from core.indicators import compute_all

# ── 中文字体（Windows 兼容） ──
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CHARTS_DIR = ROOT / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════
#  超参数默认值（所有参数均可通过 CLI 调整）
#  ──────────────────────────────────────────────────
#  调参建议：
#    seq_len     : 越长捕捉趋势越好，但需要更多数据；建议 20~60
#    hidden_dim  : 64/128/256，数据少时用小值防止过拟合
#    num_layers  : 1~3，层越深表达力越强但更难训练
#    dropout     : 0.1~0.3，数据少时增大
#    lr          : 学习率太大震荡，太小不收敛；0.5e-3~2e-3 常用
#    epochs      : walk-forward 模式建议 40~80（每次重训）
#    warm_up_days: 初始预热天数，越大初始训练集越充足
#    retrain_freq: 越小预测越准确但计算越慢；建议 10~30
#    fee_lambda  : 手续费惩罚倍率，越大越保守（减少频繁交易）
# ══════════════════════════════════════════════════════

DEFAULT_HP: dict = {
    # ── 序列 ──────────────────────────────────────────
    "seq_len":       60,      # 回看窗口（交易日），越大记忆越长
    # ── 模型 ──────────────────────────────────────────
    "hidden_dim":    128,      # LSTM 隐层维度（32/64/128/256）
    "num_layers":    4,       # LSTM 层数（1～4）
    "dropout":       0.2,     # Dropout（单层时自动关闭）
    # ── 优化 ──────────────────────────────────────────
    "lr":            1e-3,    # 初始学习率（余弦退火到 lr*0.01）
    "epochs":        60,      # 每次 walk-forward 重训轮数
    "batch_size":    32,      # 批大小
    "weight_decay":  1e-5,    # L2 正则化强度
    "grad_clip":     1.0,     # 梯度裁剪阈值（防止梯度爆炸）
    # ── Walk-forward ────────────────────────────────
    "walk_forward":  True,    # 是否启用 walk-forward 滚动训练
    "warm_up_days":  252,     # 初始训练集天数（约 1 年）
    "retrain_freq":  1,      # 每隔多少天重训一次（约 1 个月）
    # ── 策略 ──────────────────────────────────────────
    "train_ratio":   0.7,     # 简单模式下的训练集比例
    "fee_rate":      0.001,   # 单边手续费（千分之一）
    "max_pos":       1.0,     # 最大仓位（1 = 满仓，不做空）
    "fee_lambda":    1.5,     # 损失函数中的手续费惩罚倍率
    "loss_type":     "sharpe", # 损失函数类型：sharpe（夏普代理）或 pnl（纯P&L）
}


# ══════════════════════════════════════════════════════
#  特征工程
#  ──────────────────────────────────────────────────
#  【信息隔离规则】
#  预测日 t 的特征向量：
#    ✅ open[t]          - 今日开盘价（唯一可见的当日信息）
#    ✅ close/high/low/volume[t-1]  前日全部信息
#    ✅ 所有技术指标[t-1]           基于前日的指标
#    ❌ close[t] / high[t] / low[t]  完全掩盖
# ══════════════════════════════════════════════════════

# 以当日收盘价归一化的均线指标
_PRICE_INDS = {"ma5", "ma20", "ema12", "ema26", "bb_upper", "bb_mid", "bb_lower"}
# 以收盘价比例归一化的 MACD 相关指标
_SCALED_INDS = {"diff", "dea", "macd"}


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    对每个预测日 t（t ≥ 1），构建仅用开盘时信息的特征向量。

    特征共 23 维：
      [0]   : 今日开盘缺口 (open[t]/close[t-1] - 1)
      [1-4] : 前日 OHLC 变化率（相对更前一日收盘）
      [5]   : 前日成交量比率（前日量 / 近5日均量）
      [6-9] : 过去4日收益率序列（close-to-close，捕捉短期动量）
      [10-22]: 技术指标（ma5/ma20/ema12/ema26/diff/dea/macd/rsi14/bb系列）

    数据处理：NaN→0, Inf→clip, 最终裁剪到 [-5, 5]
    返回 shape = (len(df)-1, feat_dim)
    """
    df = df.reset_index(drop=True)
    rows = []

    for t in range(1, len(df)):
        # ── 前日收盘价（归一化基准） ──
        c_prev = float(df["close"].iloc[t - 1])
        if pd.isna(c_prev) or c_prev == 0:
            c_prev = 1e-8

        row: list[float] = []

        # ① 今日开盘缺口（仅当日信息中的开盘价）
        row.append(float(df["open"].iloc[t] / c_prev - 1))

        # ② 前一日 OHLC 变化率（相对更前一日收盘，反映整体日K形态）
        c_prev2 = float(df["close"].iloc[t - 2]) if t >= 2 else c_prev
        if pd.isna(c_prev2) or c_prev2 == 0:
            c_prev2 = c_prev
        row += [
            float(df["open"].iloc[t - 1]  / c_prev2 - 1),   # 前日开盘
            float(df["close"].iloc[t - 1] / c_prev2 - 1),    # 前日收盘
            float(df["high"].iloc[t - 1]  / c_prev2 - 1),    # 前日最高
            float(df["low"].iloc[t - 1]   / c_prev2 - 1),    # 前日最低
        ]

        # ③ 成交量比率（前一日成交量 / 近5日均量，衡量放量程度）
        if "volume" in df.columns:
            vol    = float(df["volume"].iloc[t - 1])
            vol_ma = float(df["volume"].iloc[max(0, t - 6): t - 1].mean())
            row.append(float(vol / (vol_ma + 1e-8) - 1) if not pd.isna(vol_ma) else 0.0)
        else:
            row.append(0.0)

        # ④ 过去4日收盘收益序列（短期动量特征，close-to-close）
        for lag in range(1, 5):
            i0 = t - lag - 1
            i1 = t - lag
            if i0 >= 0 and i1 >= 0:
                c0 = float(df["close"].iloc[i0])
                c1 = float(df["close"].iloc[i1])
                row.append(float(c1 / (c0 + 1e-8) - 1) if c0 != 0 else 0.0)
            else:
                row.append(0.0)

        # ⑤ 技术指标（前一日值，各自归一化）
        for col in ["ma5", "ma20", "ema12", "ema26",
                    "diff", "dea", "macd",
                    "rsi14",
                    "bb_upper", "bb_mid", "bb_lower", "bb_width"]:
            if col not in df.columns:
                row.append(0.0)
                continue
            val = df[col].iloc[t - 1]
            if pd.isna(val):
                row.append(0.0)
                continue
            if col in _PRICE_INDS:
                # 均线类：相对当日收盘价的偏差（反映价格偏离程度）
                row.append(float(val / c_prev - 1))
            elif col in _SCALED_INDS:
                # MACD 类：相对收盘价的比例
                row.append(float(val / (abs(c_prev) + 1e-8)))
            elif col == "rsi14":
                # RSI：中心化到 [-0.5, 0.5]——50分界线为0
                row.append(float(val / 100.0 - 0.5))
            elif col == "bb_width":
                # 布林带宽：直接使用（已是相对量）
                row.append(float(val))
            else:
                row.append(float(val))

        rows.append(row)

    arr = np.array(rows, dtype=np.float32)
    # 清理异常值
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -5.0, 5.0)
    return arr


def build_returns(df: pd.DataFrame) -> np.ndarray:
    """
    计算每日开盘→收盘收益。
    信号在开盘执行（以开盘价建仓），收盘结算。
    长度 = len(df) - 1，与 build_features 严格对齐。
    """
    opens  = df["open"].values[1:].astype(np.float32)
    closes = df["close"].values[1:].astype(np.float32)
    return closes / (opens + 1e-8) - 1.0


# ══════════════════════════════════════════════════════
#  数据集（窗口滑动）
# ══════════════════════════════════════════════════════

class TradingDataset(Dataset):
    """
    滑动窗口数据集。
    每个样本 = (features[i:i+seq_len], returns[i:i+seq_len])
    用于序列级损失计算。
    """
    def __init__(self, features: np.ndarray, returns: np.ndarray, seq_len: int):
        assert len(features) == len(returns), "特征与收益率长度必须一致"
        self.X       = torch.from_numpy(features)     # (N, feat_dim)
        self.R       = torch.from_numpy(returns)       # (N,)
        self.seq_len = seq_len

    def __len__(self) -> int:
        # 每个样本需要 seq_len 天的历史
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx: int):
        # 返回连续 seq_len 天的特征和收益率（序列完整，用于 Sharpe 损失）
        x = self.X[idx: idx + self.seq_len]    # (seq_len, feat_dim)
        r = self.R[idx: idx + self.seq_len]    # (seq_len,)
        return x, r


# ══════════════════════════════════════════════════════
#  模型：LSTM + Sigmoid 头（长仓约束）
# ══════════════════════════════════════════════════════

class TradingLSTM(nn.Module):
    """
    LSTM 编码器 + 双层 MLP 头 → Sigmoid → 仓位 ∈ (0, 1)

    输出语义：
      0.0  = 完全空仓
      0.5  = 半仓
      1.0  = 满仓
      相邻时刻的差值 = 当日交易量（正=买入，负=卖出）
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        # LSTM 核心：捕捉时序依赖关系
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 输出头：LayerNorm → Linear → GELU → Dropout → Linear → Sigmoid
        # Sigmoid 保证输出在 (0,1)，对应长仓约束
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),   # ← 关键：输出天然满足 [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, feat_dim)
        返回: (B, seq_len) — 每个时刻的目标仓位
        """
        # lstm_out: (B, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        # 对每个时刻都输出仓位，用于序列级损失
        pos = self.head(lstm_out)     # (B, seq_len, 1)
        return pos.squeeze(-1)        # (B, seq_len)


# ══════════════════════════════════════════════════════
#  损失函数：序列级 Sharpe 代理
#  ──────────────────────────────────────────────────
#  思路：在长度 seq_len 的时间窗口内模拟完整交易过程：
#    1. 跟踪每日仓位变化（delta = pos[t] - pos[t-1]）
#    2. 计算手续费（|delta| × fee_rate × fee_lambda）
#    3. 计算净损益（pos[t] × return[t] - fee[t]）
#    4. 最小化 -Sharpe（= 最大化 Sharpe）
# ══════════════════════════════════════════════════════

def sharpe_loss(positions: torch.Tensor, returns: torch.Tensor,
                fee_rate: float, fee_lambda: float) -> torch.Tensor:
    """
    序列级 Sharpe 代理损失。

    参数:
      positions : (B, seq_len) 目标仓位序列（来自模型输出，[0,1]）
      returns   : (B, seq_len) 对应的开盘→收盘收益率
      fee_rate  : 单边手续费率（千分之一 = 0.001）
      fee_lambda: 手续费放大系数（越大越保守）

    返回:
      标量 loss（越小 = Sharpe 越高）
    """
    # 计算仓位变化量：prev[0]=0（初始空仓），prev[t]=pos[t-1]
    # shape: (B, seq_len)
    zeros    = torch.zeros(positions.shape[0], 1, device=positions.device)
    prev_pos = torch.cat([zeros, positions[:, :-1]], dim=1)
    delta    = positions - prev_pos

    # 手续费：按仓位变化量扣除，fee_lambda 控制保守程度
    fees = fee_rate * fee_lambda * delta.abs()

    # 净损益：当日仓位 × 当日收益 - 手续费
    pnl = positions * returns - fees  # (B, seq_len)

    # 展平到全局再计算 Sharpe（跨所有批次、所有时刻）
    pnl_flat = pnl.reshape(-1)
    sharpe   = pnl_flat.mean() / (pnl_flat.std() + 1e-8)

    return -sharpe   # 最小化负 Sharpe = 最大化 Sharpe


def pnl_loss(positions: torch.Tensor, returns: torch.Tensor,
             fee_rate: float, fee_lambda: float) -> torch.Tensor:
    """
    纯 P&L 损失：直接最大化累计净收益。

    参数与 sharpe_loss 相同，fee_lambda 同样用于放大手续费惩罚。
    返回标量 loss（越小 = 净收益越高）。
    """
    zeros    = torch.zeros(positions.shape[0], 1, device=positions.device)
    prev_pos = torch.cat([zeros, positions[:, :-1]], dim=1)
    delta    = positions - prev_pos
    fees     = fee_rate * fee_lambda * delta.abs()
    pnl      = positions * returns - fees
    return -pnl.mean()


# ══════════════════════════════════════════════════════
#  训练（单次）
# ══════════════════════════════════════════════════════

def _train_once(features: np.ndarray, returns: np.ndarray,
                hp: dict, device: torch.device,
                model: TradingLSTM | None = None,
                verbose: bool = True) -> TradingLSTM:
    """
    在给定特征/收益率上训练（或继续训练）模型。

    参数:
      features : (N, feat_dim) 特征矩阵
      returns  : (N,) 收益率序列
      hp       : 超参数字典
      device   : 训练设备
      model    : 若提供则继续训练（transfer learning），否则新建
      verbose  : 是否打印训练日志

    返回:
      训练好的 TradingLSTM
    """
    seq_len  = hp["seq_len"]
    feat_dim = features.shape[1]

    dataset = TradingDataset(features, returns, seq_len)
    if len(dataset) < 2:
        raise ValueError(
            f"数据只有 {len(features)} 行，seq_len={seq_len}，样本量不足。"
            f"请减小 --seq-len 或扩大数据范围。"
        )
    loader = DataLoader(
        dataset, batch_size=hp["batch_size"],
        shuffle=True, drop_last=True,
        num_workers=0,   # Windows 兼容
    )

    # 新建或复用模型
    if model is None:
        model = TradingLSTM(
            feat_dim, hp["hidden_dim"], hp["num_layers"], hp["dropout"]
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp["epochs"], eta_min=hp["lr"] * 0.01
    )

    log_every = max(1, hp["epochs"] // 5)   # 每训练 20% 打印一次

    model.train()
    for epoch in range(1, hp["epochs"] + 1):
        epoch_loss = 0.0
        for x_batch, r_batch in loader:
            x_batch = x_batch.to(device)
            r_batch = r_batch.to(device)

            # 前向：输出 (B, seq_len) 的仓位序列
            pos  = model(x_batch) * hp["max_pos"]
            _loss_fn = pnl_loss if hp.get("loss_type", "sharpe") == "pnl" else sharpe_loss
            loss = _loss_fn(pos, r_batch, hp["fee_rate"], hp["fee_lambda"])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        if verbose and epoch % log_every == 0:
            print(f"      Epoch {epoch:3d}/{hp['epochs']}  "
                  f"loss={epoch_loss / len(loader):.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    return model


# ══════════════════════════════════════════════════════
#  推理（逐日滑动窗口）
# ══════════════════════════════════════════════════════

@torch.no_grad()
def _predict_days(model: TradingLSTM, features: np.ndarray,
                  hp: dict, device: torch.device) -> np.ndarray:
    """
    对 features 逐日推理，返回目标仓位数组。

    输出长度 = len(features) - seq_len + 1
    （前 seq_len-1 天因历史不足跳过，填 0.0）
    """
    seq_len   = hp["seq_len"]
    max_pos   = hp["max_pos"]
    model.eval()
    positions = []

    for i in range(seq_len - 1, len(features)):
        x   = torch.from_numpy(
            features[i - seq_len + 1: i + 1]
        ).unsqueeze(0).to(device)          # (1, seq_len, feat_dim)
        # 取最后一步的仓位输出
        pos = model(x)[0, -1].item() * max_pos
        positions.append(float(np.clip(pos, 0.0, max_pos)))

    # 前 seq_len-1 天无预测，填 0（空仓）
    pad = [0.0] * (seq_len - 1)
    return np.array(pad + positions, dtype=np.float32)


# ══════════════════════════════════════════════════════
#  Walk-forward 主流程
#  ──────────────────────────────────────────────────
#  流程：
#    1. 用前 warm_up_days 天训练初始模型
#    2. 从第 warm_up_days 天开始，每隔 retrain_freq 天：
#         a. 用所有历史数据（到当前预测日前一天）重训模型
#         b. 用新模型预测接下来 retrain_freq 天的仓位
#    3. 拼接所有预测仓位，计算完整的 walk-forward 绩效
# ══════════════════════════════════════════════════════

def run_walk_forward(features: np.ndarray, returns: np.ndarray,
                     hp: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward 滚动训练与预测。

    返回:
      positions : (N,) 全周期仓位序列（前 warm_up_days 天为 0）
      retrain_dates : 重训时刻的索引数组（用于标注图表）
    """
    n            = len(features)
    warm_up      = min(hp["warm_up_days"], int(n * 0.7))   # 预热至少要有 70% 数据
    retrain_freq = hp["retrain_freq"]

    if warm_up < hp["seq_len"] + hp["batch_size"]:
        raise ValueError(
            f"预热天数 {warm_up} 不足，请增加 --warm-up 或缩短数据范围。"
        )

    all_positions  = np.zeros(n, dtype=np.float32)  # 全周期仓位（默认空仓）
    retrain_marks  = []   # 记录每次重训的起始索引
    model          = None

    # 计算重训时刻列表（从 warm_up 开始，每隔 retrain_freq 天重训一次）
    retrain_points = list(range(warm_up, n, retrain_freq))
    if not retrain_points:
        retrain_points = [warm_up]

    total_retrain = len(retrain_points)
    for i, t_start in enumerate(retrain_points):
        t_end = min(t_start + retrain_freq, n)  # 本次预测的结束index

        # ── 用当前预测日之前的所有历史数据重训 ──
        train_feat = features[:t_start]
        train_ret  = returns[:t_start]

        print(f"\n  [Walk-forward {i+1}/{total_retrain}] "
              f"训练集: 前 {t_start} 天 → 预测: [{t_start},{t_end})")

        model = _train_once(train_feat, train_ret, hp, device,
                            model=None,    # 每次重新训练，避免灾难遗忘
                            verbose=(i == 0))   # 只显示第一次的详细日志

        retrain_marks.append(t_start)

        # ── 推理：从 t_start 到 t_end 的仓位 ──
        # 为了有足够的历史 context，取 t_start 之前的 seq_len 天作为上下文
        ctx_start  = max(0, t_start - hp["seq_len"] + 1)
        pred_feats = features[ctx_start: t_end]
        pred_pos   = _predict_days(model, pred_feats, hp, device)

        # 对齐：只保留 t_start 到 t_end 的预测
        offset = t_start - ctx_start
        all_positions[t_start: t_end] = pred_pos[offset: offset + (t_end - t_start)]

    return all_positions, np.array(retrain_marks)


# ══════════════════════════════════════════════════════
#  简单模式（单次划分，快速验证）
# ══════════════════════════════════════════════════════

def run_simple(features: np.ndarray, returns: np.ndarray,
               hp: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    单次 train/test 划分模式（--no-walk-forward 时使用）。

    返回: (all_positions, train_mask)
    """
    n       = len(features)
    n_train = int(n * hp["train_ratio"])

    print(f"\n  [简单模式] 训练集: 前 {n_train} 天 | 测试集: 后 {n - n_train} 天")

    model = _train_once(features[:n_train], returns[:n_train], hp, device)

    # 全量推理
    all_pos = _predict_days(model, features, hp, device)

    return all_pos, np.array([n_train])


# ══════════════════════════════════════════════════════
#  绩效计算
# ══════════════════════════════════════════════════════

def calc_performance(positions: np.ndarray, returns: np.ndarray,
                     fee_rate: float) -> dict:
    """
    精确计算策略绩效（手续费基于仓位变化量）。

    手续费计算逻辑：
      - 每次仓位变化产生交易（|delta| = 交易比例）
      - 手续费 = |delta| × fee_rate（买卖均扣收）
      - 符合 A股千分之一手续费标准
    """
    n   = min(len(positions), len(returns))
    pos = positions[:n].astype(np.float64)
    ret = returns[:n].astype(np.float64)

    # 手续费：以仓位变化量计算（首日从0开始）
    delta = np.diff(np.concatenate([[0.0], pos]))
    fees  = fee_rate * np.abs(delta)

    # 每日净损益
    pnl    = pos * ret - fees
    equity = np.cumprod(1.0 + pnl)

    total_ret = float(equity[-1] - 1)
    n_years   = max(n / 252, 1e-6)
    ann_ret   = float((1 + total_ret) ** (1.0 / n_years) - 1)

    sharpe  = float(pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252))
    run_max = np.maximum.accumulate(equity)
    dd      = (equity - run_max) / (run_max + 1e-8)
    max_dd  = float(dd.min())
    calmar  = ann_ret / (abs(max_dd) + 1e-8)

    # 买持基准（仅持股期间）
    bh_ret = float(np.prod(1 + ret) - 1)
    bh_ann = float((1 + bh_ret) ** (1.0 / n_years) - 1)

    # 胜率（每日盈利的天数比例）
    win_days = int((pnl > 0).sum())

    # 换手率统计
    avg_turnover = float(np.abs(delta).mean())
    total_fees   = float(fees.sum())

    # 仓位统计
    avg_pos  = float(pos.mean())
    pos_days = int((pos > 0.05).sum())   # 持仓天数（仓位>5%）

    return {
        "total_ret":    total_ret,
        "ann_ret":      ann_ret,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "calmar":       float(calmar),
        "win_rate":     float(win_days / n),
        "win_days":     win_days,
        "total_days":   n,
        "pos_days":     pos_days,
        "avg_pos":      avg_pos,
        "avg_turnover": avg_turnover,
        "total_fees":   total_fees,
        "bh_ret":       bh_ret,
        "bh_ann":       bh_ann,
        "equity":       equity,
        "pnl":          pnl,
        "positions":    pos,
        "drawdown":     dd,
    }


# ══════════════════════════════════════════════════════
#  报告输出（Markdown + 控制台）
# ══════════════════════════════════════════════════════

def build_md_report(m: dict, symbol: str, start: str, end: str,
                    hp: dict, mode: str,
                    train_m: dict | None = None,
                    test_m:  dict | None = None) -> str:
    """
    生成 Markdown 格式的完整性能报告。

    包含：元信息、超参数表、分段绩效表、字段解释。
    """
    today = date.today().isoformat()
    lines = [
        f"# {symbol} 机器学习交易模型报告",
        "",
        f"> **生成日期**: {today}  |  **模式**: {mode}  |  "
        f"**数据**: {start[:4]}-{start[4:6]}-{start[6:]} ~ {end[:4]}-{end[4:6]}-{end[6:]}",
        "",
        "---",
        "",
        "## 超参数配置",
        "",
        "| 参数 | 值 | 说明 |",
        "| :--- | :--- | :--- |",
        f"| seq_len | {hp['seq_len']} | 回看窗口（交易日） |",
        f"| hidden_dim | {hp['hidden_dim']} | LSTM 隐层宽度 |",
        f"| num_layers | {hp['num_layers']} | LSTM 深度 |",
        f"| dropout | {hp['dropout']} | Dropout 比例 |",
        f"| lr | {hp['lr']} | 初始学习率 |",
        f"| epochs | {hp['epochs']} | 每次重训轮数 |",
    ]
    if hp.get("walk_forward"):
        lines += [
            f"| warm_up_days | {hp['warm_up_days']} | 初始预热天数 |",
            f"| retrain_freq | {hp['retrain_freq']} | 重训间隔（交易日） |",
        ]
    lines += [
        f"| fee_rate | {hp['fee_rate']} | 单边手续费（千分之一） |",
        f"| fee_lambda | {hp['fee_lambda']} | 手续费惩罚倍率 |",
        f"| max_pos | {hp['max_pos']} | 最大仓位 |",
        "",
        "---",
        "",
        "## 整体绩效（全周期）",
        "",
        "| 指标 | 策略 | 买持基准 | 说明 |",
        "| :--- | ---: | ---: | :--- |",
        f"| **总收益率** | **{m['total_ret']*100:+.2f}%** | {m['bh_ret']*100:+.2f}% | 整个回测期累计收益 |",
        f"| **年化收益率** | **{m['ann_ret']*100:+.2f}%** | {m['bh_ann']*100:+.2f}% | 复利折算的年均收益 |",
        f"| **最大回撤** | **{m['max_dd']*100:.2f}%** | - | 净值从峰值到谷底的最大跌幅 |",
        f"| **夏普比率** | **{m['sharpe']:.3f}** | - | 每单位波动的超额收益（>1 为优） |",
        f"| **卡玛比率** | **{m['calmar']:.3f}** | - | 年化收益 / 最大回撤（越大越好） |",
        f"| **日胜率** | **{m['win_rate']*100:.1f}%** | - | 当日盈利天数占比 |",
        f"| **持仓天数** | **{m['pos_days']}** | - | 仓位>5% 的天数 |",
        f"| **平均仓位** | **{m['avg_pos']*100:.1f}%** | - | 全周期平均持仓比例 |",
        f"| **日均换手** | **{m['avg_turnover']*100:.2f}%** | - | 每日平均仓位变化量 |",
        f"| **总手续费** | **{m['total_fees']*100:.3f}%** | - | 合计付出的手续费（相对本金） |",
    ]

    if train_m and test_m:
        lines += [
            "",
            "---",
            "",
            "## 训练集 vs 测试集绩效对比",
            "",
            "| 指标 | 训练集 | 测试集 | 说明 |",
            "| :--- | ---: | ---: | :--- |",
            f"| 总收益率 | {train_m['total_ret']*100:+.2f}% | **{test_m['total_ret']*100:+.2f}%** | |",
            f"| 年化收益率 | {train_m['ann_ret']*100:+.2f}% | **{test_m['ann_ret']*100:+.2f}%** | |",
            f"| 最大回撤 | {train_m['max_dd']*100:.2f}% | **{test_m['max_dd']*100:.2f}%** | |",
            f"| 夏普比率 | {train_m['sharpe']:.3f} | **{test_m['sharpe']:.3f}** | |",
            f"| 卡玛比率 | {train_m['calmar']:.3f} | **{test_m['calmar']:.3f}** | |",
            f"| 日胜率 | {train_m['win_rate']*100:.1f}% | **{test_m['win_rate']*100:.1f}%** | |",
        ]

    lines += [
        "",
        "---",
        "",
        "## 字段与术语解释",
        "",
        "| 术语 | 解释 |",
        "| :--- | :--- |",
        "| **仓位 [0,1]** | 模型输出的目标持仓比例；0=空仓，1=满仓 |",
        "| **Sharpe 损失** | 训练直接最大化夏普比率代理（净 P&L 均值/标准差），"
        "比单纯最大化总收益更稳健 |",
        "| **Walk-forward** | 每隔 retrain_freq 天，用当前日期前所有历史数据重训模型，"
        "模拟真实上线时的运行方式，避免未来数据泄露 |",
        "| **手续费千分之一** | 每次仓位变化按 |delta| × 0.001 扣收，双边对称 |",
        "| **信息隔离** | 预测日 t 的特征只包含昨日 close/high/low/volume 和今日 open，"
        "今日 close/high/low 完全不可见 |",
        "",
        "> **风险提示**: 回测结果不代表未来收益，A 股受政策影响显著，本模型仅供研究参考。",
        "",
    ]
    return "\n".join(lines)


def print_console_report(m: dict, label: str) -> None:
    """控制台简洁版绩效报告。"""
    sep = "-" * 52
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    rows = [
        ("总收益率",   f"{m['total_ret']*100:+.2f}%"),
        ("年化收益率", f"{m['ann_ret']*100:+.2f}%"),
        ("夏普比率",   f"{m['sharpe']:.3f}"),
        ("最大回撤",   f"{m['max_dd']*100:.2f}%"),
        ("卡玛比率",   f"{m['calmar']:.3f}"),
        ("日胜率",     f"{m['win_rate']*100:.1f}%"),
        ("平均仓位",   f"{m['avg_pos']*100:.1f}%"),
        ("总手续费",   f"{m['total_fees']*100:.3f}%"),
        ("买持基准",   f"{m['bh_ret']*100:+.2f}%"),
    ]
    for k, v in rows:
        print(f"  {k:<10}: {v}")
    print(sep)


# ══════════════════════════════════════════════════════
#  图表输出
# ══════════════════════════════════════════════════════

def plot_results(m: dict, dates: pd.Series, symbol: str,
                 retrain_marks: np.ndarray | None = None,
                 split_idx: int | None = None) -> str:
    """
    生成三联图：净值曲线 / 仓位序列 / 日 P&L。

    返回图表保存路径。
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    n     = len(m["equity"])
    xs    = range(n)
    dts   = dates.values[:n] if len(dates) >= n else list(dates.values) + [None] * (n - len(dates))

    # ── ① 净值曲线 ──
    ax = axes[0]
    ax.plot(xs, m["equity"], color="#1565C0", lw=1.4, label="策略净值")
    # 买持基准曲线（近似：累乘每日收益）
    bh_eq = np.cumprod(1 + m["pnl"] / (m["positions"] + 1e-8) * m["positions"])
    ax.set_ylabel("净值", fontsize=10)
    ax.set_title(f"{symbol}  LSTM 交易模型 — 净值曲线", fontsize=13, fontweight="bold")
    ax.axhline(1.0, color="#ccc", lw=0.8, ls=":")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # 标注划分线
    if split_idx is not None:
        ax.axvline(split_idx, color="#FF7043", lw=1.5, ls="--", label="训练/测试分界")
        ax.legend(fontsize=9)

    # 标注 walk-forward 重训时刻
    if retrain_marks is not None:
        for mk in retrain_marks:
            if mk < n:
                ax.axvline(mk, color="#78909C", lw=0.6, ls=":", alpha=0.6)

    # ── ② 仓位序列 ──
    ax = axes[1]
    ax.fill_between(xs, 0, m["positions"], alpha=0.4, color="#1976D2", label="仓位")
    ax.plot(xs, m["positions"], color="#1976D2", lw=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="#888", lw=0.6, ls="--", label="0.5 基准线")
    ax.set_ylabel("仓位 [0,1]", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    if split_idx is not None:
        ax.axvline(split_idx, color="#FF7043", lw=1.5, ls="--")
    if retrain_marks is not None:
        for mk in retrain_marks:
            if mk < n:
                ax.axvline(mk, color="#78909C", lw=0.6, ls=":", alpha=0.6)

    # ── ③ 日 P&L 柱状图 ──
    ax = axes[2]
    colors = ["#27ae60" if p >= 0 else "#c0392b" for p in m["pnl"]]
    ax.bar(xs, m["pnl"] * 100, color=colors, width=1.0, alpha=0.8)
    ax.set_ylabel("日 P&L (%)", fontsize=10)
    ax.set_xlabel("交易日（序号）", fontsize=10)
    ax.grid(True, alpha=0.2)

    if split_idx is not None:
        ax.axvline(split_idx, color="#FF7043", lw=1.5, ls="--")

    plt.tight_layout()
    out = CHARTS_DIR / f"{symbol}_ml_v2_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out)


# ══════════════════════════════════════════════════════
#  预测接口（对外 API，供外部调用）
# ══════════════════════════════════════════════════════

def make_predictor(model: TradingLSTM, hp: dict, device: torch.device):
    """
    返回 predict(df) 单次推理函数。

    使用方式：
      predictor = make_predictor(model, hp, device)
      position  = predictor(df)   # df 最后一行 open = 今日开盘价

    输入 df：含 OHLCV 列的 DataFrame
      - 最后一行 date = 今日，open = 今日开盘价（已知）
      - 最后一行 close/high/low = NaN 或任意（模型不使用）
    输出：今日建议仓位 float ∈ [0, max_pos]
    """
    @torch.no_grad()
    def predict(df: pd.DataFrame) -> float:
        # 计算技术指标（若未提前算好）
        df_w = compute_all(df) if "ma5" not in df.columns else df.copy()
        # 保留有完整收盘价的历史行（最后一行今日 close 可能是 NaN）
        df_hist = df_w.iloc[:-1].dropna(subset=["close"]).reset_index(drop=True)
        # 拼上今日行（只保留 open，其余 NaN）
        today_row = df_w.iloc[[-1]].copy()
        df_full = pd.concat([df_hist, today_row], ignore_index=True)

        feats = build_features(df_full)
        seq   = hp["seq_len"]
        if len(feats) < seq:
            return 0.0   # 历史数据不足，返回空仓

        x   = torch.from_numpy(feats[-seq:]).unsqueeze(0).to(device)
        pos = model(x)[0, -1].item() * hp["max_pos"]
        return float(np.clip(pos, 0.0, hp["max_pos"]))

    return predict


# ══════════════════════════════════════════════════════
#  主流程入口
# ══════════════════════════════════════════════════════

def run(symbol: str, start: str, end: str, hp: dict) -> dict:
    """
    完整运行：数据 → 特征 → 训练 → 推理 → 报告。

    返回包含 model / predict / performance 等的结果字典。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    today  = date.today().strftime("%Y%m%d")
    # 一直拉到今天的数据
    end_  = today if end > today else end

    print(f"\n{'='*55}")
    print(f"  ML 交易模型 v2 — {symbol}  {start}~{end_}")
    print(f"  模式: {'Walk-forward' if hp['walk_forward'] else '简单 Train/Test'}")
    print(f"  设备: {device}")
    print(f"{'='*55}")

    # ── 数据获取与指标计算 ──────────────────────────
    df = fetch_daily(symbol, start, end_)
    df = compute_all(df).dropna().reset_index(drop=True)
    print(f"\n  有效数据: {len(df)} 天  "
          f"({df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()})")

    features = build_features(df)
    rets     = build_returns(df)
    n        = len(features)
    print(f"  特征维度: {features.shape[1]}  |  样本数: {n}")

    # ── Walk-forward 或简单模式 ─────────────────────
    if hp["walk_forward"]:
        positions, retrain_marks = run_walk_forward(features, rets, hp, device)
        split_idx   = hp["warm_up_days"]   # 显示训练/预测分界
        # 整体绩效（从 warm_up 之后开始计算，之前均为空仓）
        wf_pos = positions[split_idx:]
        wf_ret = rets[split_idx:]
        test_m = calc_performance(wf_pos, wf_ret, hp["fee_rate"])
        # 虚拟训练集（warm_up 之内）绩效（固定空仓 = 0）
        warm_pos = np.zeros(split_idx)
        train_m  = calc_performance(warm_pos, rets[:split_idx], hp["fee_rate"])
        all_m    = calc_performance(positions, rets, hp["fee_rate"])
        mode     = "Walk-forward"
    else:
        positions, marks = run_simple(features, rets, hp, device)
        split_idx  = int(n * hp["train_ratio"])
        retrain_marks = marks
        train_m  = calc_performance(positions[:split_idx], rets[:split_idx], hp["fee_rate"])
        test_m   = calc_performance(positions[split_idx:], rets[split_idx:], hp["fee_rate"])
        all_m    = calc_performance(positions, rets, hp["fee_rate"])
        mode     = f"Train({int(hp['train_ratio']*100)}%) / Test"

    # ── 控制台报告 ──────────────────────────────────
    print_console_report(train_m, "训练阶段")
    print_console_report(test_m,  "测试阶段（Walk-forward 样本外）")
    print_console_report(all_m,   "整体绩效（全周期）")

    # ── Markdown 报告 ───────────────────────────────
    md = build_md_report(all_m, symbol, start, end_, hp, mode, train_m, test_m)
    md_path = CHARTS_DIR / f"{symbol}_ml_v2_report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\n  [Markdown 报告] -> {md_path}")

    # ── 图表 ────────────────────────────────────────
    chart = plot_results(all_m, df["date"], symbol,
                         retrain_marks=retrain_marks,
                         split_idx=split_idx)
    print(f"  [图表]          -> {chart}")

    # ── 重训最终模型（用全量数据） ───────────────────
    print("\n  重训最终模型（使用全量历史数据）...")
    final_model = _train_once(features, rets, hp, device, verbose=False)

    # ── 最新仓位预测 ─────────────────────────────────
    predictor = make_predictor(final_model, hp, device)
    latest_pos = predictor(df)
    last_date  = df["date"].iloc[-1].date()
    signal     = "买入/加仓" if latest_pos > 0.6 else ("减仓/卖出" if latest_pos < 0.3 else "持仓观望")

    print(f"\n{'='*55}")
    print(f"  最新信号（基于 {last_date} 数据）")
    print(f"  建议目标仓位 = {latest_pos:.4f}  [{signal}]")
    print(f"  (0=空仓，0.5=半仓，1=满仓)")
    print(f"{'='*55}\n")

    return {
        "model":    final_model,
        "device":   device,
        "hp":       hp,
        "df":       df,
        "features": features,
        "returns":  rets,
        "positions": positions,
        "all":      all_m,
        "train":    train_m,
        "test":     test_m,
        "predict":  predictor,
        "retrain_marks": retrain_marks,
    }


# ══════════════════════════════════════════════════════
#  CLI 入口
# ══════════════════════════════════════════════════════

def main() -> None:
    today = date.today().strftime("%Y%m%d")

    p = argparse.ArgumentParser(
        description="LSTM 端到端交易模型 v2（长仓+Walk-forward）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── 必填参数 ──
    p.add_argument("--symbol", required=True, help="股票代码（如 002837）")
    p.add_argument("--start",  required=True, help="开始日期 YYYYMMDD")
    p.add_argument("--end",    default=today,  help=f"结束日期 YYYYMMDD（默认今天 {today}）")

    # ── Walk-forward 控制 ──
    p.add_argument("--walk-forward",    dest="walk_forward", action="store_true",
                   default=DEFAULT_HP["walk_forward"], help="启用 Walk-forward（默认开启）")
    p.add_argument("--no-walk-forward", dest="walk_forward", action="store_false",
                   help="关闭 Walk-forward，使用简单 Train/Test 划分")
    p.add_argument("--warm-up",    type=int,   default=DEFAULT_HP["warm_up_days"],
                   metavar="N",   help=f"预热天数，默认 {DEFAULT_HP['warm_up_days']}")
    p.add_argument("--retrain-freq", type=int, default=DEFAULT_HP["retrain_freq"],
                   metavar="N",   help=f"重训间隔（交易日），默认 {DEFAULT_HP['retrain_freq']}")
    p.add_argument("--train-ratio", type=float, default=DEFAULT_HP["train_ratio"],
                   metavar="F",   help=f"简单模式训练比例，默认 {DEFAULT_HP['train_ratio']}")

    # ── 模型超参数 ──
    hp_g = p.add_argument_group("模型超参数（均有默认值，可调）")
    hp_g.add_argument("--seq-len",  type=int,   default=DEFAULT_HP["seq_len"],
                      metavar="N", help=f"回看窗口，默认 {DEFAULT_HP['seq_len']}")
    hp_g.add_argument("--hidden",   type=int,   default=DEFAULT_HP["hidden_dim"],
                      metavar="N", help=f"LSTM 隐层维度，默认 {DEFAULT_HP['hidden_dim']}")
    hp_g.add_argument("--layers",   type=int,   default=DEFAULT_HP["num_layers"],
                      metavar="N", help=f"LSTM 层数，默认 {DEFAULT_HP['num_layers']}")
    hp_g.add_argument("--dropout",  type=float, default=DEFAULT_HP["dropout"],
                      metavar="F", help=f"Dropout，默认 {DEFAULT_HP['dropout']}")
    hp_g.add_argument("--lr",       type=float, default=DEFAULT_HP["lr"],
                      metavar="F", help=f"学习率，默认 {DEFAULT_HP['lr']}")
    hp_g.add_argument("--epochs",   type=int,   default=DEFAULT_HP["epochs"],
                      metavar="N", help=f"训练轮数，默认 {DEFAULT_HP['epochs']}")
    hp_g.add_argument("--batch",    type=int,   default=DEFAULT_HP["batch_size"],
                      metavar="N", help=f"批大小，默认 {DEFAULT_HP['batch_size']}")
    hp_g.add_argument("--fee",      type=float, default=DEFAULT_HP["fee_rate"],
                      metavar="F", help=f"手续费率，默认 {DEFAULT_HP['fee_rate']}")
    hp_g.add_argument("--max-pos",  type=float, default=DEFAULT_HP["max_pos"],
                      metavar="F", help=f"最大仓位，默认 {DEFAULT_HP['max_pos']}")
    hp_g.add_argument("--fee-lambda", type=float, default=DEFAULT_HP["fee_lambda"],
                      metavar="F", help=f"手续费惩罚倍率，默认 {DEFAULT_HP['fee_lambda']}")

    args = p.parse_args()

    hp = {
        "seq_len":      args.seq_len,
        "hidden_dim":   args.hidden,
        "num_layers":   args.layers,
        "dropout":      args.dropout,
        "lr":           args.lr,
        "epochs":       args.epochs,
        "batch_size":   args.batch,
        "train_ratio":  args.train_ratio,
        "fee_rate":     args.fee,
        "max_pos":      args.max_pos,
        "walk_forward": args.walk_forward,
        "warm_up_days": args.warm_up,
        "retrain_freq": args.retrain_freq,
        "weight_decay": DEFAULT_HP["weight_decay"],
        "grad_clip":    DEFAULT_HP["grad_clip"],
        "fee_lambda":   args.fee_lambda,
    }

    run(args.symbol, args.start, args.end, hp)


if __name__ == "__main__":
    main()
