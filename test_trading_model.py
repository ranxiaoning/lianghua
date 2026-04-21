#!/usr/bin/env python3
"""
ML 交易模型 v2 测试脚本

测试覆盖:
  1. 特征矩阵形状与 NaN 检查
  2. 仓位范围约束 [0, 1]（长仓约束）
  3. 损失函数单调性验证
  4. 数据信息隔离（预测日不可见 close/high/low）
  5. Dataset 长度与索引正确性
  6. 绩效计算准确性（手续费、收益积累、夏普）
  7. Walk-forward 分段不泄露
  8. make_predictor 返回合法仓位
  9. 模型输出形状正确

运行方式:
  python test_trading_model.py
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from ml.trading_model import (
    DEFAULT_HP,
    build_features,
    build_returns,
    TradingDataset,
    TradingLSTM,
    sharpe_loss,
    calc_performance,
    _train_once,
    _predict_days,
    run_walk_forward,
    make_predictor,
)


# ══════════════════════════════════════════════════════
# 辅助：生成模拟行情数据（含技术指标）
# ══════════════════════════════════════════════════════

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """生成 n 天的模拟日线数据（含技术指标列）。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2021-01-01", periods=n, freq="B")
    closes = [10.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.uniform(-0.03, 0.03)))
    closes = np.array(closes, dtype=np.float32)
    highs  = closes * rng.uniform(1.00, 1.03, n)
    lows   = closes * rng.uniform(0.97, 1.00, n)
    opens  = closes * rng.uniform(0.98, 1.02, n)

    df = pd.DataFrame({
        "date":   dates,
        "open":   np.round(opens, 2),
        "high":   np.round(highs, 2),
        "low":    np.round(lows,  2),
        "close":  np.round(closes, 2),
        "volume": rng.randint(1_000, 10_000, n),
    })

    # 模拟技术指标列（简化版，测试只需存在）
    df["ma5"]     = df["close"].rolling(5,  min_periods=5).mean()
    df["ma20"]    = df["close"].rolling(20, min_periods=20).mean()
    df["ema12"]   = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"]   = df["close"].ewm(span=26, adjust=False).mean()
    ema12 = df["ema12"]; ema26 = df["ema26"]
    df["diff"]    = ema12 - ema26
    df["dea"]     = df["diff"].ewm(span=9, adjust=False).mean()
    df["macd"]    = 2.0 * (df["diff"] - df["dea"])
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    df["rsi14"]   = 100 - 100 / (1 + gain / (loss + 1e-8))
    mid_ = df["ma20"]; std_ = df["close"].rolling(20, min_periods=20).std(ddof=0)
    df["bb_upper"] = mid_ + 2 * std_
    df["bb_mid"]   = mid_
    df["bb_lower"] = mid_ - 2 * std_
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (mid_ + 1e-8)

    return df.dropna().reset_index(drop=True)


# ══════════════════════════════════════════════════════
class TestFeatureEngineering(unittest.TestCase):
    """特征工程正确性测试"""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_ohlcv(200)
        cls.feat = build_features(cls.df)
        cls.rets = build_returns(cls.df)

    def test_feature_shape(self):
        """特征矩阵行数应为 len(df)-1"""
        self.assertEqual(self.feat.shape[0], len(self.df) - 1,
                         "特征行数应比 df 少1（首行样本需要前一天信息）")

    def test_feature_dim(self):
        """特征维度应 >= 22（5基础+4动量+12指标+至少1开盘缺口）"""
        self.assertGreaterEqual(self.feat.shape[1], 22,
                                f"特征维度只有 {self.feat.shape[1]}，可能有指标缺失")

    def test_no_nan_in_features(self):
        """特征矩阵中不应含 NaN"""
        self.assertFalse(np.any(np.isnan(self.feat)),
                         "特征矩阵包含 NaN，请检查归一化逻辑")

    def test_no_inf_in_features(self):
        """特征矩阵中不应含 Inf"""
        self.assertFalse(np.any(np.isinf(self.feat)),
                         "特征矩阵包含 Inf")

    def test_feature_clipped(self):
        """特征值应裁剪在 [-5, 5]"""
        self.assertTrue(np.all(self.feat >= -5.0) and np.all(self.feat <= 5.0),
                        "特征值超出 [-5,5] 裁剪范围")

    def test_returns_length(self):
        """收益率长度与特征严格对齐"""
        self.assertEqual(len(self.rets), len(self.feat))

    def test_returns_reasonable(self):
        """A股日收益不应超过 ±50%"""
        self.assertTrue(np.all(self.rets > -0.5) and np.all(self.rets < 0.5),
                        f"收益率超出合理范围: [{self.rets.min():.3f}, {self.rets.max():.3f}]")

    def test_information_leakage_open_only(self):
        """
        信息隔离测试：特征向量第 0 维是今日开盘缺口（open/prev_close - 1）。
        直接测试第一个时刻，验证计算正确。
        """
        # 第一个特征行对应 t=1
        df = self.df
        c_prev = float(df["close"].iloc[0])
        expected_gap = float(df["open"].iloc[1] / c_prev - 1)
        actual_gap   = float(self.feat[0, 0])
        self.assertAlmostEqual(actual_gap, expected_gap, places=5,
                               msg="特征第0维应是今日开盘缺口 open/prev_close-1")


# ══════════════════════════════════════════════════════
class TestDataset(unittest.TestCase):
    """TradingDataset 测试"""

    def setUp(self):
        df = _make_ohlcv(100)
        self.feat = build_features(df)
        self.rets = build_returns(df)

    def test_dataset_length(self):
        """Dataset 长度 = N - seq_len + 1"""
        for seq in [10, 20, 30]:
            ds = TradingDataset(self.feat, self.rets, seq)
            expected = max(0, len(self.feat) - seq + 1)
            self.assertEqual(len(ds), expected,
                             f"seq_len={seq} 时 Dataset 长度不对")

    def test_item_shapes(self):
        """每个样本的 x/r 形状正确"""
        seq = 20
        ds  = TradingDataset(self.feat, self.rets, seq)
        x, r = ds[0]
        self.assertEqual(x.shape, (seq, self.feat.shape[1]))
        self.assertEqual(r.shape, (seq,))

    def test_item_dtype(self):
        """样本为 float32 张量"""
        ds = TradingDataset(self.feat, self.rets, 10)
        x, r = ds[0]
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(r.dtype, torch.float32)


# ══════════════════════════════════════════════════════
class TestModel(unittest.TestCase):
    """TradingLSTM 输出测试"""

    def setUp(self):
        df = _make_ohlcv(100)
        self.feat_dim = build_features(df).shape[1]
        self.hp = {**DEFAULT_HP, "seq_len": 10, "hidden_dim": 32, "num_layers": 1}
        self.device = torch.device("cpu")

    def test_output_in_zero_one(self):
        """模型输出必须在 [0, 1]（Sigmoid 保证）"""
        model = TradingLSTM(self.feat_dim, 32, 1, 0.0)
        B, T  = 4, self.hp["seq_len"]
        x     = torch.randn(B, T, self.feat_dim)
        with torch.no_grad():
            out = model(x)   # (B, T)
        self.assertEqual(out.shape, (B, T))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0),
                        f"模型输出 [{out.min():.3f}, {out.max():.3f}] 超出 [0,1]")

    def test_output_shape_seq(self):
        """模型对每个时刻均输出仓位（序列级推理）"""
        model = TradingLSTM(self.feat_dim, 32, 2, 0.1)
        x     = torch.randn(2, 15, self.feat_dim)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 15),
                         "模型应输出 (B, seq_len) 的仓位序列")


# ══════════════════════════════════════════════════════
class TestLossFunction(unittest.TestCase):
    """损失函数单调性与特性测试"""

    def test_loss_decreasing_with_profit(self):
        """
        仿真场景：完美策略（市场涨时持满仓）损失 < 随机策略损失。
        """
        T = 50
        # 市场每天上涨
        returns  = torch.ones(1, T) * 0.01

        # 完美策略：持满仓
        perfect  = torch.ones(1, T)
        # 随机策略：仓位随机
        random_p = torch.rand(1, T)

        loss_perfect = sharpe_loss(perfect, returns, 0.001, 1.0).item()
        loss_random  = sharpe_loss(random_p, returns, 0.001, 1.0).item()
        self.assertLess(loss_perfect, loss_random,
                        "在单边上涨市场中，满仓策略的 Sharpe 损失应更小")

    def test_fee_reduces_profit(self):
        """手续费越高，损失应越大（收益越低）"""
        T = 30
        # 上涨市，满仓，每天换手（制造手续费）
        pos  = torch.rand(1, T)   # 频繁变动
        rets = torch.ones(1, T) * 0.005

        loss_low_fee  = sharpe_loss(pos, rets, 0.0001, 1.0).item()
        loss_high_fee = sharpe_loss(pos, rets, 0.005,  1.0).item()
        self.assertLess(loss_low_fee, loss_high_fee,
                        "手续费越高，利润越低，损失应越大")

    def test_zero_pos_neutral_loss(self):
        """全空仓（pos=0）时：无收益、无手续费、Sharpe=0"""
        T    = 50
        pos  = torch.zeros(1, T)
        rets = torch.ones(1, T) * 0.01
        loss = sharpe_loss(pos, rets, 0.001, 1.0).item()
        # 全0时 pnl 全为0, loss = -0/稳定项 ≈ 0
        self.assertAlmostEqual(loss, 0.0, delta=1e-4,
                               msg="全空仓时损失应约为0")


# ══════════════════════════════════════════════════════
class TestPerformanceCalc(unittest.TestCase):
    """绩效计算正确性测试"""

    def test_zero_position_zero_fees(self):
        """全空仓：收益=0，手续费=0"""
        pos = np.zeros(100)
        ret = np.random.uniform(-0.02, 0.02, 100).astype(np.float32)
        m   = calc_performance(pos, ret, 0.001)
        self.assertAlmostEqual(m["total_ret"],   0.0, places=6)
        self.assertAlmostEqual(m["total_fees"],  0.0, places=6)

    def test_full_pos_no_change_zero_fees(self):
        """满仓且不换手：手续费仅开仓一次（delta[0]=1）"""
        pos = np.ones(100, dtype=np.float32)
        ret = np.zeros(100, dtype=np.float32)
        m   = calc_performance(pos, ret, 0.001)
        # 只有第一天 delta=1，其余 delta=0
        expected_fee = 0.001
        self.assertAlmostEqual(m["total_fees"], expected_fee, places=4)

    def test_equity_cumulative(self):
        """等值收益：净值应以复利方式积累"""
        pos = np.ones(10, dtype=np.float32)
        ret = np.full(10, 0.01, dtype=np.float32)   # 每天1%
        m   = calc_performance(pos, ret, 0.0)        # 无手续费
        expected = (1.01 ** 10) - 1
        self.assertAlmostEqual(m["total_ret"], expected, places=4)

    def test_position_clamp(self):
        """模拟仓位在 [0,1] 内，绩效计算应正常"""
        rng = np.random.RandomState(0)
        pos = rng.uniform(0, 1, 200).astype(np.float32)
        ret = rng.uniform(-0.03, 0.03, 200).astype(np.float32)
        m   = calc_performance(pos, ret, 0.001)
        self.assertIn("sharpe", m)
        self.assertIn("max_dd", m)
        self.assertLessEqual(m["max_dd"], 0.0, "最大回撤应为非正数")

    def test_bh_return_correct(self):
        """买持基准 = 全程持仓（pos=1）无手续费"""
        ret = np.random.uniform(-0.02, 0.02, 50).astype(np.float32)
        m   = calc_performance(np.ones(50), ret, 0.0)
        expected_bh = float(np.prod(1 + ret) - 1)
        self.assertAlmostEqual(m["bh_ret"], expected_bh, places=5)


# ══════════════════════════════════════════════════════
class TestTrainAndPredict(unittest.TestCase):
    """训练与推理集成测试"""

    @classmethod
    def setUpClass(cls):
        df        = _make_ohlcv(200)
        cls.feat  = build_features(df)
        cls.rets  = build_returns(df)
        cls.device = torch.device("cpu")
        cls.hp    = {
            **DEFAULT_HP,
            "seq_len":    10,
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout":    0.0,
            "epochs":     3,     # 只训练3轮，测试速度
            "batch_size": 16,
        }

    def test_train_returns_model(self):
        """_train_once 应返回 TradingLSTM 实例"""
        model = _train_once(self.feat, self.rets, self.hp, self.device, verbose=False)
        self.assertIsInstance(model, TradingLSTM)

    def test_predict_length(self):
        """_predict_days 输出长度 = len(features)（含前 seq_len-1 个零填充）"""
        model = _train_once(self.feat, self.rets, self.hp, self.device, verbose=False)
        pos   = _predict_days(model, self.feat, self.hp, self.device)
        self.assertEqual(len(pos), len(self.feat),
                         "predict_days 输出长度应等于特征数组长度")

    def test_predict_in_range(self):
        """推理结果必须在 [0, max_pos]"""
        model = _train_once(self.feat, self.rets, self.hp, self.device, verbose=False)
        pos   = _predict_days(model, self.feat, self.hp, self.device)
        self.assertTrue(np.all(pos >= 0.0) and np.all(pos <= self.hp["max_pos"]),
                        f"仓位超出 [0,{self.hp['max_pos']}]: [{pos.min():.4f},{pos.max():.4f}]")

    def test_no_nan_in_predictions(self):
        """推理结果中不应有 NaN"""
        model = _train_once(self.feat, self.rets, self.hp, self.device, verbose=False)
        pos   = _predict_days(model, self.feat, self.hp, self.device)
        self.assertFalse(np.any(np.isnan(pos)), "仓位序列含 NaN")


# ══════════════════════════════════════════════════════
class TestWalkForward(unittest.TestCase):
    """Walk-forward 数据合法性测试"""

    @classmethod
    def setUpClass(cls):
        df       = _make_ohlcv(300)
        cls.feat = build_features(df)
        cls.rets = build_returns(df)
        cls.hp   = {
            **DEFAULT_HP,
            "seq_len":      10,
            "hidden_dim":   32,
            "num_layers":   1,
            "dropout":      0.0,
            "epochs":       2,
            "batch_size":   16,
            "warm_up_days": 100,
            "retrain_freq": 50,
        }
        cls.device = torch.device("cpu")
        cls.pos, cls.marks = run_walk_forward(
            cls.feat, cls.rets, cls.hp, cls.device
        )

    def test_output_length(self):
        """walk-forward 仓位序列长度 = 特征数组长度"""
        self.assertEqual(len(self.pos), len(self.feat))

    def test_warmup_is_zero(self):
        """预热期（前 warm_up_days 天）仓位应全为0（空仓）"""
        wu = self.hp["warm_up_days"]
        self.assertTrue(np.all(self.pos[:wu] == 0.0),
                        "预热期内模型未预测，应填充0（空仓）")

    def test_positions_in_range(self):
        """walk-forward 预测的仓位必须在 [0, max_pos]"""
        self.assertTrue(np.all(self.pos >= 0.0),
                        "出现负仓位，违反长仓约束")
        self.assertTrue(np.all(self.pos <= self.hp["max_pos"]),
                        f"仓位超过 max_pos={self.hp['max_pos']}")

    def test_retrain_marks_count(self):
        """重训次数 = ceil((N - warm_up) / retrain_freq)"""
        n       = len(self.feat)
        wu      = self.hp["warm_up_days"]
        freq    = self.hp["retrain_freq"]
        expected = len(range(wu, n, freq))
        self.assertEqual(len(self.marks), expected,
                         f"重训次数: 预期 {expected}，实际 {len(self.marks)}")

    def test_no_nan(self):
        """Walk-forward 仓位无 NaN"""
        self.assertFalse(np.any(np.isnan(self.pos)))


# ══════════════════════════════════════════════════════
class TestPredictorAPI(unittest.TestCase):
    """make_predictor（对外 API）测试"""

    @classmethod
    def setUpClass(cls):
        df        = _make_ohlcv(200)
        cls.df    = df
        feat      = build_features(df)
        rets      = build_returns(df)
        cls.hp    = {
            **DEFAULT_HP,
            "seq_len":    10,
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout":    0.0,
            "epochs":     2,
            "batch_size": 16,
        }
        cls.device = torch.device("cpu")
        cls.model  = _train_once(feat, rets, cls.hp, cls.device, verbose=False)

    def test_predictor_returns_float(self):
        """predict(df) 应返回 float"""
        predictor = make_predictor(self.model, self.hp, self.device)
        pos = predictor(self.df)
        self.assertIsInstance(pos, float)

    def test_predictor_in_range(self):
        """predict(df) 返回值必须在 [0, max_pos]"""
        predictor = make_predictor(self.model, self.hp, self.device)
        pos = predictor(self.df)
        self.assertGreaterEqual(pos, 0.0)
        self.assertLessEqual(pos, self.hp["max_pos"])

    def test_predictor_with_nan_close(self):
        """最后一行 close=NaN 时（模拟今日预测场景），predict 依然正常"""
        df_today = self.df.copy()
        df_today.loc[df_today.index[-1], "close"] = float("nan")
        predictor = make_predictor(self.model, self.hp, self.device)
        # 不应抛出异常
        try:
            pos = predictor(df_today)
            self.assertIsInstance(pos, float)
            self.assertGreaterEqual(pos, 0.0)
        except Exception as e:
            self.fail(f"当 close=NaN 时 predictor 抛出异常: {e}")


# ══════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
