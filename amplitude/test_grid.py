#!/usr/bin/env python3
"""
网格搜索详细报告功能测试脚本

测试覆盖范围:
  1. _run_one 函数返回的字段完整性（新增字段验证）
  2. _print_full_grid_table 控制台输出内容验证
  3. _build_comprehensive_txt 综合报告内容验证（十一个章节）
  4. grid_search → _analyze_and_recommend 参数传递正确性
  5. 边界条件测试（空数据、单组合、全不达标等）

运行方式:
  python amplitude/test_grid.py
"""

import io
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

# ── 添加项目根目录到路径 ──
sys.path.insert(0, str(Path(__file__).parent.parent))

from amplitude.grid import (
    _run_one,
    _print_full_grid_table,
    _build_comprehensive_md,
    _analyze_and_recommend,
)
from amplitude.backtest import run_backtest, calc_metrics


# ══════════════════════════════════════════════════════════
# 辅助函数：生成模拟行情数据
# ══════════════════════════════════════════════════════════

def _make_fake_df(n_days: int = 100, base_price: float = 10.0,
                  seed: int = 42) -> pd.DataFrame:
    """
    生成一份模拟日线数据，包含 date/open/high/low/close/volume 列。

    参数:
      n_days     : 交易日天数
      base_price : 基础价格
      seed       : 随机种子（保证可重现）

    返回:
      pd.DataFrame，列结构与真实日线数据一致
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range('2020-01-01', periods=n_days, freq='B')
    closes = [base_price]

    for _ in range(n_days - 1):
        change = rng.uniform(-0.03, 0.03)
        closes.append(closes[-1] * (1 + change))

    closes = np.array(closes)
    highs  = closes * rng.uniform(1.00, 1.03, n_days)
    lows   = closes * rng.uniform(0.97, 1.00, n_days)
    opens  = closes * rng.uniform(0.98, 1.02, n_days)

    return pd.DataFrame({
        'date':   dates[:n_days],
        'open':   np.round(opens,  2),
        'high':   np.round(highs,  2),
        'low':    np.round(lows,   2),
        'close':  np.round(closes, 2),
        'volume': rng.randint(1000, 10000, n_days),
    })


def _make_grid_results(df: pd.DataFrame,
                       buy_vals: list[float] = None,
                       sell_vals: list[float] = None,
                       cash: float = 100_000.0) -> pd.DataFrame:
    """
    对给定的 buy/sell 参数组合运行 _run_one，返回网格结果 DataFrame。
    """
    if buy_vals is None:
        buy_vals = [95.0, 96.0, 97.0]
    if sell_vals is None:
        sell_vals = [103.0, 104.0, 105.0]

    rows = []
    for b in buy_vals:
        for s in sell_vals:
            if b < s:
                rows.append(_run_one(df, b, s, cash))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 测试类
# ══════════════════════════════════════════════════════════

class TestRunOne(unittest.TestCase):
    """测试 _run_one 函数返回的字段完整性和数据类型"""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_fake_df(200)

    def test_return_all_expected_fields(self):
        """_run_one 必须返回所有预期字段"""
        result = _run_one(self.df, 95.0, 105.0, 100_000.0)

        expected_fields = [
            'buy_pct', 'sell_pct',
            'ann_ret', 'total_ret', 'max_dd',
            'sharpe', 'calmar',
            'win_rate', 'avg_profit',
            'n_trades', 'ann_trades',
            'years',
            'bh_ret', 'bh_ann',
            'profit_factor',
        ]
        for field in expected_fields:
            self.assertIn(field, result, f"缺少字段: {field}")

    def test_return_types(self):
        """_run_one 返回值的数据类型检查"""
        result = _run_one(self.df, 95.0, 105.0, 100_000.0)

        for key, val in result.items():
            self.assertIsInstance(
                val, (int, float),
                f"字段 {key} 类型应为数值, 实际为 {type(val)}"
            )

    def test_buy_sell_pct_preserved(self):
        """_run_one 返回的 buy_pct/sell_pct 应与输入一致"""
        result = _run_one(self.df, 92.5, 107.5, 100_000.0)
        self.assertEqual(result['buy_pct'], 92.5)
        self.assertEqual(result['sell_pct'], 107.5)

    def test_years_positive(self):
        """回测年限应大于 0"""
        result = _run_one(self.df, 95.0, 105.0, 100_000.0)
        self.assertGreater(result['years'], 0)

    def test_profit_factor_non_negative(self):
        """盈亏比不应为负数"""
        result = _run_one(self.df, 95.0, 105.0, 100_000.0)
        self.assertGreaterEqual(result['profit_factor'], 0)


class TestPrintFullGridTable(unittest.TestCase):
    """测试控制台全量表格输出"""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_fake_df(300)
        cls.results = _make_grid_results(cls.df)
        cls.valid = cls.results[cls.results['ann_trades'] >= 1].copy()
        if cls.valid.empty:
            cls.valid = cls.results.copy()
        cls.bp = cls.valid.loc[cls.valid['ann_ret'].idxmax()]
        cls.bs = cls.valid.loc[cls.valid['calmar'].idxmax()]

    def _capture_output(self) -> str:
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            _print_full_grid_table(
                self.results, self.valid, 1.0, self.bp, self.bs)
        return buf.getvalue()

    def test_output_contains_statistics(self):
        output = self._capture_output()
        self.assertIn('统计概览', output)
        self.assertIn('平均', output)
        self.assertIn('中位数', output)

    def test_output_contains_full_table(self):
        output = self._capture_output()
        self.assertIn('全量结果', output)

    def test_output_contains_valid_table(self):
        output = self._capture_output()
        self.assertIn('有效参数', output)

    def test_output_contains_recommendation(self):
        output = self._capture_output()
        self.assertIn('收益最高', output)
        self.assertIn('最稳妥', output)

    def test_output_contains_win_rate_column(self):
        output = self._capture_output()
        self.assertIn('胜率%', output)

    def test_output_contains_profit_factor_column(self):
        output = self._capture_output()
        self.assertIn('盈亏比', output)

    def test_output_contains_profitable_stats(self):
        output = self._capture_output()
        self.assertIn('盈利组合', output)

    def test_output_contains_benchmark_comparison(self):
        output = self._capture_output()
        self.assertIn('跑赢买持', output)


class TestBuildComprehensiveMd(unittest.TestCase):
    """测试综合报告 Markdown 内容——验证十一个章节是否都存在"""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_fake_df(400)
        cls.results = _make_grid_results(
            cls.df,
            buy_vals=[94.0, 95.0, 96.0, 97.0],
            sell_vals=[103.0, 104.0, 105.0, 106.0],
        )
        cls.valid = cls.results[cls.results['ann_trades'] >= 0.5].copy()
        if cls.valid.empty:
            cls.valid = cls.results.copy()
        cls.bp = cls.valid.loc[cls.valid['ann_ret'].idxmax()]
        cls.bs = cls.valid.loc[cls.valid['calmar'].idxmax()]

        cls.report = _build_comprehensive_md(
            results=cls.results,
            valid=cls.valid,
            min_trades=0.5,
            bp=cls.bp,
            bs=cls.bs,
            symbol='TEST',
            start='20200101',
            end='20251231',
            cash=100_000.0,
            buy_range=(94.0, 97.0),
            sell_range=(103.0, 106.0),
            step=1.0,
        )

    def test_section_zero_global_overview(self):
        self.assertIn('【零】全局概览统计', self.report)

    def test_section_one_full_results(self):
        self.assertIn('【一】全量扫描结果', self.report)

    def test_section_two_valid_params(self):
        self.assertIn('【二】有效参数详细', self.report)

    def test_section_three_matrices(self):
        self.assertIn('【三】参数敏感性矩阵 — 年化收益率', self.report)
        self.assertIn('【三】参数敏感性矩阵 — 卡玛比率', self.report)
        self.assertIn('【三】参数敏感性矩阵 — 最大回撤', self.report)
        self.assertIn('【三】参数敏感性矩阵 — 夏普比率', self.report)
        self.assertIn('【三】参数敏感性矩阵 — 胜率', self.report)
        self.assertIn('【三】参数敏感性矩阵 — 年均交易数', self.report)

    def test_section_four_rankings(self):
        self.assertIn('【四】', self.report)

    def test_section_five_stability(self):
        self.assertIn('【五】稳定性分析', self.report)

    def test_section_six_marginal(self):
        self.assertIn('【六】参数边际效应分析', self.report)

    def test_section_seven_benchmark(self):
        self.assertIn('【七】跑赢买持基准分析', self.report)

    def test_section_eight_quadrant(self):
        self.assertIn('【八】风险收益象限分析', self.report)

    def test_section_nine_scoring(self):
        self.assertIn('【九】参数组合综合评分排行', self.report)

    def test_section_ten_recommendation(self):
        self.assertIn('【十】最终推荐', self.report)

    def test_header_contains_scan_range(self):
        self.assertIn('buy 94.0%~97.0%', self.report)
        self.assertIn('sell 103.0%~106.0%', self.report)

    def test_header_contains_date_range(self):
        self.assertIn('2020-01-01', self.report)
        self.assertIn('2025-12-31', self.report)

    def test_overview_contains_statistics_table(self):
        self.assertIn('均值', self.report)
        self.assertIn('中位数', self.report)
        self.assertIn('标准差', self.report)
        self.assertIn('最小值', self.report)
        self.assertIn('最大值', self.report)

    def test_overview_contains_metric_labels(self):
        for label in ['年化收益率', '总收益率', '最大回撤', '夏普比率',
                       '卡玛比率', '胜率', '年均交易数', '盈亏比']:
            self.assertIn(label, self.report)

    def test_overview_contains_profitable_count(self):
        self.assertIn('盈利组合', self.report)
        self.assertIn('跑赢买持', self.report)

    def test_matrices_contain_asterisk(self):
        if (self.results['ann_trades'] >= 0.5).any():
            self.assertIn('*', self.report)

    def test_marginal_analysis_buy_pct(self):
        self.assertIn('buy_pct 边际效应', self.report)

    def test_marginal_analysis_sell_pct(self):
        self.assertIn('sell_pct 边际效应', self.report)

    def test_quadrant_analysis_has_q1(self):
        self.assertIn('Q1', self.report)

    def test_scoring_has_weights(self):
        self.assertIn('评分权重', self.report)
        self.assertIn('30%', self.report)

    def test_recommendation_contains_details(self):
        self.assertIn('年化', self.report)
        self.assertIn('最大回撤', self.report)
        self.assertIn('胜率', self.report)
        self.assertIn('单笔盈亏', self.report)
        self.assertIn('盈亏比', self.report)
        self.assertIn('年均交易', self.report)

    def test_benchmark_excess_return(self):
        self.assertIn('超额指标', self.report)
        self.assertIn('平均超额', self.report)

    def test_risk_disclaimer(self):
        self.assertIn('附录：核心字段与术语解释', self.report)

    def test_report_length_substantial(self):
        lines = self.report.strip().split('\n')
        self.assertGreater(len(lines), 80,
                           f"报告只有 {len(lines)} 行，太短了")


class TestEdgeCases(unittest.TestCase):
    """边界条件测试"""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_fake_df(100)

    def test_single_combination(self):
        results = _make_grid_results(
            self.df, buy_vals=[95.0], sell_vals=[105.0])
        valid = results.copy()
        bp = valid.iloc[0]
        bs = valid.iloc[0]

        report = _build_comprehensive_md(
            results, valid, 0, bp, bs,
            'TEST', '20200101', '20201231', 100_000.0,
            (95.0, 95.0), (105.0, 105.0), 1.0)
        self.assertIn('【十】最终推荐', report)

    def test_all_below_min_trades(self):
        results = _make_grid_results(self.df)
        valid = results[results['ann_trades'] >= 99999].copy()
        if valid.empty:
            valid = results.copy()
        bp = valid.loc[valid['ann_ret'].idxmax()]
        bs = valid.loc[valid['calmar'].idxmax()]

        report = _build_comprehensive_md(
            results, valid, 99999, bp, bs,
            'TEST', '20200101', '20201231', 100_000.0,
            (95.0, 97.0), (103.0, 105.0), 1.0)
        self.assertIn('【零】全局概览统计', report)

    def test_grid_csv_columns(self):
        results = _make_grid_results(self.df)
        required_cols = [
            'buy_pct', 'sell_pct', 'ann_ret', 'total_ret', 'max_dd',
            'sharpe', 'calmar', 'win_rate', 'avg_profit', 'n_trades',
            'ann_trades', 'years', 'bh_ret', 'bh_ann', 'profit_factor',
        ]
        for col in required_cols:
            self.assertIn(col, results.columns, f"CSV 缺少列: {col}")


class TestAnalyzeAndRecommendSignature(unittest.TestCase):
    """测试 _analyze_and_recommend 函数签名兼容性"""

    def test_accepts_all_required_params(self):
        import inspect
        sig = inspect.signature(_analyze_and_recommend)
        param_names = list(sig.parameters.keys())
        self.assertIn('buy_range', param_names)
        self.assertIn('sell_range', param_names)
        self.assertIn('step', param_names)


# ══════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    unittest.main(verbosity=2)
