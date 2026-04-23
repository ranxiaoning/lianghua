# ML 交易模型实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 `ml/trading_model.py` 中新增 `--loss pnl` 纯P&L损失选项，然后以逐日Walk-forward模式训练 000001 平安银行（20180101→今天）。

**Architecture:** 现有LSTM模型已完整实现Walk-forward训练、Sharpe损失、预测API。只需新增`pnl_loss`函数、将损失类型通过`hp["loss_type"]`字段传递、添加CLI参数，再运行训练。

**Tech Stack:** Python 3.10+, PyTorch, pandas, akshare, matplotlib

---

## 文件修改范围

| 文件 | 操作 | 修改内容 |
|------|------|---------|
| `ml/trading_model.py` | 修改 | 新增`pnl_loss`函数；`DEFAULT_HP`加`loss_type`；`_train_once`按`hp["loss_type"]`选损失；CLI加`--loss`参数 |
| `test_trading_model.py` | 修改 | 新增`TestPnlLoss`测试类 |

---

### Task 1：为 pnl_loss 添加测试（TDD先行）

**Files:**
- Modify: `test_trading_model.py`

- [ ] **Step 1：在 test_trading_model.py 末尾、`if __name__ == "__main__":` 之前，插入以下测试类**

```python
# ══════════════════════════════════════════════════════
class TestPnlLoss(unittest.TestCase):
    """pnl_loss 函数测试"""

    def test_uptrend_full_pos_lower_loss(self):
        """上涨市场中满仓策略的 pnl_loss 应低于随机策略"""
        from ml.trading_model import pnl_loss
        T = 50
        returns  = torch.ones(1, T) * 0.01
        perfect  = torch.ones(1, T)
        random_p = torch.rand(1, T)
        loss_perfect = pnl_loss(perfect, returns, 0.001, 1.0).item()
        loss_random  = pnl_loss(random_p, returns, 0.001, 1.0).item()
        self.assertLess(loss_perfect, loss_random,
                        "上涨市场满仓时 pnl_loss 应更小（盈利更多）")

    def test_fee_increases_loss(self):
        """手续费越高，pnl_loss 应越大"""
        from ml.trading_model import pnl_loss
        T = 30
        pos  = torch.rand(1, T)
        rets = torch.ones(1, T) * 0.005
        loss_low  = pnl_loss(pos, rets, 0.0001, 1.0).item()
        loss_high = pnl_loss(pos, rets, 0.005,  1.0).item()
        self.assertLess(loss_low, loss_high,
                        "手续费越高，loss 应越大（净收益越低）")

    def test_zero_pos_zero_pnl(self):
        """全空仓时 pnl_loss 应约为 0"""
        from ml.trading_model import pnl_loss
        T    = 50
        pos  = torch.zeros(1, T)
        rets = torch.ones(1, T) * 0.01
        loss = pnl_loss(pos, rets, 0.001, 1.0).item()
        self.assertAlmostEqual(loss, 0.0, delta=1e-4,
                               msg="全空仓时 pnl_loss 应约为0")

    def test_output_is_scalar(self):
        """pnl_loss 必须返回标量张量"""
        from ml.trading_model import pnl_loss
        pos  = torch.rand(4, 20)
        rets = torch.randn(4, 20) * 0.01
        loss = pnl_loss(pos, rets, 0.001, 1.5)
        self.assertEqual(loss.shape, torch.Size([]),
                         "pnl_loss 应返回标量（shape=[]）")
```

- [ ] **Step 2：运行测试，确认因 pnl_loss 未定义而 FAIL**

```bash
cd "C:/Users/ranxn/OneDrive/桌面/rxn/lianghua"
python -m pytest test_trading_model.py::TestPnlLoss -v
```

期望输出：`ImportError: cannot import name 'pnl_loss'` 或 `AttributeError`，4个测试均 FAIL。

---

### Task 2：实现 pnl_loss + 修改 DEFAULT_HP + 修改 _train_once

**Files:**
- Modify: `ml/trading_model.py`

- [ ] **Step 1：在 sharpe_loss 函数（第333行）之后，紧接着插入 pnl_loss 函数**

在 `return -sharpe   # 最小化负 Sharpe = 最大化 Sharpe` 这行之后，空一行插入：

```python

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
```

- [ ] **Step 2：在 DEFAULT_HP 字典（约第94行 fee_lambda 行）末尾追加 loss_type 字段**

找到：
```python
    "fee_lambda":    1.5,     # 损失函数中的手续费惩罚倍率
}
```

替换为：
```python
    "fee_lambda":    1.5,     # 损失函数中的手续费惩罚倍率
    "loss_type":     "sharpe", # 损失函数类型：sharpe（夏普代理）或 pnl（纯P&L）
}
```

- [ ] **Step 3：修改 _train_once 中的损失函数调用（约第397行）**

找到：
```python
            loss = sharpe_loss(pos, r_batch, hp["fee_rate"], hp["fee_lambda"])
```

替换为：
```python
            _loss_fn = pnl_loss if hp.get("loss_type", "sharpe") == "pnl" else sharpe_loss
            loss = _loss_fn(pos, r_batch, hp["fee_rate"], hp["fee_lambda"])
```

- [ ] **Step 4：运行 pnl_loss 测试，确认全部 PASS**

```bash
python -m pytest test_trading_model.py::TestPnlLoss -v
```

期望输出：
```
test_trading_model.py::TestPnlLoss::test_fee_increases_loss PASSED
test_trading_model.py::TestPnlLoss::test_output_is_scalar PASSED
test_trading_model.py::TestPnlLoss::test_uptrend_full_pos_lower_loss PASSED
test_trading_model.py::TestPnlLoss::test_zero_pos_zero_pnl PASSED
4 passed
```

- [ ] **Step 5：运行全部测试，确认无回归**

```bash
python -m pytest test_trading_model.py -v
```

期望：所有测试 PASS（TestWalkForward 较慢，约1~2分钟）。

- [ ] **Step 6：提交**

```bash
git add ml/trading_model.py test_trading_model.py
git commit -m "feat: add pnl_loss option and loss_type hyperparameter"
```

---

### Task 3：添加 --loss CLI 参数

**Files:**
- Modify: `ml/trading_model.py`（CLI 部分，约第999-1021行）

- [ ] **Step 1：在 --fee-lambda 参数注册之后（约第1000行）插入 --loss 参数**

找到：
```python
    hp_g.add_argument("--fee-lambda", type=float, default=DEFAULT_HP["fee_lambda"],
                      metavar="F", help=f"手续费惩罚倍率，默认 {DEFAULT_HP['fee_lambda']}")
```

替换为：
```python
    hp_g.add_argument("--fee-lambda", type=float, default=DEFAULT_HP["fee_lambda"],
                      metavar="F", help=f"手续费惩罚倍率，默认 {DEFAULT_HP['fee_lambda']}")
    hp_g.add_argument("--loss", choices=["sharpe", "pnl"],
                      default=DEFAULT_HP["loss_type"],
                      help="损失函数：sharpe（夏普代理，默认）或 pnl（纯P&L最大收益）")
```

- [ ] **Step 2：在 hp 字典构建处（约第1021行）添加 loss_type 字段**

找到：
```python
        "fee_lambda":   args.fee_lambda,
    }
```

替换为：
```python
        "fee_lambda":   args.fee_lambda,
        "loss_type":    args.loss,
    }
```

- [ ] **Step 3：验证 CLI 帮助信息包含 --loss**

```bash
python ml/trading_model.py --help
```

期望输出中包含：`--loss {sharpe,pnl}`

- [ ] **Step 4：运行全部测试确认无回归**

```bash
python -m pytest test_trading_model.py -v
```

期望：全部 PASS。

- [ ] **Step 5：提交**

```bash
git add ml/trading_model.py
git commit -m "feat: add --loss CLI argument for loss function selection"
```

---

### Task 4：运行 000001 平安银行逐日 Walk-forward 训练

**Files:**
- 运行（不修改代码）

- [ ] **Step 1：确认环境依赖已安装**

```bash
pip install torch akshare pandas numpy matplotlib seaborn scipy
```

验证：`python -c "import torch, akshare, pandas; print('OK')"` 输出 `OK`。

- [ ] **Step 2：启动训练（Sharpe损失，默认配置）**

```bash
python ml/trading_model.py --symbol 000001 --start 20180101
```

训练开始后会打印：
```
=======================================================
  ML 交易模型 v2 — 000001  20180101~YYYYMMDD
  模式: Walk-forward
  设备: cpu  (或 cuda)
=======================================================
  有效数据: ~2000 天  (2018-01-02 ~ 2026-04-23)
  特征维度: 23  |  样本数: ~2000
  [Walk-forward 1/~1750] 训练集: 前 252 天 → 预测: [252,253)
  ...
```

> **注意**：逐日重训约需 4~8 小时（CPU）或 30~60 分钟（GPU）。
> 可在另一窗口随时 Ctrl+C 中断，已预测的仓位不会丢失（但报告需训练完成才输出）。

- [ ] **Step 3：（可选）快速验证版——缩短 epochs 先测试流程**

```bash
python ml/trading_model.py --symbol 000001 --start 20180101 --epochs 5 --warm-up 120
```

约 20~30 分钟完成，用于确认数据拉取、特征计算、训练循环均正常，无报错。

- [ ] **Step 4：训练完成后，查看输出报告**

控制台末尾会打印：
```
====================================================
  最新信号（基于 YYYY-MM-DD 数据）
  建议目标仓位 = 0.XXXX  [买入/加仓 | 持仓观望 | 减仓/卖出]
  (0=空仓，0.5=半仓，1=满仓)
====================================================
```

- [ ] **Step 5：查看生成的报告文件**

```bash
# Markdown 报告（含超参数、六大指标、训练/测试对比）
cat charts/000001_ml_v2_report.md

# 图表（净值曲线 / 仓位序列 / 日P&L）
# 在文件管理器中打开：
start charts/000001_ml_v2_result.png
```

- [ ] **Step 6：（可选）使用纯 P&L 损失重训对比**

```bash
python ml/trading_model.py --symbol 000001 --start 20180101 --loss pnl
```

报告保存路径相同，会覆盖之前的结果。建议先备份：
```bash
cp charts/000001_ml_v2_report.md charts/000001_ml_v2_report_sharpe.md
cp charts/000001_ml_v2_result.png charts/000001_ml_v2_result_sharpe.png
```

---

## 使用预测接口（训练完成后）

```python
# 在 Python 中调用
import sys
sys.path.insert(0, ".")
from ml.trading_model import DEFAULT_HP, run

hp = {**DEFAULT_HP}  # 可修改任意超参数

result = run("000001", "20180101", "20260423", hp)

# 获取预测函数
predictor = result["predict"]

# 实盘预测：df 最后一行 open = 今日开盘价，close = NaN
pos = predictor(df)
print(f"建议目标仓位: {pos:.4f}  (0=空仓, 1=满仓)")
```

---

## 性能指标参考（预期）

| 指标 | 说明 |
|------|------|
| 年化收益率 | 目标 > 买持基准 |
| 最大回撤 | 目标 < 30% |
| 夏普比率 | 目标 > 0.8（逐日重训应有改善空间） |
| 日胜率 | 通常 48%~55% |
| 总手续费 | 与换手率正相关，fee_lambda 越大越低 |
