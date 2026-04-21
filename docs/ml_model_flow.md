# ML 交易模型 v2 — 代码流程说明

> 文件：`ml/trading_model.py`  
> 测试：`test_trading_model.py`

---

## 整体架构一览

```
输入 df (OHLCV+指标)
      │
      ▼
build_features()        ← 特征工程（信息隔离：预测日仅看 open）
      │  build_returns()
      ▼
TradingDataset          ← 滑动窗口数据集 (seq_len 天一组)
      │
      ▼
TradingLSTM             ← LSTM + Sigmoid 头 → 仓位 ∈ [0,1]
      │  训练目标: minimize -Sharpe(positions, returns)
      ▼
Walk-forward 循环       ← 每 retrain_freq 天重训, 扩展历史窗口
      │
      ▼
calc_performance()      ← 计算净值/夏普/最大回撤/手续费等
      │
      ▼
Markdown 报告 + PNG 图表
```

---

## 模块流程详解

### Step 1: 数据获取与指标计算

```python
df = fetch_daily(symbol, start, today)    # 一直拉到今天
df = compute_all(df).dropna()             # 计算 MA/EMA/MACD/RSI/布林带
```

### Step 2: 特征工程（`build_features`）

每个预测日 `t` 的特征向量（共 **22 维**）：

| 维度 | 特征 | 说明 |
|------|------|------|
| [0] | 今日开盘缺口 | `open[t]/close[t-1]-1` ← 预测日唯一可见的当日数据 |
| [1-4] | 前日 OHLC 变化率 | 相对更前一日收盘 |
| [5] | 前日成交量比率 | `vol[t-1] / avg_vol_5d` |
| [6-9] | 过去4日收盘收益 | 短期动量特征 |
| [10-21] | 技术指标（前日）| MA/EMA/MACD/RSI/布林带，各自归一化 |

> **信息隔离保证**：`close[t]`、`high[t]`、`low[t]` 均被掩盖，模型绝对看不到预测日的收盘信息。

### Step 3: 模型架构（`TradingLSTM`）

```
输入 (B, seq_len, 22)
    └─→ LSTM(hidden_dim, num_layers)
         └─→ LayerNorm → Linear → GELU → Dropout → Linear → Sigmoid
输出 (B, seq_len)   ← 每时刻的目标仓位 ∈ (0, 1)
```

**关键设计**：Sigmoid 激活替换 Tanh，输出天然落在 `[0, 1]`，满足长仓约束（A股 T+1，不做空）。

### Step 4: 损失函数（`sharpe_loss`）

```python
prev_pos = [0, pos[0], pos[1], ..., pos[T-2]]   # 前一时刻仓位（初始空仓）
delta    = pos - prev_pos                         # 仓位变化 = 交易量
fees     = fee_rate × fee_lambda × |delta|        # 手续费（千分之一）
pnl      = pos × returns - fees                   # 净损益
loss     = -pnl.mean() / (pnl.std() + 1e-8)      # 负夏普（最小化=最大化夏普）
```

**为什么用 Sharpe 而不是单纯总收益**：防止模型走极端策略（e.g., 全程满仓），鼓励在风险调整后收益最大化。

### Step 5: Walk-forward 滚动训练（`run_walk_forward`）

```
Day 0  ─────────────────► Day warm_up  ────────────────────► Day N
│← 预热期：用这段数据训练初始模型 →│← Walk-forward 预测区间 →│

具体流程：
  for t in range(warm_up, N, retrain_freq):
      train_on( all_data[0 : t] )    ← 只用预测日之前的数据！
      predict( data[t : t+retrain_freq] )
  拼接所有预测 → 全周期仓位序列
```

**防止数据泄露的关键**：训练集永远截止到 `t-1`，预测 `t` 日时绝对不会看到 `t` 日及以后的数据。

### Step 6: 绩效计算（`calc_performance`）

```python
delta  = diff([0] + pos)          # 仓位变化（初始从0开始）
fees   = fee_rate × |delta|       # 手续费（千分之一双边）
pnl    = pos × return - fees      # 逐日净损益
equity = cumprod(1 + pnl)         # 净值曲线（复利）
```

输出指标：总收益率、年化收益率、夏普比率、最大回撤、卡玛比率、日胜率、持仓天数、总手续费。

### Step 7: 预测接口（`make_predictor`）

```python
predictor = make_predictor(model, hp, device)

# 每日调用方式：
# df 的最后一行 = 今日（只有 open 已知，close/high/low 为 NaN）
position = predictor(df)    # float ∈ [0, 1]

# 解读：
# 0.0  → 空仓
# 0.5  → 半仓
# 1.0  → 满仓
# 与当前持仓对比，delta>0=买入，delta<0=卖出
```

---

## 超参数调整指南

| 场景 | 建议调整 |
|------|---------|
| 过拟合（训练好测试差） | 增大 `dropout`(0.3)、减小 `hidden_dim`(32)、增大 `fee_lambda` |
| 欠拟合（训练也差） | 增大 `hidden_dim`(128)、增大 `epochs`(100)、增大 `seq_len`(40) |
| 信号太少（很少交易） | 减小 `fee_lambda`(0.5~1.0) |
| 换手太频繁 | 增大 `fee_lambda`(3~5) |
| 想要更及时的预测 | 减小 `retrain_freq`(5~10，慢但更准）|
| 想要更快的运行速度 | 增大 `retrain_freq`(60)、减小 `epochs`(30) |

---

## 运行命令

```bash
# 标准 Walk-forward（推荐，最真实）
python ml/trading_model.py --symbol 002837 --start 20200101

# 快速测试（简单 Train/Test 划分）
python ml/trading_model.py --symbol 002837 --start 20200101 --no-walk-forward --epochs 30

# 自定义超参数（激进配置）
python ml/trading_model.py --symbol 002837 --start 20200101 \
    --epochs 100 --hidden 128 --seq-len 40 --lr 5e-4 \
    --retrain-freq 10 --warm-up 200
```

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `charts/{symbol}_ml_v2_report.md` | Markdown 综合绩效报告 |
| `charts/{symbol}_ml_v2_result.png` | 三联图（净值/仓位/日P&L） |

---

> **风险提示**：所有分析结果仅供参考，不构成投资建议。A股受政策影响显著，回测结果不代表未来表现。
