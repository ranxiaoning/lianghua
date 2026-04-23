# 端到端 LSTM 交易模型设计文档

**日期**: 2026-04-23
**股票**: 000001（平安银行）
**数据范围**: 20180101 → 今天
**仓位约束**: [0, 1]（长仓，不做空）

---

## 1. 架构与输入

### 模型结构

```
输入序列 (seq_len × 23维特征)
       ↓
LSTM (num_layers层, hidden_dim维)
       ↓
LayerNorm → Linear(hidden_dim, hidden_dim//2) → GELU → Dropout → Linear(hidden_dim//2, 1)
       ↓
Sigmoid → 仓位 ∈ (0, 1)
```

### 23维特征（预测日 t 的输入向量）

| 维度 | 来源 | 特征描述 | 信息可见性 |
|------|------|---------|-----------|
| [0] | 今日 | 开盘缺口 `open[t]/close[t-1] - 1` | ✅ 可见 |
| [1-4] | 昨日 | OHLC 变化率（相对更前一日收盘） | ✅ 可见 |
| [5] | 昨日 | 成交量比率（量/近5日均量 - 1） | ✅ 可见 |
| [6-9] | 历史 | 过去4日 close-to-close 收益率（动量） | ✅ 可见 |
| [10-21] | 昨日 | 技术指标：MA5/MA20/EMA12/EMA26/DIFF/DEA/MACD/RSI14/BB上中下/BB宽 | ✅ 可见 |
| — | 今日 | `close[t]`, `high[t]`, `low[t]` | ❌ 完全遮蔽 |

**特征处理**：NaN→0，Inf→clip，最终裁剪至 [-5, 5]

---

## 2. Walk-forward 训练策略

### 模式：逐日扩张窗口（Daily Expanding Walk-forward）

```
时间轴 ──────────────────────────────────────────────────────────►
│← 预热期 252天（约1年）→│←── 逐日滚动预测区 ──►│ 今天

第 252 天：训练集 [0, 251]  → 预测第 252 天
第 253 天：训练集 [0, 252]  → 预测第 253 天
...
第 N 天：  训练集 [0, N-1]  → 预测第 N 天
```

### 关键规则

- **扩张窗口**：每次训练集包含从第0天到预测日前一天的全部历史
- **严格无泄露**：预测日 t 的所有训练数据上界为 t-1
- **每次全新建模**：不继承上一次权重，防止灾难性遗忘
- **retrain_freq = 1**：每天重训（最高精度）

### 计算量预估

| 设备 | epochs=20 | epochs=40 | epochs=60 |
|------|-----------|-----------|-----------|
| CPU  | ~1.5小时  | ~3小时    | ~5小时    |
| GPU  | ~10分钟   | ~20分钟   | ~30分钟   |

---

## 3. 损失函数

通过 `--loss` 参数切换：

### `--loss sharpe`（默认，推荐）

```
P&L_t  = pos[t] × r[t] − |Δpos[t]| × fee_rate × fee_lambda
Loss   = −mean(P&L) / (std(P&L) + ε)
```

- 直接最大化夏普比率代理
- `fee_lambda` 放大手续费惩罚，抑制频繁换手
- 训练更稳定，对波动的容忍度更低

### `--loss pnl`

```
P&L_t  = pos[t] × r[t] − |Δpos[t]| × fee_rate
Loss   = −mean(P&L)
```

- 直接最大化累计净收益
- 对齐"每日盈亏程度"的优化目标
- 更激进，可能产生更高收益但波动更大

### 手续费计算

```
fee_t = |pos[t] − pos[t-1]| × fee_rate
fee_rate = 0.001（千分之一，买卖均扣）
```

---

## 4. 超参数配置

| 参数 | 默认值 | CLI | 调参建议 |
|------|--------|-----|---------|
| 回看窗口 | 60天 | `--seq-len` | 20~120；越大记忆越长 |
| LSTM隐层 | 128 | `--hidden` | 32~256；数据少时用小值 |
| LSTM层数 | 4 | `--layers` | 1~4；深层需更多数据 |
| Dropout | 0.2 | `--dropout` | 0.0~0.4 |
| 初始学习率 | 1e-3 | `--lr` | 5e-4~2e-3（余弦退火到lr×0.01） |
| 每次训练轮数 | 60 | `--epochs` | 20~100 |
| 批大小 | 32 | `--batch` | 16~64 |
| 预热天数 | 252 | `--warm-up` | 120~504（约0.5~2年） |
| 手续费率 | 0.001 | `--fee` | 固定（千分之一） |
| 手续费惩罚倍率 | 1.5 | `--fee-lambda` | 1.0~3.0 |
| 损失函数 | sharpe | `--loss` | sharpe / pnl |

---

## 5. 输出规格

### 控制台输出
- 训练阶段绩效（预热期内，固定空仓）
- 样本外绩效（walk-forward 预测区间）
- 整体绩效（全周期）
- 最新仓位信号（买入/持仓观望/减仓卖出）

### 文件输出

| 文件 | 路径 | 内容 |
|------|------|------|
| Markdown报告 | `charts/000001_ml_v2_report.md` | 超参数表 + 六大指标 + 训练/测试对比 |
| 三联图 | `charts/000001_ml_v2_result.png` | 净值曲线 / 仓位序列 / 日P&L柱图 |

### 预测接口

```python
result = run("000001", "20180101", "today", hp)
predictor = result["predict"]

# 实盘使用：
pos = predictor(df)  # df最后一行open=今日开盘价，close=NaN
# 返回：float ∈ [0, 1]，0=空仓，1=满仓
```

---

## 6. 实施范围

**现有代码**（`ml/trading_model.py`）已实现：
- LSTM模型、Sigmoid输出、[0,1]约束
- 特征工程（23维，信息隔离）
- Walk-forward主循环
- Sharpe损失函数
- 绩效计算、报告、图表
- `make_predictor` 推理API

**唯一新增**：在现有代码中添加 `--loss pnl` 选项（纯P&L损失函数）

---

## 7. 运行命令

```bash
# 标准运行（逐日walk-forward，Sharpe损失）
python ml/trading_model.py --symbol 000001 --start 20180101

# 使用纯P&L损失
python ml/trading_model.py --symbol 000001 --start 20180101 --loss pnl

# 快速版（每40epoch，约3小时CPU）
python ml/trading_model.py --symbol 000001 --start 20180101 --epochs 40

# GPU加速
python ml/trading_model.py --symbol 000001 --start 20180101 --epochs 60
```

---

> **风险提示**: 回测结果不代表未来收益，A 股受政策影响显著，本模型仅供研究参考。
