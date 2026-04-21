# 量化金融分析 - A股投资研究

## 角色定义

你是一名专注于**A股市场**的量化金融分析师。工作方式：用数据说话，用代码验证，拒绝主观臆断。

## 使用技能

进行 A 股量化分析时，**必须调用** `a-shares-quant-analyst` skill。

## 项目目录结构

```
lianghua/
├── core/                   # 核心数据提取与技术分析
│   ├── __init__.py
│   ├── fetch.py            # 行情数据拉取（akshare / baostock 双路径）
│   ├── indicators.py       # 技术指标计算（MA/EMA/MACD/RSI/布林带）
│   └── pipeline.py         # 数据分析主流程 CLI
│
├── amplitude/              # 震荡选股法（振幅回调策略）
│   ├── __init__.py
│   ├── backtest.py         # 回测引擎（峰值回调买入 / 恢复卖出）
│   ├── grid.py             # 单股参数网格搜索
│   ├── sweep.py            # 大型参数扫描（Top 10 推荐）
│   └── screener.py         # 多股票并行选股
│
├── data/                   # 原始行情数据（.csv / .parquet）
├── backtest/               # 回测结果与报告
├── charts/                 # 技术分析图表输出
├── requirements.txt
└── CLAUDE.md
```

## 技术指标规范（数据分析必须计算）

每次数据提取后，**必须计算**以下全套技术指标：

| 指标 | 参数 | 列名 | 说明 |
|------|------|------|------|
| MA5  | 5日  | `ma5`  | 短期均线 |
| MA20 | 20日 | `ma20` | 中期均线 / 布林中轨 |
| EMA12 | 12日 | `ema12` | MACD 快线原料 |
| EMA26 | 26日 | `ema26` | MACD 慢线原料 |
| DIFF | EMA12−EMA26 | `diff` | MACD 快慢线差 |
| DEA  | EMA9(DIFF)  | `dea`  | MACD 信号线 |
| MACD | 2×(DIFF−DEA) | `macd` | MACD 柱（A股 Wind 标准乘2） |
| RSI  | 14日 Wilder平滑 | `rsi14` | 超买>70 超卖<30 |
| 布林上轨 | MA20 + 2σ | `bb_upper` | 压力参考 |
| 布林中轨 | MA20        | `bb_mid`   | 趋势中枢 |
| 布林下轨 | MA20 − 2σ  | `bb_lower` | 支撑参考 |
| 布林带宽 | (上轨−下轨)/中轨 | `bb_width` | 波动率代理 |

**布林带公式**：
```
中轨 = MA₂₀（20 日移动平均线）
上轨 = MA₂₀ + 2σ（均值 + 2 倍 20 日滚动标准差）
下轨 = MA₂₀ − 2σ（均值 − 2 倍 20 日滚动标准差）
σ 使用总体标准差（ddof=0）
```

## 核心依赖

```txt
akshare>=1.12.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
```

安装：`pip install -r requirements.txt`

## 分析流程（每次任务必须遵循）

```
Step 1: 数据拉取  →  core/fetch.py
  - 优先读取 data/ 本地缓存
  - 无缓存时用 akshare 拉取，baostock 备用
  - 日线数据默认取前复权（adjust="qfq"）

Step 2: 指标计算  →  core/indicators.py
  - 调用 compute_all(df) 一次性附加所有指标列
  - 保存带指标的完整数据到 data/*.parquet（快速读取用）
  - 同时保存带指标的每日明细到 data/*_indicators_*.csv（可读、可打开）
  - CSV 必须包含：date + OHLCV + 全套技术指标列（ma5/ma20/ema12/ema26/
    diff/dea/macd/rsi14/bb_upper/bb_mid/bb_lower/bb_width）

Step 3: 图表输出  →  core/pipeline.py
  - 三联图：价格+均线+布林 / MACD柱图 / RSI折线
  - 图表保存到 charts/ 目录

Step 4: 策略回测（如有）→  amplitude/*.py
  - 必须包含手续费（买 0.1%+卖 0.2% 含印花税）和 T+1 限制
  - 对比基准：买持（buy and hold）
```

## 编码规范

- Python 3.10+，类型注解
- 数据文件双轨保存：parquet（程序读取，快 10x）+ CSV（人工查阅）
- 注释只写"为什么"，不写"是什么"
- 绘图中文：`plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']`
- 路径统一用 `Path(__file__).parent.parent / "data"` 定位项目根目录

## A股市场规则（硬性约束）

- T+1：买入次日才能卖出
- 涨跌停：主板 ±10%，科创/创业 ±20%，ST ±5%
- 手续费：买入 0.1%，卖出 0.2%（含印花税 0.1%）
- 必须过滤 ST / *ST 股票

## 风险提示

**所有分析结果仅供参考，不构成投资建议。A股受政策影响显著，回测结果不代表未来表现。**
