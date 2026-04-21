# 网格搜索详细报告 — 代码流程文档

> 本文档详细描述 `amplitude_pipeline.py` 中网格搜索功能从启动到生成综合报告的 **完整调用链路**，
> 确保读者能理解每一个环节的数据流向和处理逻辑。

---

## 目录

1. [总体架构](#1-总体架构)
2. [启动入口 main()](#2-启动入口-main)
3. [数据加载 get_data()](#3-数据加载-get_data)
4. [网格搜索 grid_search()](#4-网格搜索-grid_search)
5. [单组合回测 _run_one()](#5-单组合回测-_run_one)
6. [分析与推荐 _analyze_and_recommend()](#6-分析与推荐-_analyze_and_recommend)
7. [控制台表格 _print_full_grid_table()](#7-控制台表格-_print_full_grid_table)
8. [热力图绘制 _plot_heatmap()](#8-热力图绘制-_plot_heatmap)
9. [综合报告 _build_comprehensive_txt()](#9-综合报告-_build_comprehensive_txt)
10. [推荐参数完整回测](#10-推荐参数完整回测)
11. [输出文件清单](#11-输出文件清单)
12. [报告章节详解](#12-报告章节详解)

---

## 1. 总体架构

```
用户命令行 (--grid)
      │
      ▼
   main()              ← CLI 解析
      │
      ├─ get_data()     ← 数据加载/拉取
      │
      └─ grid_search()  ← 网格搜索入口
            │
            ├─ _run_one() × N 次     ← 逐个参数组合回测
            │     │
            │     ├─ run_backtest()   ← 回测引擎 (backtest_amplitude.py)
            │     ├─ calc_metrics()   ← 绩效计算
            │     └─ 计算盈亏比       ← profit_factor
            │
            ├─ 保存 grid CSV          ← 全量结果落盘
            │
            └─ _analyze_and_recommend()  ← 分析+推荐
                  │
                  ├─ _print_full_grid_table()      ← 控制台详细表格
                  ├─ _plot_heatmap()                ← 热力图 PNG
                  ├─ _build_comprehensive_txt()     ← 综合报告 TXT
                  └─ 推荐参数完整回测               ← 图表+月度+交易记录
```

---

## 2. 启动入口 main()

**文件**: `amplitude_pipeline.py` → `main()`

**流程**:
1. `argparse` 解析命令行参数
2. 判断 `--grid` 标志位
3. 如果 `--grid` 为 True → 调用 `grid_search()`
4. 否则 → 调用 `single_run()`（单次回测，不在本文档范围内）

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | 必填 | 股票代码 |
| `--start` | 必填 | 开始日期 YYYYMMDD |
| `--end` | 必填 | 结束日期 YYYYMMDD |
| `--grid` | False | 启用网格学习 |
| `--buy-range` | 88 99 | buy_pct 扫描范围 |
| `--sell-range` | 101 115 | sell_pct 扫描范围 |
| `--step` | 1.0 | 网格步长 |
| `--min-trades` | 3.0 | 年均最小交易数过滤 |
| `--cash` | 100000 | 初始资金 |

---

## 3. 数据加载 get_data()

**流程**:
1. 扫描 `data/` 目录下匹配 `{symbol}_daily_*_*_{adjust}.csv` 的文件
2. 逐文件读取，过滤日期范围，取行数最多的文件作为最佳缓存
3. 如果找到缓存 → 直接使用
4. 如果无缓存 → 调用 `fetch_daily()` 从 akshare 拉取，保存到 `data/`

**输出**: `pd.DataFrame`，包含 `date, open, high, low, close, volume` 列

---

## 4. 网格搜索 grid_search()

**流程**:
1. 根据 `buy_range`, `sell_range`, `step` 生成参数网格
   - `buy_vals = np.arange(buy_range[0], buy_range[1] + step/2, step)`
   - `sell_vals = np.arange(sell_range[0], sell_range[1] + step/2, step)`
   - 只保留 `buy < sell` 的有效组合
2. 逐个组合调用 `_run_one()`，收集结果到 `rows` 列表
3. 组装为 `pd.DataFrame`
4. 保存完整网格 CSV → `backtest/{symbol}_grid_{start}_{end}.csv`
5. 调用 `_analyze_and_recommend()`，传入 `buy_range, sell_range, step`

**关键**: 第 5 步传参修复了之前缺少 `buy_range/sell_range/step` 的 bug。

---

## 5. 单组合回测 _run_one()

**流程**:
1. 调用 `run_backtest(df, buy, sell, cash)` → 获取回测结果
2. 调用 `calc_metrics(result, df)` → 计算绩效指标
3. **新增**: 计算盈亏比 (Profit Factor)
   - 提取所有 SELL 交易
   - `total_win` = 盈利交易的 profit_pct 之和
   - `total_loss` = 亏损交易的 profit_pct 之和（取绝对值）
   - `profit_factor = total_win / total_loss`（无亏损时为 inf）

**返回字段** (共 15 个):

| 字段 | 说明 | 来源 |
|------|------|------|
| `buy_pct` | 买入参数 | 输入 |
| `sell_pct` | 卖出参数 | 输入 |
| `ann_ret` | 年化收益率 (%) | calc_metrics |
| `total_ret` | 总收益率 (%) | calc_metrics |
| `max_dd` | 最大回撤 (%) | calc_metrics |
| `sharpe` | 夏普比率 | calc_metrics |
| `calmar` | 卡玛比率 | calc_metrics |
| `win_rate` | 胜率 (%) | calc_metrics |
| `avg_profit` | 平均单笔盈亏 (%) | calc_metrics |
| `n_trades` | 交易笔数 | calc_metrics |
| `ann_trades` | 年均交易笔数 | 计算 |
| `years` | 回测年限 | calc_metrics |
| `bh_ret` | 买持基准总收益 (%) | calc_metrics |
| `bh_ann` | 买持基准年化 (%) | calc_metrics |
| `profit_factor` | 盈亏比 | 本函数计算 |

---

## 6. 分析与推荐 _analyze_and_recommend()

**流程**:
1. **过滤有效组合**: `valid = results[ann_trades >= min_trades]`
   - 如果 valid 为空 → 警告并使用全部结果
2. **确定推荐参数**:
   - `bp_row` = valid 中 `ann_ret` 最高的行（收益最高）
   - `bs_row` = valid 中 `calmar` 最高的行（最稳妥）
3. **输出四部分**:
   - ① `_print_full_grid_table()` → 控制台详细表格
   - ② `_plot_heatmap()` → 热力图 PNG
   - ③ `_build_comprehensive_txt()` → 综合报告 TXT
   - ④ 推荐参数完整回测 → 图表 + 月度收益 + 交易记录

---

## 7. 控制台表格 _print_full_grid_table()

**输出结构**:

```
═══════════════════════════════════════════
  网格学习结果  过滤: ...  扫描: ... 组  有效: ... 组
═══════════════════════════════════════════

  ── 统计概览 ──                          ← 新增: 各指标的平均/中位数/极值
  [全量] 年化收益: 平均=...  中位数=...
  [全量] 最大回撤: ...
  [全量] 夏普比率: ...
  [全量] 胜率: ...
  [全量] 盈利组合: .../...  买持基准年化: ...
  [全量] 跑赢买持: .../...
  [有效] ...（同上）

  ── 全量结果 ──                          ← 按年化收益降序
  # buy%  sell%  年化%  总收益%  回撤%  夏普  卡玛  胜率%  盈亏比  年均交易  备注

  ── 有效参数 ──                          ← 按卡玛比率降序

  ★ 收益最高: buy=...  sell=...  年化=...  总收益=...  回撤=...  夏普=...  胜率=...
  ★ 最稳妥:   buy=...  sell=...  年化=...  总收益=...  回撤=...  卡玛=...  胜率=...
═══════════════════════════════════════════
```

相比之前的改进:
- **新增统计概览区**: 一眼看清全量/有效组合的分布情况
- **表格新增胜率%、盈亏比列**: 信息更完整
- **推荐摘要扩展**: 增加总收益、回撤、夏普/卡玛、胜率
- **新增盈利组合统计和跑赢基准占比**

---

## 8. 热力图绘制 _plot_heatmap()

**输出**: `backtest/{symbol}_grid_heatmap_{start}_{end}.png`

**4 个子图**:
1. 年化收益率热力图
2. 卡玛比率热力图
3. 最大回撤热力图
4. 年均交易数热力图

**特性**: 达到年均交易数要求的格子数字加粗显示。

---

## 9. 综合报告 _build_comprehensive_txt()

这是本次修改的**核心重点**。报告从之前的 5 个章节扩展为 **11 个章节**。

### 报告章节结构

```
═══════════ 报告头部 ═══════════
  符号、扫描范围、回测区间、回测年限、过滤条件、有效率、买持基准

【零】全局概览统计
  ├─ 指标汇总表（均值/中位数/标准差/最小/最大）
  │   年化收益率、总收益率、最大回撤、夏普、卡玛、胜率、盈亏比...
  ├─ 盈利/亏损组合统计
  └─ 跑赢买持基准占比

【一】全量扫描结果（按年化收益降序）
  └─ 完整表格，含胜率、盈亏比

【二】有效参数详细（按卡玛比率降序）
  └─ 仅年均交易达标的组合

【三a~f】参数敏感性矩阵（6 个子矩阵）
  ├─ 三a: 年化收益率 (%)
  ├─ 三b: 卡玛比率
  ├─ 三c: 最大回撤 (%)
  ├─ 三d: 夏普比率
  ├─ 三e: 胜率 (%)
  └─ 三f: 年均交易数

【四】各指标排行 Top-10 & Bottom-10
  ├─ 年化收益率 Top/Bottom
  ├─ 卡玛比率 Top
  ├─ 夏普比率 Top
  ├─ 最大回撤 最小/最大
  ├─ 胜率 Top
  └─ 盈亏比 Top

【五】稳定性分析
  └─ 回撤<中位数 & 胜率>中位数 & 交易达标 → 低风险高胜率组合

【六】参数边际效应分析
  ├─ (a) buy_pct 边际效应: 各 buy_pct 的平均年化/回撤/夏普/卡玛/胜率/交易数
  └─ (b) sell_pct 边际效应: 各 sell_pct 的平均年化/回撤/夏普/卡玛/胜率/交易数

【七】跑赢基准分析
  ├─ 买持基准年化/总收益
  ├─ 超额收益分布（均值/中位数/最大/最小）
  └─ [有效且跑赢基准] 组合列表

【八】风险收益象限分析
  ├─ Q1 高收益+低风险 (★最佳) → 详情列表
  ├─ Q2 高收益+高风险
  ├─ Q3 低收益+低风险
  └─ Q4 低收益+高风险 (最差)

【九】参数组合综合评分排行
  ├─ 评分公式: 年化*30% + 卡玛*25% + 夏普*20% + 胜率*15% + 低回撤*10%
  ├─ Min-Max 归一化后加权得分
  └─ 按综合分降序排列

【十】最终推荐
  ├─ ★ 收益最高: 完整 7 项指标详情
  ├─ ★ 最稳妥: 完整 7 项指标详情
  ├─ 风险提示
  ├─ 参数说明
  └─ 评分说明
```

### 关键数据流

```
results DataFrame (全量)
      │
      ├─────────────────────────────────────────────────┐
      │                                                 │
      ▼                                                 ▼
  valid DataFrame (过滤后)                       统计计算
      │                                          (mean/median/std/min/max)
      ├─ bp_row (ann_ret 最大)                          │
      ├─ bs_row (calmar 最大)                           ▼
      │                                         【零】概览统计表
      ├───────────────────────────────┐
      │                               │
      ▼                               ▼
  【一】全量表          【二】有效参数表
      │                               │
      ▼                               ▼
  【三】矩阵            【四】排行榜
  (buy_pct × sell_pct)  (Top/Bottom 10)
      │                               │
      ▼                               ▼
  【五】稳定性筛选      【六】边际效应
  (dd<中位 & wr>中位)   (按参数分组聚合)
      │                               │
      ▼                               ▼
  【七】超额收益        【八】四象限
  (策略 - 买持基准)     (收益×风险交叉)
      │
      ▼
  【九】综合评分
  (归一化加权)
      │
      ▼
  【十】最终推荐
```

---

## 10. 推荐参数完整回测

对收益最高和最稳妥两个推荐参数，分别执行完整回测：

1. `run_backtest()` → 完整回测
2. `calc_metrics()` → 绩效计算
3. `print_report()` → 详细回测报告 TXT
4. `plot_result()` → 净值曲线 + 回撤图 PNG
5. 月度收益 CSV
6. 交易记录 CSV

---

## 11. 输出文件清单

执行一次 `--grid` 网格搜索后，生成以下文件：

| 文件 | 说明 |
|------|------|
| `{symbol}_grid_{start}_{end}.csv` | 全量网格结果 CSV |
| `{symbol}_grid_heatmap_{start}_{end}.png` | 4 面板热力图 |
| `{symbol}_grid_report_{start}_{end}.txt` | **综合报告（11 章节）** |
| `{symbol}_grid_profit_{buy}_{sell}_{start}_{end}.txt` | 收益最高参数回测报告 |
| `{symbol}_amplitude_{buy}_{sell}_{start}_{end}.png` | 收益最高参数净值图 |
| `{symbol}_grid_profit_monthly_{buy}_{sell}_{start}_{end}.csv` | 收益最高参数月度收益 |
| `{symbol}_grid_profit_trades_{buy}_{sell}_{start}_{end}.csv` | 收益最高参数交易记录 |
| `{symbol}_grid_safest_*` | 最稳妥参数同上 4 个文件（如果与收益最高不同） |

---

## 12. 报告章节详解

### 【零】全局概览统计

**目的**: 一眼掌握整个参数空间的表现分布。

**内容**:
- 9 个指标的统计描述（均值/中位数/标准差/最小/最大）
- 盈利 vs 亏损组合占比
- 跑赢买持基准的组合占比
- 有效组合的单独统计

### 【三】参数敏感性矩阵 (6 个)

**目的**: 可视化参数空间，快速定位最优区域。

6 个矩阵分别展示:
1. **年化收益率** — 核心收益指标
2. **卡玛比率** — 风险调整后收益
3. **最大回撤** — 风险控制
4. **夏普比率** — 每单位风险的超额回报
5. **胜率** — 交易成功率
6. **年均交易数** — 策略活跃度

### 【五】稳定性分析

**目的**: 筛选出"又稳又准"的参数组合。

**筛选条件**: 同时满足
- 回撤 ≤ 有效组合的中位数回撤
- 胜率 ≥ 有效组合的中位数胜率

### 【六】参数边际效应

**目的**: 分析 buy_pct 和 sell_pct 各自对策略的影响。

**方法**: 固定一个参数维度，对另一个维度求平均。

### 【八】风险收益象限

**目的**: 四象限分类，直观展示参数组合的风险收益定位。

```
            低风险 ←───→ 高风险
高收益 │  Q1 (★最佳)  │  Q2        │
       │               │            │
低收益 │  Q3           │  Q4 (最差) │
```

### 【九】综合评分

**目的**: 多维度加权得分，给出客观排名。

**评分公式**:
```
综合分 = 归一化(年化收益) × 30%
       + 归一化(卡玛比率) × 25%
       + 归一化(夏普比率) × 20%
       + 归一化(胜率)     × 15%
       + 归一化(-回撤)    × 10%
```

归一化方法: Min-Max，映射到 [0, 1]。

---

## 附录: 运行示例

```bash
# 完整网格搜索（使用默认参数范围）
python amplitude_pipeline.py --symbol 601288 --start 20200101 --end 20251231 --grid

# 自定义参数范围和步长
python amplitude_pipeline.py --symbol 601288 --start 20200101 --end 20251231 \
    --grid --buy-range 90 99 --sell-range 101 115 --step 1 --min-trades 5

# 测试
python -m pytest test_grid_report.py -v
```
