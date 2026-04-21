import numpy as np
import pandas as pd
import statsmodels.api as sm

def calculate_alpha_beta(portfolio_returns, market_returns, rf_rate=0.02/252):
    """
    计算日频 Alpha 和 Beta
    """
    # 1. 计算超额收益
    y = portfolio_returns - rf_rate
    x = market_returns - rf_rate

    # 2. 线性回归: y = alpha + beta * x
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()

    alpha = model.params[0]
    beta = model.params[1]

    return alpha, beta

# 示例数据
alpha_daily, beta = calculate_alpha_beta(returns_df['strategy'], returns_df['benchmark'])
print(f"Daily Alpha: {alpha_daily:.4f}, Beta: {beta:.2f}")
