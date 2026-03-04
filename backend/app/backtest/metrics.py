from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_return(daily_returns: pd.Series) -> float:
    return float((1.0 + daily_returns).prod() - 1.0)


def annualized_return(daily_returns: pd.Series, trading_days: int = 252) -> float:
    if len(daily_returns) == 0:
        return 0.0
    total = (1.0 + daily_returns).prod()
    years = len(daily_returns) / trading_days
    if years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def sharpe_ratio(daily_returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free_daily
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(252) * excess.mean() / std)


def max_drawdown(daily_returns: pd.Series) -> float:
    if len(daily_returns) == 0:
        return 0.0
    nav = (1.0 + daily_returns).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())

