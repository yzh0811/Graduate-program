from __future__ import annotations

from datetime import date

import pandas as pd

from app.api.schemas import BacktestMetrics, BacktestRequest, BacktestResult, Holding
from app.agents.graph import run_portfolio_agent
from app.data.providers import get_price_panel
from app.backtest.metrics import (
    annualized_return,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
)


def _month_range(start_month: str, end_month: str) -> list[str]:
    s = date.fromisoformat(start_month + "-01")
    e = date.fromisoformat(end_month + "-01")
    months: list[str] = []
    cur = s
    while cur <= e:
        months.append(cur.strftime("%Y-%m"))
        # add one month
        y = cur.year + (1 if cur.month == 12 else 0)
        m = 1 if cur.month == 12 else (cur.month + 1)
        cur = date(y, m, 1)
    return months


def _month_end_iso(month: str) -> str:
    d = date.fromisoformat(month + "-01")
    y = d.year + (1 if d.month == 12 else 0)
    m = 1 if d.month == 12 else (d.month + 1)
    next_month = date(y, m, 1)
    end = date.fromordinal(next_month.toordinal() - 1)
    return end.isoformat()


def run_monthly_backtest(req: BacktestRequest) -> BacktestResult:
    months = _month_range(req.start_month, req.end_month)
    if len(months) == 0:
        return BacktestResult(
            start_month=req.start_month,
            end_month=req.end_month,
            metrics=BacktestMetrics(
                cumulative_return=0.0,
                annualized_return=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
            ),
        )

    all_daily_returns: list[pd.Series] = []

    for month in months:
        as_of = _month_end_iso(month)
        # 1) 让 agent 生成该月末对“下个月”的组合建议
        agent_result = run_portfolio_agent(
            as_of=as_of,
            top_k=10,
            min_holdings=5,
            model_tag=req.model_tag,
            provider=req.provider,
            model_name=req.model_name,
        )
        holdings: list[Holding] = agent_result.holdings
        codes = [h.code for h in holdings]
        w = pd.Series({h.code: h.weight for h in holdings})

        # 2) 用“下个月”的价格计算组合日收益
        # 简化：用 as_of 往后 30 个自然日近似“下个月”
        start = date.fromisoformat(as_of).isoformat()
        end = date.fromordinal(date.fromisoformat(as_of).toordinal() + 30).isoformat()
        close = get_price_panel(codes=codes, start=start, end=end)
        daily_ret = close.pct_change().fillna(0.0)
        port_ret = (daily_ret * w).sum(axis=1)
        all_daily_returns.append(port_ret)

    daily_returns = pd.concat(all_daily_returns).sort_index()
    nav = (1.0 + daily_returns).cumprod()
    metrics = BacktestMetrics(
        cumulative_return=cumulative_return(daily_returns),
        annualized_return=annualized_return(daily_returns),
        sharpe=sharpe_ratio(daily_returns),
        max_drawdown=max_drawdown(daily_returns),
    )
    return BacktestResult(
        start_month=req.start_month,
        end_month=req.end_month,
        metrics=metrics,
        nav_series=[float(x) for x in nav.values],
        nav_dates=[str(x.date()) for x in nav.index],
    )

