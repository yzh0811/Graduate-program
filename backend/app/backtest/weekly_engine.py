from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from app.api.schemas import (
    BacktestMetrics,
    Holding,
    WeeklyBacktestRequest,
    WeeklyBacktestResult,
)
from app.agents.graph import run_portfolio_agent
from app.data.providers import get_ohlc_panel
from app.data.trading_calendar import week_trading_range, prev_week_last_trading_day
from app.backtest.metrics import (
    annualized_return,
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
)


def _week_range(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    """Generate list of (week_start, week_end, as_of) tuples from start_date to end_date.
    
    Returns weeks where as_of (last trading day of previous week) >= start_date.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    
    # Start from the Monday of the week containing start_date
    weeks = []
    current = start
    
    while current <= end:
        week_start, week_end = week_trading_range(current)
        as_of = prev_week_last_trading_day(week_start)
        
        # Only include weeks where the as_of is within range and there's a valid trading week
        if date.fromisoformat(as_of) >= start - timedelta(days=7):
            weeks.append((week_start, week_end, as_of))
        
        # Move to next week
        current = date.fromisoformat(week_end) + timedelta(days=1)
    
    return weeks


def run_weekly_backtest(req: WeeklyBacktestRequest) -> WeeklyBacktestResult:
    """Run weekly fund manager backtest with benchmark comparison.
    
    For each week:
    1. Use last Friday's data (as_of) to generate portfolio
    2. Buy at Monday's open price
    3. Sell/calculate at Friday's close price
    4. Compare against benchmark ETF (e.g., sh.510300)
    """
    weeks = _week_range(req.start_date, req.end_date)
    
    if not weeks:
        return WeeklyBacktestResult(
            ai_nav_series=[1.0],
            ai_nav_dates=[req.start_date],
            bench_nav_series=[1.0],
            bench_nav_dates=[req.start_date],
            metrics={},
        )
    
    ai_returns: list[tuple[str, float]] = []  # (date, weekly_return)
    bench_returns: list[tuple[str, float]] = []
    
    for week_start, week_end, as_of in weeks:
        # 1. Generate portfolio using last Friday's data
        try:
            agent_result = run_portfolio_agent(
                target_date=week_start,  # This will internally compute as_of
                top_k=10,
                min_holdings=5,
                model_tag=req.model_tag,
                provider=req.provider,
                model_name=req.model_name,
            )
            holdings: list[Holding] = agent_result.holdings
        except Exception as e:
            print(f"[WARN] Failed to generate portfolio for week {week_start}-{week_end}: {e}")
            continue
        
        if not holdings:
            print(f"[WARN] Empty portfolio for week {week_start}-{week_end}")
            continue
        
        codes = [h.code for h in holdings]
        weights = pd.Series({h.code: h.weight for h in holdings})
        
        # 2. Get Monday open and Friday close prices using OHLC panel
        try:
            panel_start = min(week_start, as_of)
            panel_end = week_end
            
            # Get OHLC panel for portfolio codes
            ohlc = get_ohlc_panel(codes=codes, start=panel_start, end=panel_end)
            
            if week_start not in ohlc["open"].index or week_end not in ohlc["close"].index:
                print(f"[WARN] Missing price data for week {week_start}-{week_end}")
                continue
            
            # Calculate portfolio value change: buy at week_start open, sell at week_end close
            start_prices = ohlc["open"].loc[week_start]
            end_prices = ohlc["close"].loc[week_end]
            
            # Weekly return for each stock
            stock_returns = (end_prices / start_prices - 1).fillna(0)
            port_return = (stock_returns * weights).sum()
            
            ai_returns.append((week_end, float(port_return)))
            
            # 3. Get benchmark return for the same week (open to close)
            bench_ohlc = get_ohlc_panel(codes=[req.benchmark], start=panel_start, end=panel_end)
            
            if week_start in bench_ohlc["open"].index and week_end in bench_ohlc["close"].index:
                bench_start = bench_ohlc["open"].loc[week_start, req.benchmark]
                bench_end = bench_ohlc["close"].loc[week_end, req.benchmark]
                bench_return = float(bench_end / bench_start - 1)
                bench_returns.append((week_end, bench_return))
            else:
                bench_returns.append((week_end, 0.0))
                
        except Exception as e:
            print(f"[WARN] Failed to calculate returns for week {week_start}-{week_end}: {e}")
            continue
    
    # Calculate NAV series
    if not ai_returns:
        return WeeklyBacktestResult(
            ai_nav_series=[1.0],
            ai_nav_dates=[req.start_date],
            bench_nav_series=[1.0],
            bench_nav_dates=[req.start_date],
            metrics={},
        )
    
    # AI NAV
    ai_dates = [r[0] for r in ai_returns]
    ai_weekly_rets = pd.Series([r[1] for r in ai_returns], index=pd.to_datetime(ai_dates))
    ai_nav = (1.0 + ai_weekly_rets).cumprod()
    
    # Benchmark NAV
    bench_dates = [r[0] for r in bench_returns]
    bench_weekly_rets = pd.Series([r[1] for r in bench_returns], index=pd.to_datetime(bench_dates))
    bench_nav = (1.0 + bench_weekly_rets).cumprod()
    
    # Calculate metrics
    def calc_metrics(rets: pd.Series, label: str) -> dict[str, Any]:
        if len(rets) < 2:
            return {
                "cumulative_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }
        
        # For weekly returns, annualize with 52 weeks
        weekly_vol = rets.std(ddof=1)
        annual_vol = weekly_vol * (52 ** 0.5)
        
        # Sharpe (assuming 0% risk-free for simplicity)
        sharpe = (rets.mean() * 52) / annual_vol if annual_vol > 0 else 0.0
        
        # Win rate
        win_rate = (rets > 0).sum() / len(rets)
        
        # Max drawdown from NAV
        nav = (1.0 + rets).cumprod()
        running_max = nav.cummax()
        drawdown = (nav - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            "cumulative_return": float(nav.iloc[-1] - 1.0),
            "annualized_return": float(rets.mean() * 52),
            "volatility": float(annual_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
        }
    
    ai_metrics = calc_metrics(ai_weekly_rets, "AI")
    bench_metrics = calc_metrics(bench_weekly_rets, "Benchmark")
    
    # Excess metrics
    excess_rets = ai_weekly_rets - bench_weekly_rets
    excess_metrics = {
        "excess_cumulative": float((1.0 + excess_rets).cumprod().iloc[-1] - 1.0) if len(excess_rets) > 0 else 0.0,
        "excess_annualized": float(excess_rets.mean() * 52) if len(excess_rets) > 0 else 0.0,
        "information_ratio": float(excess_rets.mean() / excess_rets.std(ddof=1) * (52 ** 0.5)) if len(excess_rets) > 1 and excess_rets.std() > 0 else 0.0,
        "excess_win_rate": float((excess_rets > 0).sum() / len(excess_rets)) if len(excess_rets) > 0 else 0.0,
    }
    
    return WeeklyBacktestResult(
        ai_nav_series=[float(x) for x in ai_nav.values],
        ai_nav_dates=[str(d.date()) for d in ai_nav.index],
        bench_nav_series=[float(x) for x in bench_nav.values],
        bench_nav_dates=[str(d.date()) for d in bench_nav.index],
        metrics={
            "ai": ai_metrics,
            "benchmark": bench_metrics,
            "excess": excess_metrics,
        },
    )
