from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.agents.graph import run_portfolio_agent
from app.api.schemas import (
    AgentRunResult,
    BacktestRequest,
    BacktestResult,
    RunAgentRequest,
    WeeklyBacktestRequest,
    WeeklyBacktestResult,
)
from app.backtest.engine import run_monthly_backtest
from app.backtest.weekly_engine import run_weekly_backtest

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.post("/agent/run", response_model=AgentRunResult)
def agent_run(req: RunAgentRequest) -> AgentRunResult:
    try:
        return run_portfolio_agent(req)
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/backtest/run", response_model=BacktestResult)
def backtest_run(req: BacktestRequest) -> BacktestResult:
    try:
        return run_monthly_backtest(req)
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/backtest/weekly", response_model=WeeklyBacktestResult)
def backtest_weekly(req: WeeklyBacktestRequest) -> WeeklyBacktestResult:
    try:
        return run_weekly_backtest(req)
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e)) from e
