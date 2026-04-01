from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunAgentRequest(BaseModel):
    as_of: str | None = Field(
        None, description="数据窗口的基准日期（YYYY-MM-DD）。若不传，则默认使用 target_date 或其前一交易日。"
    )
    target_date: str = Field(
        ..., description="预测/建仓目标日期（YYYY-MM-DD）。系统会基于该日期前一月数据生成建议。"
    )
    top_k: int = Field(10, ge=5, le=500, description="候选股票数量上限（最终持仓≥5）。")
    min_holdings: int = Field(5, ge=5, le=30, description="最小持仓数量。")
    model_tag: str = Field(
        "primary",
        description="用于横向对比的模型标签（例如 primary/secondary）。",
    )
    provider: str | None = Field(None, description="可选：本次运行指定模型提供商（mock/openai_compatible/ollama/qwen/kimi/n1n）。")
    model_name: str | None = Field(None, description="可选：本次运行指定模型名（如 deepseek-chat）。")


class Holding(BaseModel):
    code: str
    name: str | None = None
    weight: float = Field(..., ge=0.0, le=1.0)


class AgentStep(BaseModel):
    name: str
    output: str


class SelectedStock(BaseModel):
    code: str
    name: str | None = None
    reason: str


class AgentRunResult(BaseModel):
    as_of: str
    rationale: str
    holdings: list[Holding]
    steps: list[AgentStep] = Field(default_factory=list)
    selected: list[SelectedStock] = Field(default_factory=list)


class BacktestRequest(BaseModel):
    start_month: str = Field(..., description="回测开始月份（YYYY-MM）。")
    end_month: str = Field(..., description="回测结束月份（YYYY-MM），包含该月。")
    model_tag: str = Field("primary", description="要使用的模型标签。")
    provider: str | None = Field(None, description="可选：指定回测时使用的模型提供商。")
    model_name: str | None = Field(None, description="可选：指定回测时使用的模型名称。")


class BacktestMetrics(BaseModel):
    cumulative_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float


class BacktestResult(BaseModel):
    start_month: str
    end_month: str
    metrics: BacktestMetrics
    nav_series: list[float] = Field(default_factory=list)
    nav_dates: list[str] = Field(default_factory=list)


class WeeklyBacktestRequest(BaseModel):
    start_date: str = Field(..., description="回测开始日期（YYYY-MM-DD）。")
    end_date: str = Field(..., description="回测结束日期（YYYY-MM-DD）。")
    model_tag: str = Field("primary", description="要使用的模型标签。")
    provider: str | None = Field(None, description="可选：指定回测时使用的模型提供商。")
    model_name: str | None = Field(None, description="可选：指定回测时使用的模型名称。")
    benchmark: str = Field("sh.510300", description="对比基准代码，默认沪深300ETF。")
    align_mode: bool = Field(False, description="是否启用对齐模式（固定持仓 + close-close口径）。")
    use_fixed_portfolio: bool = Field(False, description="是否在整个回测区间复用第一周AI持仓。")
    price_mode: str = Field("open_close", description="收益口径：open_close 或 close_close。")


class WeeklyBacktestResult(BaseModel):
    ai_nav_series: list[float]
    ai_nav_dates: list[str]
    bench_nav_series: list[float]
    bench_nav_dates: list[str]
    metrics: dict[str, Any]  # 包含 AI 和基准的对比指标
    total_weeks: int = 0
    processed_weeks: int = 0
    failed_weeks: int = 0
    warnings: list[str] = Field(default_factory=list)