from __future__ import annotations

import json
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.models import get_llm
from app.agents.prompts import (
    ANALYST_PROMPT,
    PLANNER_PROMPT,
    SELECTOR_PROMPT,
    SYSTEM_PROMPT,
    WEIGHTER_PROMPT,
)
from app.api.schemas import AgentRunResult, AgentStep, Holding, RunAgentRequest, SelectedStock
from app.core.config import settings
from app.data.providers import enrich_selected_with_metrics, get_market_snapshot
from app.data.trading_calendar import week_trading_range, prev_week_last_trading_day


class AgentState(TypedDict, total=False):
    as_of: str
    target_date: str
    top_k: int
    min_holdings: int
    model_tag: str
    provider: str
    model_name: str

    market_summary: str
    universe: list[dict]

    plan: str
    analysis: str
    selected: list[dict]
    holdings: list[Holding]
    rationale: str


def _node_fetch_data(state: AgentState) -> AgentState:
    snap = get_market_snapshot(state["as_of"], top_k=state["top_k"])
    target_date = state.get("target_date", "")
    week_start = state.get("week_start", "")
    week_end = state.get("week_end", "")
    week_line = f"周度区间：{week_start} ~ {week_end}" if week_start and week_end else ""
    return {
        "market_summary": (
            snap.summary_text
            + (f"\n预测目标日：{target_date}" if target_date else "")
            + (f"\n{week_line}" if week_line else "")
        ),
        "universe": snap.universe,
    }


def _node_plan(state: AgentState) -> AgentState:
    llm = get_llm(
        state.get("model_tag", "primary"),
        provider_override=state.get("provider"),
        model_override=state.get("model_name"),
    )
    prompt = "\n\n".join(
        [
            SYSTEM_PROMPT,
            PLANNER_PROMPT,
            f"as_of={state['as_of']}\ntarget_date={state.get('target_date','')}",
            state.get("market_summary", ""),
        ]
    )
    return {"plan": llm.invoke(prompt).text}


def _node_analyze(state: AgentState) -> AgentState:
    llm = get_llm(
        state.get("model_tag", "primary"),
        provider_override=state.get("provider"),
        model_override=state.get("model_name"),
    )
    prompt = "\n\n".join(
        [
            SYSTEM_PROMPT,
            ANALYST_PROMPT,
            "数据摘要：",
            state.get("market_summary", ""),
            "候选池（示例）：",
            "\n".join(
                f"- {x.get('code')} {x.get('name','')} {x.get('industry','')}"
                for x in (state.get("universe") or [])[:15]
            ),
        ]
    )
    return {"analysis": llm.invoke(prompt).text}


def _safe_parse_json_list(text: str) -> list[dict]:
    t = (text or "").strip()
    if not t:
        return []
    # 尝试直接解析
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        return []
    except Exception:
        pass

    # 尝试截取第一个 [ ... ] 段
    l = t.find("[")
    r = t.rfind("]")
    if l >= 0 and r > l:
        try:
            obj = json.loads(t[l : r + 1])
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except Exception:
            return []
    return []


def _node_select(state: AgentState) -> AgentState:
    universe = state.get("universe") or []
    min_holdings = 10

    # 给选股阶段前置真实的近20日指标，让模型必须“看实际”而不是讲故事
    enriched_universe = enrich_selected_with_metrics(
        universe[: min(50, len(universe))],
        as_of=state["as_of"],
        lookback_days=20,
    )

    llm = get_llm(
        state.get("model_tag", "primary"),
        provider_override=state.get("provider"),
        model_override=state.get("model_name"),
    )
    prompt = "\n\n".join(
        [
            SYSTEM_PROMPT,
            SELECTOR_PROMPT.format(min_holdings=min_holdings),
            "分析要点：",
            state.get("analysis", ""),
            "候选池（近20日真实指标）：",
            "\n".join(
                (
                    lambda m: (
                        f"- {x.get('code')} {x.get('name','')} {x.get('industry','')}"
                        f" | 20d收益={m.get('total_return_20d','NA')}"
                        f" | 20d波动(年化)={m.get('volatility_20d','NA')}"
                        f" | 20d最大回撤={m.get('max_drawdown_20d','NA')}"
                        f" | 20d成交额均值={m.get('avg_amount_20d','NA')}"
                    )
                )(x.get("metrics") or {})
                for x in enriched_universe
            ),
        ]
    )

    raw = llm.invoke(prompt).text
    picked = _safe_parse_json_list(raw)

    # 校验与回退：过滤掉不在候选池的 code
    universe_map = {str(x.get("code")): x for x in universe if x.get("code")}
    selected: list[dict] = []
    for item in picked:
        code = str(item.get("code") or "").strip()
        if not code or code not in universe_map:
            continue
        base = universe_map[code]
        selected.append(
            {
                "code": code,
                "name": item.get("name") or base.get("name"),
                "reason": str(item.get("reason") or "").strip() or "未提供理由",
            }
        )

    if len(selected) < max(min_holdings, 5):
        # 不足则基于指标打分回退（Momentum 逻辑）
        # 计算打分：0.5*收益 + 0.3*回撤(越接近0越好) + 0.2*成交额，并对极端波动做惩罚
        def calc_score(x):
            m = x.get("metrics") or {}
            ret = m.get("total_return_20d") or 0.0
            dd = m.get("max_drawdown_20d") or -1.0
            amt = m.get("avg_amount_20d") or 0.0
            vol = m.get("volatility_20d") or 0.0
            
            # 简单线性打分 (实际应做 rank 归一化，这里做简化处理)
            # 成交额仅做辅助，降低其权重
            score = ret * 12.0 + (1.0 + dd) * 6.0 + (amt / 1e9) * 0.5
            if vol > 0.8: # 允许高波动，但排除极端高波动
                score -= 5.0
            return score

        sorted_universe = sorted(enriched_universe, key=calc_score, reverse=True)
        selected = []
        for x in sorted_universe[: max(min_holdings, 5)]:
            m = x.get("metrics") or {}
            def safe_pct(v):
                try:
                    return f"{float(v):.2%}"
                except (TypeError, ValueError):
                    return "NA"
            
            selected.append(
                {
                    "code": x.get("code"),
                    "name": x.get("name"),
                    "reason": (
                        f"回退选择(Momentum评分)：20d收益={safe_pct(m.get('total_return_20d'))}, "
                        f"最大回撤={safe_pct(m.get('max_drawdown_20d'))}, "
                        f"波动={safe_pct(m.get('volatility_20d'))}"
                    ),
                }
            )
        selector_notes = "模型选股输出不可用或数量不足，已按 Momentum 指标打分回退。\n\n" + raw
        return {"selected": selected, "rationale": selector_notes}

    selector_notes = "模型选股 JSON 输出：\n" + raw
    return {"selected": selected, "rationale": selector_notes}


def _normalize_and_clip_weights(
    items: list[dict],
    max_weight: float,
) -> list[dict]:
    # items: [{"code":..., "name":..., "reason":..., "weight":...}]
    cleaned: list[dict] = []
    for x in items:
        code = str(x.get("code") or "").strip()
        if not code:
            continue
        try:
            w = float(x.get("weight"))
        except Exception:
            continue
        if w < 0:
            continue
        cleaned.append({**x, "weight": w})

    if not cleaned:
        return []

    # 先 clip 再归一
    for x in cleaned:
        if x["weight"] > max_weight:
            x["weight"] = max_weight

    s = sum(x["weight"] for x in cleaned)
    if s <= 0:
        return []

    for x in cleaned:
        x["weight"] = x["weight"] / s

    # 归一后可能再次略超 max_weight（因为其他被裁剪/丢弃），再做一次 soft clip
    for _ in range(2):
        over = [x for x in cleaned if x["weight"] > max_weight + 1e-9]
        if not over:
            break
        excess = 0.0
        for x in over:
            excess += x["weight"] - max_weight
            x["weight"] = max_weight
        under = [x for x in cleaned if x["weight"] < max_weight - 1e-9]
        if not under or excess <= 0:
            break
        under_sum = sum(x["weight"] for x in under)
        if under_sum <= 0:
            break
        for x in under:
            x["weight"] += excess * (x["weight"] / under_sum)

    # 最终再归一一次
    s2 = sum(x["weight"] for x in cleaned)
    if s2 > 0:
        for x in cleaned:
            x["weight"] = x["weight"] / s2

    return cleaned


def _node_weight(state: AgentState) -> AgentState:
    min_holdings = 10
    selected = state.get("selected") or []
    selected = selected[: max(min_holdings, 5)]

    max_weight = 0.25  # 稳健型上限

    llm = get_llm(
        state.get("model_tag", "primary"),
        provider_override=state.get("provider"),
        model_override=state.get("model_name"),
    )
    selected_enriched = enrich_selected_with_metrics(selected, as_of=state["as_of"], lookback_days=20)

    prompt = "\n\n".join(
        [
            SYSTEM_PROMPT,
            WEIGHTER_PROMPT.format(min_holdings=min_holdings),
            "候选股票（含近20日指标）：",
            "\n".join(
                (
                    lambda m: (
                        f"- {x.get('code')} {x.get('name','')}（{x.get('reason','')}）"
                        f" | 20d收益={m.get('total_return_20d','NA')}"
                        f" | 20d波动(年化)={m.get('volatility_20d','NA')}"
                        f" | 20d最大回撤={m.get('max_drawdown_20d','NA')}"
                        f" | 20d成交额均值={m.get('avg_amount_20d','NA')}"
                    )
                )(x.get('metrics') or {})
                for x in selected_enriched
            ),
        ]
    )

    raw = llm.invoke(prompt).text
    weights_raw = _safe_parse_json_list(raw)

    # 只接受在 selected 内的 code
    selected_map = {str(x.get("code")): x for x in selected if x.get("code")}
    merged: list[dict] = []
    for x in weights_raw:
        code = str(x.get("code") or "").strip()
        if not code or code not in selected_map:
            continue
        base = selected_map[code]
        merged.append({
            "code": code,
            "name": base.get("name"),
            "reason": base.get("reason"),
            "weight": x.get("weight"),
        })

    merged = _normalize_and_clip_weights(merged, max_weight=max_weight)

    if len(merged) < max(min_holdings, 5):
        # 回退等权
        n = len(selected) if selected else max(min_holdings, 5)
        w = 1.0 / n
        holdings = [
            Holding(code=str(x.get("code")), name=x.get("name"), weight=float(w))
            for x in (selected[:n])
        ]
        rationale = "\n\n".join(
            [
                "### 选股说明",
                state.get("rationale", "").strip(),
                "### 权重输出（回退等权）",
                "模型权重 JSON 输出不可用或数量不足，已回退等权。\n\n" + raw.strip(),
            ]
        ).strip()
        return {"holdings": holdings, "rationale": rationale}

    holdings = [
        Holding(
            code=str(x.get("code")),
            name=x.get("name"),
            weight=float(x.get("weight")),
        )
        for x in merged
    ]

    # 指标摘要（近20日）
    metrics_summary = "\n".join(
        (
            lambda m: (
                f"- {x.get('code')} {x.get('name','')} | 20d收益={m.get('total_return_20d','NA')} | 20d波动(年化)={m.get('volatility_20d','NA')} | 20d最大回撤={m.get('max_drawdown_20d','NA')} | 20d成交额均值={m.get('avg_amount_20d','NA')}"
            )
        )(x.get('metrics') or {})
        for x in selected_enriched
    )

    rationale = "\n\n".join(
        [
            "### 选股说明",
            state.get("rationale", "").strip(),
            "### 指标摘要（近20日）",
            metrics_summary,
            "### 权重 JSON 输出",
            raw.strip(),
            "### 风控摘要（系统约束）",
            f"- 稳健型：单票权重上限 {max_weight:.2f}\n- 权重已自动归一化与裁剪\n- 若模型输出不合法将自动回退等权",
        ]
    ).strip()

    return {"holdings": holdings, "rationale": rationale}


def _build_graph() -> Any:
    g = StateGraph(AgentState)
    g.add_node("fetch_data", _node_fetch_data)
    g.add_node("plan", _node_plan)
    g.add_node("analyze", _node_analyze)
    g.add_node("select", _node_select)
    g.add_node("weight", _node_weight)

    g.set_entry_point("fetch_data")
    g.add_edge("fetch_data", "plan")
    g.add_edge("plan", "analyze")
    g.add_edge("analyze", "select")
    g.add_edge("select", "weight")
    g.add_edge("weight", END)
    return g.compile()


_GRAPH = _build_graph()


def run_portfolio_agent(req: RunAgentRequest | None = None, /, **kwargs: Any) -> AgentRunResult:
    """运行一次决策链路。"""
    try:
        if req is None:
            req = RunAgentRequest(**kwargs)

        # 核心逻辑：周基金经理模式
        # 1. 确定本周交易区间 (周一开盘买入, 周五收盘结算)
        week_date = req.target_date or date.today().isoformat()
        week_start, week_end = week_trading_range(week_date)
        
        # 2. 确定分析截止日 (上周最后一个交易日)
        as_of = prev_week_last_trading_day(week_start)

        init: AgentState = {
            "as_of": as_of,
            "target_date": week_date,
            "week_start": week_start,
            "week_end": week_end,
            "top_k": req.top_k,
            "min_holdings": req.min_holdings,
            "model_tag": req.model_tag,
            "provider": req.provider,
            "model_name": req.model_name,
        }
        out: AgentState = _GRAPH.invoke(init)  # type: ignore[assignment]

        holdings = out.get("holdings") or []
        selected_raw = out.get("selected") or []
        selected: list[SelectedStock] = []
        for item in selected_raw:
            selected.append(
                SelectedStock(
                    code=str(item.get("code")),
                    name=item.get("name"),
                    reason=str(item.get("reason") or ""),
                )
            )

        steps: list[AgentStep] = []
        # 增加周度信息的 steps
        steps.append(AgentStep(name="周期对齐", output=f"本周交易区间：{week_start} 至 {week_end}\n决策数据截止(as_of)：{as_of}"))
        
        for k, title in [
            ("market_summary", "数据摘要"),
            ("plan", "规划"),
            ("analysis", "分析"),
            ("rationale", "选股说明"),
        ]:
            v = out.get(k)
            if isinstance(v, str) and v.strip():
                steps.append(AgentStep(name=title, output=v.strip()))

        return AgentRunResult(
            as_of=as_of,
            rationale=out.get("rationale", ""),
            holdings=holdings,
            steps=steps,
            selected=selected,
        )
    except Exception as e:
        import traceback
        print("="*50)
        print(f"AGENT ERROR: {str(e)}")
        traceback.print_exc()
        print("="*50)
        raise e

