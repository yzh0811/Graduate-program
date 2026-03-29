import json
from dataclasses import dataclass
import pandas as pd


@dataclass
class FundConfig:
    code: str
    name: str
    nav_col: str
    daily_change_col: str


def _to_float_percent(series: pd.Series) -> pd.Series:
    # 支持 "12.3%" / "12.3" / 空值
    out = pd.to_numeric(series.astype(str).str.replace("%", "", regex=False), errors="coerce")

    # 自动识别“已是小数收益率”的场景（例如 0.0123 代表 1.23%）
    # 若绝大多数非空值绝对值都很小，则视为小数并转成百分比口径
    nz = out.dropna()
    if not nz.empty:
        q95 = float(nz.abs().quantile(0.95))
        if q95 <= 0.2:  # 经验阈值：20%以内通常是小数口径
            out = out * 100.0

    return out


def _max_drawdown_from_returns(daily_returns_pct: pd.Series) -> float:
    # daily_returns_pct: 单位为百分比（例如 1.2 表示 1.2%）
    r = 1 + daily_returns_pct.fillna(0) / 100.0
    nav = r.cumprod()
    peak = nav.cummax()
    drawdown = nav / peak - 1
    return float(drawdown.min() * 100)


def _weekly_return_from_nav(nav_series: pd.Series) -> float:
    nav_series = nav_series.dropna()
    if nav_series.empty:
        return 0.0
    start = nav_series.iloc[0]
    end = nav_series.iloc[-1]
    if start == 0:
        return 0.0
    return float((end / start - 1) * 100)


def _weekly_return_from_daily_returns(daily_returns_pct: pd.Series) -> float:
    r = 1 + daily_returns_pct.fillna(0) / 100.0
    return float((r.prod() - 1) * 100)


def _stock_key_from_col(col: str) -> str:
    s = str(col)
    if "模型组合" in s and "(" in s:
        # 例如：模型组合-华胜天成(涨跌幅)
        left = s.split("(", 1)[0]
        if "-" in left:
            return left.split("-", 1)[1]
    return s


def _detect_model_weights(df: pd.DataFrame, model_stock_cols: list[str]) -> tuple[dict[str, float], str]:
    """尝试从Excel中识别模型权重；识别失败则回退等权。"""
    weight_cols = [c for c in df.columns if "模型组合" in c and "权重" in c]
    if not weight_cols:
        n = len(model_stock_cols)
        return {c: 1.0 / n for c in model_stock_cols}, "equal"

    # 建立“股票名 -> 权重列”映射
    w_map_by_key: dict[str, str] = {}
    for wc in weight_cols:
        w_map_by_key[_stock_key_from_col(wc)] = wc

    raw_weights: dict[str, float] = {}
    for rc in model_stock_cols:
        key = _stock_key_from_col(rc)
        wc = w_map_by_key.get(key)
        if not wc:
            continue
        wraw = df[wc].astype(str).str.replace("%", "", regex=False)
        ws = pd.to_numeric(wraw, errors="coerce").dropna()
        if ws.empty:
            continue
        # 取第一个非空值作为该周建仓权重（更符合周初建仓、持有一周）
        raw_weights[rc] = float(ws.iloc[0])

    # 若匹配不全或权重无效，回退等权
    if len(raw_weights) < len(model_stock_cols):
        n = len(model_stock_cols)
        return {c: 1.0 / n for c in model_stock_cols}, "equal"

    vals = [max(v, 0.0) for v in raw_weights.values()]
    total = sum(vals)
    if total <= 0:
        n = len(model_stock_cols)
        return {c: 1.0 / n for c in model_stock_cols}, "equal"

    # 兼容两种权重写法：
    # - 小数权重：0.1 / 0.2
    # - 百分数权重：10 / 20（或带 %，前面已去掉）
    if total > 1.5:
        vals = [v / 100.0 for v in vals]

    total2 = sum(vals)
    if total2 <= 0:
        n = len(model_stock_cols)
        return {c: 1.0 / n for c in model_stock_cols}, "equal"

    normed = {k: vals[i] / total2 for i, k in enumerate(raw_weights.keys())}
    return normed, "weighted"


def build_weekly_report(
    excel_path: str,
    output_excel: str = "周度对比结果_完整版.xlsx",
) -> tuple[dict, str]:
    df = pd.read_excel(excel_path)

    def _normalize_col(c: str) -> str:
        s = str(c)
        s = s.replace("\n", "").replace("\r", "")
        s = s.replace("（", "(").replace("）", ")")
        s = s.replace("－", "-").replace("—", "-").replace("–", "-")
        s = s.replace(" ", "").strip()
        return s

    # 规范化列名（去掉换行/空格，统一中英文括号与横线）
    original_cols = list(df.columns)
    normalized_cols = [_normalize_col(c) for c in original_cols]
    df.columns = normalized_cols

    # 模型组合列（兼容不同写法）
    model_stock_cols = [
        col for col in df.columns if "模型组合" in col and "涨跌幅" in col
    ]
    if not model_stock_cols:
        raise ValueError("未找到模型组合涨跌幅列（列名需包含 '模型组合' 和 '涨跌幅'）")

    for col in model_stock_cols:
        df[col] = _to_float_percent(df[col])

    # 优先按权重列计算；若Excel未提供权重列则自动等权
    weights_map, weight_mode = _detect_model_weights(df, model_stock_cols)
    df["模型组合-日收益"] = df[model_stock_cols].apply(
        lambda x: sum((x[c] if pd.notna(x[c]) else 0.0) * weights_map.get(c, 0.0) for c in model_stock_cols),
        axis=1,
    )

    # 诊断明细：每只股票的日贡献 = 当日涨跌幅 * 权重
    diag_rows = []
    for _, row in df.iterrows():
        item = {"日期": row.get(_normalize_col("日期"), "")}
        total = 0.0
        for c in model_stock_cols:
            w = float(weights_map.get(c, 0.0))
            r = float(row[c]) if pd.notna(row[c]) else 0.0
            contrib = r * w
            total += contrib
            item[f"收益:{c}"] = r
            item[f"权重:{c}"] = w
            item[f"贡献:{c}"] = contrib
        item["组合日收益(诊断)"] = total
        diag_rows.append(item)
    diag_df = pd.DataFrame(diag_rows)

    # 周收益/回撤
    model_weekly_return = _weekly_return_from_daily_returns(df["模型组合-日收益"])
    model_max_drawdown = _max_drawdown_from_returns(df["模型组合-日收益"])

    # 基金配置（你可以在这里增删）
    funds: list[FundConfig] = [
        FundConfig(
            code="001445",
            name="华安国企改革混合A",
            nav_col="华安国企改革(001445)-净值",
            daily_change_col="华安国企改革-日涨幅",
        ),
        FundConfig(
            code="021642",
            name="富国资源精选混合A",
            nav_col="富国资源精选(021642)-净值",
            daily_change_col="富国资源精选-日涨幅",
        ),
        FundConfig(
            code="001382",
            name="易方达国企改革混合",
            nav_col="易方达国企改革(001382)-净值",
            daily_change_col="易方达国企改革-日涨幅",
        ),
        FundConfig(
            code="166301",
            name="华商新趋势优选混合",
            nav_col="华商新趋势优选(166301)-净值",
            daily_change_col="华商新趋势优选-日涨幅",
        ),
    ]

    fund_results = []
    for f in funds:
        nav_key = _normalize_col(f.nav_col)
        daily_key = _normalize_col(f.daily_change_col)
        nav = df.get(nav_key)
        daily = df.get(daily_key)
        if nav is None or daily is None:
            # 容错：允许列名包含基金名称+净值/涨幅（对名称也做规范化）
            name_key = _normalize_col(f.name)
            nav_candidates = [c for c in df.columns if name_key in c and "净值" in c]
            daily_candidates = [c for c in df.columns if name_key in c and "涨" in c]
            nav = df[nav_candidates[0]] if nav_candidates else None
            daily = df[daily_candidates[0]] if daily_candidates else None
        if nav is None or daily is None:
            raise ValueError(f"缺少基金列：{f.nav_col} 或 {f.daily_change_col}")
        daily_pct = _to_float_percent(daily)
        weekly_return = _weekly_return_from_nav(nav)
        max_dd = _max_drawdown_from_returns(daily_pct)
        fund_results.append(
            {
                "fund_code": f.code,
                "fund_name": f.name,
                "weekly_return": round(weekly_return, 2),
                "max_drawdown": round(max_dd, 2),
            }
        )

    # 沪深300
    hs300_col = _normalize_col("沪深300-涨跌幅")
    if hs300_col not in df.columns:
        # 容错：尝试匹配包含“沪深300”和“涨跌幅”的列
        candidates = [c for c in df.columns if "沪深300" in c and "涨跌幅" in c]
        if not candidates:
            raise ValueError(f"缺少基准列：沪深300-涨跌幅")
        hs300_col = candidates[0]
    df[hs300_col] = _to_float_percent(df[hs300_col])
    hs300_weekly_return = _weekly_return_from_daily_returns(df[hs300_col])
    hs300_max_drawdown = _max_drawdown_from_returns(df[hs300_col])

    # 结构化 JSON
    result_json = {
        "comparison_cycle": f"{df.iloc[0][_normalize_col('日期')]}至{df.iloc[-1][_normalize_col('日期')]}",
        "model_portfolio": {
            "weight_mode": weight_mode,
            "total_weekly_return": round(model_weekly_return, 2),
            "max_drawdown": round(model_max_drawdown, 2),
        },
        "fund_portfolio": fund_results,
        "benchmark": {
            "hs300_weekly_return": round(hs300_weekly_return, 2),
            "hs300_max_drawdown": round(hs300_max_drawdown, 2),
        },
    }

    # Markdown 报告
    def _win_flag(x: float) -> str:
        return "是" if x > hs300_weekly_return else "否"

    md_lines = []
    md_lines.append(f"# 周度对比报告（{df.iloc[0][_normalize_col('日期')]}至{df.iloc[-1][_normalize_col('日期')]}）")
    md_lines.append("## 一、核心对比数据")
    md_lines.append(f"- 模型权重模式：{'按Excel权重列计算' if weight_mode == 'weighted' else '等权计算（未识别到完整权重列）'}")
    md_lines.append("| 对比对象 | 周收益率（%） | 周最大回撤（%） | 跑赢沪深300（是/否） |")
    md_lines.append("|---|---:|---:|---:|")
    md_lines.append(
        f"| 模型选股组合 | {model_weekly_return:.2f} | {model_max_drawdown:.2f} | {_win_flag(model_weekly_return)} |"
    )
    for fr in fund_results:
        md_lines.append(
            f"| {fr['fund_name']}({fr['fund_code']}) | {fr['weekly_return']:.2f} | {fr['max_drawdown']:.2f} | {_win_flag(fr['weekly_return'])} |"
        )
    md_lines.append(
        f"| 沪深300 | {hs300_weekly_return:.2f} | {hs300_max_drawdown:.2f} | - |"
    )

    # 结论
    all_returns = [("模型选股组合", model_weekly_return)] + [
        (f["fund_name"], f["weekly_return"]) for f in fund_results
    ]
    best_return = max(all_returns, key=lambda x: x[1])

    all_dd = [("模型选股组合", model_max_drawdown)] + [
        (f["fund_name"], f["max_drawdown"]) for f in fund_results
    ]
    best_dd = max(all_dd, key=lambda x: x[1])  # 回撤越接近0越好

    beat_count = sum(
        1 for _, v in all_returns if v > hs300_weekly_return
    )

    md_lines.append("\n## 二、核心结论")
    md_lines.append(
        f"1. 收益排名：{best_return[0]} 周收益率最高（{best_return[1]:.2f}%）；"
    )
    md_lines.append(
        f"2. 风险控制：{best_dd[0]} 周最大回撤最小（{best_dd[1]:.2f}%）；"
    )
    md_lines.append(
        f"3. 基准对比：共 {beat_count} 个组合跑赢沪深300，整体表现{'优于' if beat_count >= 3 else '弱于'}市场平均。"
    )

    md_lines.append("\n## 三、风险提示")
    md_lines.append(
        "模型组合采用等权持仓，行业集中度可能高于公募基金，短期若核心板块回调，收益回撤风险可能高于基金组合。"
    )

    markdown_report = "\n".join(md_lines)

    # 写入结果 Excel（原始+诊断两张表）
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="原始与计算", index=False)
        diag_df.to_excel(writer, sheet_name="模型收益诊断", index=False)

    return result_json, markdown_report


if __name__ == "__main__":
    result, report = build_weekly_report("周度对比原始数据.xlsx")
    print("=== 结构化JSON报告 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n=== Markdown可视化报告 ===")
    print(report)
    print("\n对比结果已写入：周度对比结果_完整版.xlsx")
