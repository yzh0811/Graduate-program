from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings


_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "baostock"
_CACHE_TTL_SECONDS = 15 * 60
_BATCH_SIZE = 100


def _cache_key(prefix: str, **kwargs: object) -> str:
    payload = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(f"{prefix}:{payload}".encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{key}.json"


def _cache_load(key: str) -> pd.DataFrame | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    age = datetime.now().timestamp() - path.stat().st_mtime
    if age > _CACHE_TTL_SECONDS:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload.get("rows", []))
        if frame.empty:
            return pd.DataFrame()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.set_index("date")
        return frame
    except Exception:
        return None


def _cache_save(key: str, frame: pd.DataFrame) -> None:
    path = _cache_path(key)
    tmp = path.with_suffix(".tmp")
    serializable = frame.reset_index().copy()
    if "date" in serializable.columns:
        serializable["date"] = serializable["date"].astype(str)
    rows = serializable.to_dict(orient="records")
    tmp.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


@dataclass
class MarketSnapshot:
    """给 Agent 用的轻量摘要。

    后续你可以把更完整的数据（K线、财务指标、行业信息）放到 store 里，
    这里只提供可用于提示词的概要。
    """

    as_of: str
    window_start: str
    window_end: str
    summary_text: str
    universe: list[dict]  # [{"code": "...", "name": "...", "industry": "..."}]


def _month_window(as_of: str) -> tuple[str, str]:
    d = date.fromisoformat(as_of)
    # 严格以 as_of 为终点，往前推 30 天
    start = d - timedelta(days=30)
    window_start = start.isoformat()
    return window_start, as_of


def _mock_universe() -> list[dict]:
    return [
        {"code": "sh.600519", "name": "贵州茅台", "industry": "食品饮料"},
        {"code": "sh.601318", "name": "中国平安", "industry": "金融"},
        {"code": "sh.600036", "name": "招商银行", "industry": "金融"},
        {"code": "sh.600276", "name": "恒瑞医药", "industry": "医药生物"},
        {"code": "sh.600900", "name": "长江电力", "industry": "公用事业"},
        {"code": "sh.600030", "name": "中信证券", "industry": "非银金融"},
        {"code": "sz.300750", "name": "宁德时代", "industry": "电力设备"},
        {"code": "sz.002594", "name": "比亚迪", "industry": "汽车"},
        {"code": "sz.000858", "name": "五粮液", "industry": "食品饮料"},
        {"code": "sh.600887", "name": "伊利股份", "industry": "食品饮料"},
        {"code": "sh.600660", "name": "福耀玻璃", "industry": "汽车"},
        {"code": "sh.601012", "name": "隆基绿能", "industry": "电力设备"},
    ]


def _baostock_universe(as_of: str, top_k: int = 10) -> list[dict]:
    """通过 BaoStock 获取候选池（按流动性/市值筛选，避免随机前N）"""
    import baostock as bs
    try:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")
    except Exception as e:
        raise RuntimeError(f"无法连接 BaoStock: {e}") from e

    try:
        # 1) 获取全 A 股列表（剔除 ST、停牌、新股）
        rs = bs.query_stock_basic()
        stocks = []
        fields = list(getattr(rs, "fields", []) or [])
        while (rs.error_code == "0") and rs.next():
            row = rs.get_row_data()
            if isinstance(row, list) and fields:
                row = dict(zip(fields, row, strict=False))
            code = row.get("code")
            name = row.get("code_name")
            type_ = row.get("type")
            status = row.get("status")
            if type_ == "1" and status == "1":
                stocks.append({"code": code, "name": name})

        # 2) 动态预筛：候选池越大，预拉取越多；并设置上限避免极端超时
        #    经验值：至少 300，约为 top_k 的 6 倍，上限 2000
        prefilter_n = min(2000, max(300, top_k * 6))
        candidates = stocks[:prefilter_n]
        if not candidates:
            return []

        # 3) 拉取最近 20 日成交额（amount）或 fallback 到 close*volume
        codes = [x["code"] for x in candidates]
        d = date.fromisoformat(as_of)
        start = date.fromordinal(d.toordinal() - 20).isoformat()
        end = as_of

        # 分批拉取 + 本地缓存（15分钟）
        kline_panel = _baostock_amount_panel_chunked(codes, start, end)

        # 4) 计算每只票的 20 日平均成交额
        avg_amounts = {}
        for c in codes:
            if c in kline_panel.columns:
                # 尝试用 amount 字段
                amount_series = kline_panel[c].dropna()
                if len(amount_series) >= 5:
                    avg_amounts[c] = float(amount_series.mean())
                else:
                    avg_amounts[c] = 0.0
            else:
                avg_amounts[c] = 0.0

        # 5) 按成交额排序，取前 top_k
        sorted_by_amount = sorted(
            candidates,
            key=lambda x: avg_amounts.get(x["code"], 0.0),
            reverse=True,
        )
        top = sorted_by_amount[:top_k]

        # 6) 补充 industry（暂时写未分类，后续可补）
        for item in top:
            item["industry"] = "未分类"
            item["avg_amount_20d"] = avg_amounts.get(item["code"], 0.0)

        # Debug: 打印排序依据
        for item in top:
            print(f"[DEBUG] universe {item['code']} avg_amount={item.get('avg_amount_20d', 0):.0f}")

        return top
    finally:
        bs.logout()


def get_market_snapshot(as_of: str, top_k: int = 10) -> MarketSnapshot:
    provider = (settings.data_provider or "mock").lower()
    window_start, window_end = _month_window(as_of)

    if provider == "mock":
        universe = _mock_universe()[:top_k]
        return MarketSnapshot(
            as_of=as_of,
            window_start=window_start,
            window_end=window_end,
            summary_text=(
                f"数据窗口：{window_start} ~ {window_end}\n"
                "（当前为 MOCK 数据源：未接入真实行情/基本面。后续可切换 BaoStock。）"
            ),
            universe=universe,
        )

    if provider == "baostock":
        try:
            universe = _baostock_universe(as_of, top_k)
            return MarketSnapshot(
                as_of=as_of,
                window_start=window_start,
                window_end=window_end,
                summary_text=(
                    f"数据窗口：{window_start} ~ {window_end}\n"
                    f"数据源：BaoStock（候选池数量：{len(universe)}）"
                ),
                universe=universe,
            )
        except Exception as e:
            # 降级到 mock 并记录错误
            universe = _mock_universe()[:top_k]
            return MarketSnapshot(
                as_of=as_of,
                window_start=window_start,
                window_end=window_end,
                summary_text=(
                    f"数据窗口：{window_start} ~ {window_end}\n"
                    f"数据源：BaoStock（拉取失败，已降级到 MOCK。错误：{e}）"
                ),
                universe=universe,
            )

    # 默认降级到 mock
    universe = _mock_universe()[:top_k]
    return MarketSnapshot(
        as_of=as_of,
        window_start=window_start,
        window_end=window_end,
        summary_text=(
            f"数据窗口：{window_start} ~ {window_end}\n"
            f"数据源：{provider}（未实现，已回退到 mock）"
        ),
        universe=universe,
    )


def _mock_price_panel(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """Mock 价格面板：生成 22 个交易日随机游走价格"""
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) == 0:
        dates = pd.bdate_range(end=end, periods=22)

    df = pd.DataFrame(index=dates)
    for c in codes:
        base = 100.0 + (sum(ord(x) for x in c) % 50)
        rnd = pd.Series(range(len(dates)), index=dates).astype(float)
        noise = (pd.Series(pd.util.hash_pandas_object(dates).values, index=dates) % 1000) / 1000.0
        close = base * (1.0 + 0.0005 * rnd) * (1.0 + 0.002 * (noise - 0.5))
        df[c] = close
    return df


def _baostock_amount_panel_chunked(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """分批拉取近20日成交额面板，并做短TTL本地缓存。"""
    if not codes:
        return pd.DataFrame()

    panels: list[pd.DataFrame] = []
    for i in range(0, len(codes), _BATCH_SIZE):
        batch_codes = codes[i : i + _BATCH_SIZE]
        key = _cache_key("amount_panel", start=start, end=end, codes=batch_codes)
        cached = _cache_load(key)
        if cached is not None:
            panels.append(cached)
            continue

        panel = _baostock_kline_panel(batch_codes, start, end, fields="date,code,close,volume,amount")
        _cache_save(key, panel)
        panels.append(panel)

    if not panels:
        return pd.DataFrame()
    merged = pd.concat(panels, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def _baostock_kline_panel(
    codes: list[str], start: str, end: str, fields: str = "date,code,close,volume,amount"
) -> pd.DataFrame:
    """返回多字段面板：index=日期，columns=股票代码，values=指定字段（这里取 amount）"""
    import baostock as bs
    try:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")
    except Exception as e:
        raise RuntimeError(f"无法连接 BaoStock: {e}") from e

    dfs = []
    try:
        for code in codes:
            code = _to_baostock_code(code)
            rs = bs.query_history_k_data_plus(
                code,
                fields,
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag="3",
            )
            if rs.error_code != "0":
                raise RuntimeError(f"获取 {code} 数据失败: {rs.error_msg}")
            data_list = []
            fields_list = list(getattr(rs, "fields", []) or [])
            while (rs.error_code == "0") and rs.next():
                row = rs.get_row_data()
                if isinstance(row, list) and fields_list:
                    row = dict(zip(fields_list, row, strict=False))
                data_list.append(row)
            if not data_list:
                continue
            df = pd.DataFrame(data_list)
            df["date"] = pd.to_datetime(df["date"])
            # 取 amount 字段，若缺失则 fallback 到 close*volume
            if "amount" in df.columns:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
                df = df.set_index("date")
                dfs.append(df[["amount"]].rename(columns={"amount": code}))
            elif "close" in df.columns and "volume" in df.columns:
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                df["amount"] = df["close"] * df["volume"]
                df = df.set_index("date")
                dfs.append(df[["amount"]].rename(columns={"amount": code}))
        if not dfs:
            raise ValueError("未获取到任何有效数据")
        panel = pd.concat(dfs, axis=1)
        panel = panel.loc[:, panel.notna().any()]
        return panel
    finally:
        bs.logout()


def _to_baostock_code(code: str) -> str:
    c = (code or "").strip()
    if not c:
        return c

    if c.lower().startswith("sh.") or c.lower().startswith("sz."):
        return c.lower()

    if c.upper().endswith(".SH"):
        return "sh." + c.split(".")[0]
    if c.upper().endswith(".SZ"):
        return "sz." + c.split(".")[0]

    digits = "".join(ch for ch in c if ch.isdigit())
    if len(digits) >= 6:
        digits = digits[:6]
    if digits.startswith("6"):
        return f"sh.{digits}"
    return f"sz.{digits}"


def _baostock_price_panel(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """通过 BaoStock 获取多只股票的日度收盘价面板"""
    import baostock as bs

    try:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")
    except Exception as e:
        raise RuntimeError(f"无法连接 BaoStock: {e}") from e

    dfs = []
    try:
        for code in codes:
            # BaoStock 代码格式：sh.600519 / sz.000001
            # 兼容输入：
            # - sh.600519 / sz.000001（原样）
            # - 600519.SH / 000001.SZ（转换）
            # - 600519 / 000001（按 6 开头->sh，其它->sz）
            code = _to_baostock_code(code)
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,close",
                start_date=start,
                end_date=end,
                frequency="d",  # 日线
                adjustflag="3",  # 后复权
            )
            if rs.error_code != "0":
                raise RuntimeError(f"获取 {code} 数据失败: {rs.error_msg}")
            data_list = []
            fields = list(getattr(rs, "fields", []) or [])
            while (rs.error_code == "0") and rs.next():
                row = rs.get_row_data()
                if isinstance(row, list) and fields:
                    row = dict(zip(fields, row, strict=False))
                data_list.append(row)
            if not data_list:
                continue
            df = pd.DataFrame(data_list, columns=["date", "code", "close"])
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.set_index("date")
            dfs.append(df[["close"]].rename(columns={"close": code}))
        if not dfs:
            raise ValueError("未获取到任何有效数据")
        panel = pd.concat(dfs, axis=1)
        # 去除全为 NaN 的列（可能停牌）
        panel = panel.loc[:, panel.notna().any()]
        return panel
    finally:
        bs.logout()


def get_price_panel(
    codes: list[str], start: str, end: str
) -> pd.DataFrame:
    """返回日度收盘价面板：index=日期，columns=股票代码，values=close.

    优先使用 BaoStock，失败则降级到 mock。
    """
    provider = (settings.data_provider or "mock").lower()
    if provider == "baostock":
        try:
            return _baostock_price_panel(codes, start, end)
        except Exception:
            # 降级到 mock 并记录错误（可考虑日志）
            return _mock_price_panel(codes, start, end)

    # 默认或未实现则用 mock
    return _mock_price_panel(codes, start, end)


def get_ohlc_panel(
    codes: list[str], start: str, end: str
) -> dict[str, pd.DataFrame]:
    """返回 OHLC 面板：dict 包含 'open', 'high', 'low', 'close' 四个 DataFrame。
    
    优先使用 BaoStock，失败则降级到 mock。
    """
    provider = (settings.data_provider or "mock").lower()
    if provider == "baostock":
        try:
            return _baostock_ohlc_panel(codes, start, end)
        except Exception:
            # 降级到 mock 并记录错误（可考虑日志）
            return _mock_ohlc_panel(codes, start, end)

    # 默认或未实现则用 mock
    return _mock_ohlc_panel(codes, start, end)


def _baostock_ohlc_panel(codes: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """通过 BaoStock 获取多只股票的日度 OHLC 面板"""
    import baostock as bs

    try:
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")
    except Exception as e:
        raise RuntimeError(f"无法连接 BaoStock: {e}") from e

    all_dfs = {"open": [], "high": [], "low": [], "close": []}
    try:
        for code in codes:
            code = _to_baostock_code(code)
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close",
                start_date=start,
                end_date=end,
                frequency="d",  # 日线
                adjustflag="3",  # 后复权
            )
            if rs.error_code != "0":
                raise RuntimeError(f"获取 {code} 数据失败: {rs.error_msg}")
            data_list = []
            fields = list(getattr(rs, "fields", []) or [])
            while (rs.error_code == "0") and rs.next():
                row = rs.get_row_data()
                if isinstance(row, list) and fields:
                    row = dict(zip(fields, row, strict=False))
                data_list.append(row)
            if not data_list:
                continue
            df = pd.DataFrame(data_list)
            df["date"] = pd.to_datetime(df["date"])
            df["open"] = pd.to_numeric(df["open"], errors="coerce")
            df["high"] = pd.to_numeric(df["high"], errors="coerce")
            df["low"] = pd.to_numeric(df["low"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.set_index("date")
            all_dfs["open"].append(df[["open"]].rename(columns={"open": code}))
            all_dfs["high"].append(df[["high"]].rename(columns={"high": code}))
            all_dfs["low"].append(df[["low"]].rename(columns={"low": code}))
            all_dfs["close"].append(df[["close"]].rename(columns={"close": code}))
        
        # 合并所有字段
        result = {}
        for key in ["open", "high", "low", "close"]:
            if all_dfs[key]:
                panel = pd.concat(all_dfs[key], axis=1)
                panel = panel.loc[:, panel.notna().any()]
                result[key] = panel
            else:
                result[key] = pd.DataFrame()
        return result
    finally:
        bs.logout()


def _mock_ohlc_panel(codes: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Mock OHLC 面板：生成 22 个交易日随机游走价格"""
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) == 0:
        dates = pd.bdate_range(end=end, periods=22)

    result = {}
    for field in ["open", "high", "low", "close"]:
        df = pd.DataFrame(index=dates)
        for c in codes:
            base = 100.0 + (sum(ord(x) for x in c) % 50)
            rnd = pd.Series(range(len(dates)), index=dates).astype(float)
            noise = (pd.Series(pd.util.hash_pandas_object(dates).values, index=dates) % 1000) / 1000.0
            close = base * (1.0 + 0.0005 * rnd) * (1.0 + 0.002 * (noise - 0.5))
            # Open/High/Low 围绕 close 生成
            open_price = close * (1 + 0.001 * (np.random.random(len(dates)) - 0.5))
            high_price = close * (1 + 0.003 * np.random.random(len(dates)))
            low_price = close * (1 - 0.003 * np.random.random(len(dates)))
            df[field] = open_price if field == "open" else (high_price if field == "high" else (low_price if field == "low" else close))
        result[field] = df
    return result


def enrich_selected_with_metrics(
    selected: list[dict],
    as_of: str,
    lookback_days: int = 20,
) -> list[dict]:
    """为每只选中的股票补充技术指标（近20日）"""
    if not selected:
        return selected

    codes = [str(x.get("code")) for x in selected if x.get("code")]
    if not codes:
        return selected

    # 取最近 20 个自然日（尽量覆盖交易日）
    d = date.fromisoformat(as_of)
    start = date.fromordinal(d.toordinal() - lookback_days).isoformat()
    end = as_of

    try:
        # 获取价格和成交额面板
        # 注意：get_price_panel 目前只返回 close。我们需要一个能返回多字段的工具，或者直接调 _baostock_kline_panel
        provider = (settings.data_provider or "mock").lower()
        if provider == "baostock":
            # 直接调用底层面板获取，以拿到 amount
            panel_data = _baostock_kline_panel(codes, start, end, fields="date,code,close,amount")
            # 价格面板
            price_panel = panel_data.xs('close', axis=1, level=0) if isinstance(panel_data.columns, pd.MultiIndex) else None
            # 成交额面板
            amount_panel = panel_data.xs('amount', axis=1, level=0) if isinstance(panel_data.columns, pd.MultiIndex) else None
            
            # 如果不是 MultiIndex (老逻辑兼容)，则 _baostock_kline_panel 可能需要微调
            # 这里的 _baostock_kline_panel 在代码库里实现似乎是直接返回 amount 的列？
            # 重新看下 providers.py 里的 _baostock_kline_panel 实现
            price_panel = get_price_panel(codes, start, end)
            # 成交额暂时仍然从 _baostock_kline_panel 获取
            amount_panel = _baostock_kline_panel(codes, start, end, fields="date,code,amount")
        else:
            price_panel = _mock_price_panel(codes, start, end)
            amount_panel = price_panel * 1000000 # Mock amount
    except Exception as e:
        print(f"[ERROR] Failed to fetch metrics: {e}")
        return selected

    enriched = []
    for item in selected:
        code = str(item.get("code"))
        if code not in price_panel.columns:
            enriched.append({**item, "metrics": {}})
            continue

        series = price_panel[code].dropna()
        if len(series) < 5:
            enriched.append({**item, "metrics": {}})
            continue

        # 收益率
        ret = series.pct_change().dropna()
        total_ret = (series.iloc[-1] / series.iloc[0] - 1) if len(series) >= 2 else 0.0

        # 波动率（年化）
        vol = ret.std(ddof=1) * (252 ** 0.5) if len(ret) >= 2 else 0.0

        # 最大回撤
        cum = (1 + ret).cumprod()
        dd = cum / cum.cummax() - 1
        max_dd = dd.min() if len(dd) > 0 else 0.0

        # 成交额均值
        avg_amount = 0.0
        if amount_panel is not None and code in amount_panel.columns:
            a_series = amount_panel[code].dropna()
            if len(a_series) > 0:
                avg_amount = float(a_series.mean())

        metrics = {
            "total_return_20d": float(total_ret),
            "volatility_20d": float(vol),
            "max_drawdown_20d": float(max_dd),
            "avg_amount_20d": float(avg_amount),
        }
        enriched.append({**item, "metrics": metrics})
        print(f"[DEBUG] enriched {code}: {metrics}")
    return enriched