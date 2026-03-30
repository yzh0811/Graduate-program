from __future__ import annotations

from datetime import date, timedelta
from calendar import monthrange

from app.core.config import settings

_TRADING_DAY_CACHE: dict[str, bool] = {}


def _to_date(x: str | date) -> date:
    if isinstance(x, date):
        return x
    return date.fromisoformat(x)


def is_trading_day(d: str | date) -> bool:
    """Return True if d is a trading day (BaoStock), else assume weekday in mock mode."""
    provider = (settings.data_provider or "mock").lower()
    dd = _to_date(d)

    if provider != "baostock":
        return dd.weekday() < 5

    ds = dd.isoformat()
    if ds in _TRADING_DAY_CACHE:
        return _TRADING_DAY_CACHE[ds]

    import baostock as bs

    # 拉取整月交易日，避免逐日 login/query/logout 导致极慢
    month_start = dd.replace(day=1)
    month_end = dd.replace(day=monthrange(dd.year, dd.month)[1])

    try:
        lg = bs.login()
        if lg.error_code != "0":
            v = dd.weekday() < 5
            _TRADING_DAY_CACHE[ds] = v
            return v

        rs = bs.query_trade_dates(start_date=month_start.isoformat(), end_date=month_end.isoformat())
        if rs.error_code != "0":
            v = dd.weekday() < 5
            _TRADING_DAY_CACHE[ds] = v
            return v

        while rs.next():
            row = rs.get_row_data()
            if not row or len(row) < 2:
                continue
            day = str(row[0])
            flag = str(row[1]) == "1"
            _TRADING_DAY_CACHE[day] = flag

        if ds in _TRADING_DAY_CACHE:
            return _TRADING_DAY_CACHE[ds]

        v = dd.weekday() < 5
        _TRADING_DAY_CACHE[ds] = v
        return v
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def latest_trading_day_on_or_before(d: str | date, max_lookback_days: int = 14) -> str:
    dd = _to_date(d)
    for i in range(max_lookback_days + 1):
        x = dd - timedelta(days=i)
        if is_trading_day(x):
            return x.isoformat()
    return dd.isoformat()


def week_trading_range(d: str | date) -> tuple[str, str]:
    """Given any date, return (week_start_trading_day, week_end_trading_day).

    Week is Monday..Sunday by calendar, but start/end are adjusted to nearest trading days within that week.
    """
    dd = _to_date(d)
    monday = dd - timedelta(days=dd.weekday())
    sunday = monday + timedelta(days=6)

    # Find first trading day between monday..sunday
    start = None
    for i in range(7):
        x = monday + timedelta(days=i)
        if is_trading_day(x):
            start = x
            break
    if start is None:
        # fallback: use latest trading day on or before monday
        start = _to_date(latest_trading_day_on_or_before(monday))

    # Find last trading day between monday..sunday
    end = None
    for i in range(7):
        x = sunday - timedelta(days=i)
        if is_trading_day(x):
            end = x
            break
    if end is None:
        end = start

    return start.isoformat(), end.isoformat()


def prev_week_last_trading_day(d: str | date) -> str:
    """Return last trading day of the previous week relative to date d."""
    dd = _to_date(d)
    monday = dd - timedelta(days=dd.weekday())
    prev_sunday = monday - timedelta(days=1)
    # last trading day on or before prev_sunday
    return latest_trading_day_on_or_before(prev_sunday)
