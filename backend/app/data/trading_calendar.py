from __future__ import annotations

from datetime import date, timedelta

from app.core.config import settings


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

    import baostock as bs

    try:
        lg = bs.login()
        if lg.error_code != "0":
            return dd.weekday() < 5

        ds = dd.isoformat()
        rs = bs.query_trade_dates(start_date=ds, end_date=ds)
        if rs.error_code != "0":
            return dd.weekday() < 5
        if not rs.next():
            return dd.weekday() < 5
        row = rs.get_row_data()
        # row: [calendar_date, is_trading_day]
        return str(row[1]) == "1"
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
