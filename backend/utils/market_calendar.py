"""Utilities for working with the US equities trading calendar.

Provides helpers for determining whether a given date is a regular
NYSE/Nasdaq trading session and for enumerating valid trading days
between two endpoints. This prevents the intraday pipeline from
repeatedly trying to download data for market holidays.
"""
from __future__ import annotations

from functools import lru_cache
from datetime import date
from typing import Iterable, List

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    GoodFriday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay


# Juneteenth became a market holiday starting in 2022. The nearest-workday
# observance mirrors the NYSE treatment when the holiday falls on a weekend.
JuneteenthNYSE = Holiday(
    "Juneteenth",
    month=6,
    day=19,
    observance=nearest_workday,
    start_date="2022-01-01",
)


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    """NYSE observed full-day market closures."""

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        JuneteenthNYSE,
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


@lru_cache(maxsize=1)
def _trading_day_offset() -> CustomBusinessDay:
    """Return a cached CustomBusinessDay matching NYSE closures."""

    return CustomBusinessDay(calendar=_NYSEHolidayCalendar())


def trading_sessions_between(start: date, end: date) -> List[date]:
    """Return the list of NYSE trading sessions in the inclusive range.

    Parameters
    ----------
    start : date
        Beginning of the range (inclusive).
    end : date
        End of the range (inclusive).
    """

    if end < start:
        return []

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sessions = pd.date_range(start_ts, end_ts, freq=_trading_day_offset())
    return [ts.date() for ts in sessions]


def is_trading_session(day: date) -> bool:
    """Return True when the NYSE is scheduled to be open on the given day."""

    return day in trading_sessions_between(day, day)


def filter_trading_sessions(days: Iterable[date]) -> List[date]:
    """Filter an iterable of dates down to valid trading sessions."""

    return [day for day in days if is_trading_session(day)]
