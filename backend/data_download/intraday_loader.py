"""Intraday (15-minute) data management for SAC + DSR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import pytz
import yfinance as yf

from database.db_manager import DatabaseManager

INTRADAY_ROOT = Path("data/intraday")
NY_TZ = pytz.timezone("America/New_York")
SUPPORTED_INTERVALS = {"15m"}


@dataclass
class IntradayStoreConfig:
    symbol: str
    interval: str = "15m"
    root: Path = INTRADAY_ROOT

    def __post_init__(self) -> None:
        if self.interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval {self.interval}. Supported: {SUPPORTED_INTERVALS}")

    @property
    def base_dir(self) -> Path:
        path = self.root / self.symbol.upper() / self.interval
        path.mkdir(parents=True, exist_ok=True)
        return path


def _parse_date(value: Optional[str | date | datetime]) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return pd.Timestamp(value).date()


def _session_path(config: IntradayStoreConfig, session_date: date) -> Path:
    return config.base_dir / f"{session_date.isoformat()}.csv"


def list_cached_sessions(config: IntradayStoreConfig) -> List[date]:
    sessions = []
    for file in sorted(config.base_dir.glob("*.csv")):
        try:
            sessions.append(date.fromisoformat(file.stem))
        except ValueError:
            continue
    return sessions


def _download_intraday(symbol: str, session_start: datetime, session_end: datetime, interval: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        start=session_start.astimezone(pytz.UTC),
        end=(session_end + timedelta(minutes=1)).astimezone(pytz.UTC),
        interval=interval,
        auto_adjust=False,
        progress=False,
        actions=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data.empty:
        return data

    data.index = pd.DatetimeIndex(data.index).tz_localize("UTC").tz_convert(NY_TZ)
    data = data.rename(columns=str.title)
    data = data.loc[(data.index >= session_start) & (data.index <= session_end)]

    return data


def _save_session(config: IntradayStoreConfig, session_date: date, df: pd.DataFrame) -> Path:
    path = _session_path(config, session_date)
    df.to_csv(path)
    return path


def download_and_cache_sessions(
    symbol: str,
    start: date,
    end: date,
    interval: str = "15m"
) -> List[Path]:
    config = IntradayStoreConfig(symbol=symbol, interval=interval)
    saved_paths: List[Path] = []

    cursor = start
    while cursor <= end:
        path = _session_path(config, cursor)
        if path.exists():
            cursor += timedelta(days=1)
            continue

        session_start = NY_TZ.localize(datetime.combine(cursor, datetime.min.time()) + timedelta(hours=9, minutes=30))
        session_end = NY_TZ.localize(datetime.combine(cursor, datetime.min.time()) + timedelta(hours=15, minutes=45))

        df = _download_intraday(symbol, session_start, session_end, interval)
        if not df.empty:
            _save_session(config, cursor, df)
            saved_paths.append(path)
        cursor += timedelta(days=1)

    return saved_paths


METADATA_COLUMNS = {'session_date', 'bar_index', 'time_fraction', 'minutes_from_open', 'is_session_end'}


def _load_sessions(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(paths):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize(NY_TZ)
        else:
            idx = idx.tz_convert(NY_TZ)
        df.index = idx
        df['session_date'] = df.index.date
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames).sort_index()
    return data


def load_intraday_data(
    symbol: str,
    interval: str = "15m",
    start: Optional[str | date | datetime] = None,
    end: Optional[str | date | datetime] = None,
    force_download: bool = False
) -> pd.DataFrame:
    config = IntradayStoreConfig(symbol=symbol, interval=interval)
    db_manager = DatabaseManager()

    cached_sessions = set(list_cached_sessions(config))
    start_date = _parse_date(start) or (min(cached_sessions) if cached_sessions else None)
    end_date = _parse_date(end) or (max(cached_sessions) if cached_sessions else None)

    if start_date is None or end_date is None:
        today = date.today()
        end_date = today if end_date is None else end_date
        start_date = (end_date - timedelta(days=30)) if start_date is None else start_date

    if start_date > end_date:
        raise ValueError("start_date must be earlier than end_date")

    data_from_sql = False
    if not force_download:
        start_ts = pd.Timestamp(start_date).tz_localize(NY_TZ).isoformat()
        end_ts = (pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)).tz_localize(NY_TZ).isoformat()
        sql_df = db_manager.get_intraday_data(symbol, interval=interval, start=start_ts, end=end_ts)
        if not sql_df.empty:
            data = sql_df.copy()
            data_from_sql = True

    if not data_from_sql and (force_download or not cached_sessions.issuperset({d for d in _daterange(start_date, end_date)})):
        download_and_cache_sessions(symbol, start_date, end_date, interval=interval)

    if not data_from_sql:
        target_paths = [
            _session_path(config, session_day)
            for session_day in _daterange(start_date, end_date)
            if _session_path(config, session_day).exists()
        ]

        data = _load_sessions(target_paths)
        if data.empty:
            raise RuntimeError("No intraday data available after download.")

        _augment_intraday_metadata(data)
        db_manager.save_intraday_data(symbol, interval, data)

    return data


def _augment_intraday_metadata(df: pd.DataFrame) -> None:
    df['minutes_from_open'] = (
        (df.index - df.index.normalize())
        .total_seconds()
        .astype(float) / 60.0 - 570.0
    )
    df['minutes_from_open'] = df['minutes_from_open'].clip(lower=0.0)

    session_groups = df.groupby('session_date')
    df['bar_index'] = session_groups.cumcount()
    counts = session_groups['bar_index'].transform('max').replace(0, 1)
    df['time_fraction'] = df['bar_index'] / counts
    df['is_session_end'] = df['bar_index'] == counts


def _daterange(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def build_intraday_dataset(
    symbols: Tuple[str, str],
    interval: str = "15m",
    start: Optional[str | date | datetime] = None,
    end: Optional[str | date | datetime] = None
) -> pd.DataFrame:
    primary, benchmark = symbols
    primary_df = load_intraday_data(primary, interval=interval, start=start, end=end)
    benchmark_df = load_intraday_data(benchmark, interval=interval, start=start, end=end)
    primary_prefixed = _prefix_columns(primary_df, primary.lower())
    benchmark_prefixed = _prefix_columns(benchmark_df, benchmark.lower())

    # Keep metadata from primary dataset only to avoid duplicate columns during join
    for meta_col in METADATA_COLUMNS:
        if meta_col in benchmark_prefixed.columns:
            benchmark_prefixed = benchmark_prefixed.drop(columns=[meta_col])

    merged = primary_prefixed.join(benchmark_prefixed, how='inner', lsuffix='', rsuffix='')
    return merged


def _prefix_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col in METADATA_COLUMNS:
            continue
        rename_map[col] = f"{symbol}_{col.lower()}"
    return df.rename(columns=rename_map)


def get_intraday_date_bounds(symbol: str, interval: str = "15m") -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return earliest and latest session dates available for a symbol."""

    config = IntradayStoreConfig(symbol=symbol, interval=interval)
    db_manager = DatabaseManager()
    earliest = None
    latest = None

    bounds = db_manager.get_intraday_session_bounds(symbol, interval)
    if bounds:
        if bounds.get("min_date"):
            earliest = pd.Timestamp(bounds["min_date"])
        if bounds.get("max_date"):
            latest = pd.Timestamp(bounds["max_date"])

    cached_sessions = list_cached_sessions(config)
    if cached_sessions:
        cache_earliest = pd.Timestamp(min(cached_sessions))
        cache_latest = pd.Timestamp(max(cached_sessions))
        earliest = cache_earliest if earliest is None else min(earliest, cache_earliest)
        latest = cache_latest if latest is None else max(latest, cache_latest)

    if earliest is None or latest is None:
        raise ValueError(f"No intraday sessions available for {symbol} ({interval})")

    return earliest.normalize(), latest.normalize()
