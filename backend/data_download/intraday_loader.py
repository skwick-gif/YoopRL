"""Intraday (15-minute) data management for SAC + DSR pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import pytz
import requests

from database.db_manager import DatabaseManager
from importlib import import_module

from data_download.config import APIKeysConfig
from backend.utils.market_calendar import trading_sessions_between

INTRADAY_ROOT = Path("data/intraday")
NY_TZ = pytz.timezone("America/New_York")
SUPPORTED_INTERVALS = {"15m"}
ALLOWED_INTRADAY_SYMBOLS = frozenset({
    "SPY",
    "QQQ",
    "IWM",
    "TNA",
    "UPRO",
    "TQQQ",
    "DIA",
    "UDOW",
})

_LOGGER = logging.getLogger(__name__)
_TWELVE_DATA_BASE_URL = "https://api.twelvedata.com/time_series"
_INTERVAL_ALIASES = {
    "15m": "15min",
}


def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").upper()


def _validate_intraday_symbol(symbol: str) -> str:
    normalized = _normalize_symbol(symbol)
    if normalized not in ALLOWED_INTRADAY_SYMBOLS:
        allowed = ", ".join(sorted(ALLOWED_INTRADAY_SYMBOLS))
        raise ValueError(
            f"Intraday downloads currently support only {{{allowed}}}; received '{symbol}'."
        )
    return normalized


@dataclass
class IntradayStoreConfig:
    symbol: str
    interval: str = "15m"
    root: Path = INTRADAY_ROOT

    def __post_init__(self) -> None:
        if self.interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval {self.interval}. Supported: {SUPPORTED_INTERVALS}")
        self.symbol = _validate_intraday_symbol(self.symbol)

    @property
    def base_dir(self) -> Path:
        path = self.root / self.symbol / self.interval
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
    td_interval = _INTERVAL_ALIASES.get(interval, interval)
    candidate_keys = _resolve_twelve_data_keys()
    if not candidate_keys:
        raise RuntimeError(
            "Twelve Data API key is not configured. Set TWELVE_DATA_KEY/TWELVEDATA_API_KEY in the environment, "
            "or update backend/data_download/twelvedata_downloader.py with a valid API_KEY."
        )

    params = {
        "symbol": symbol,
        "interval": td_interval,
        "format": "JSON",
        "timezone": "America/New_York",
        "start_date": session_start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": session_end.strftime("%Y-%m-%d %H:%M:%S"),
        "outputsize": 5000,
        "order": "asc",
    }

    last_message: Optional[str] = None
    for idx, api_key in enumerate(candidate_keys):
        params["apikey"] = api_key
        try:
            response = requests.get(_TWELVE_DATA_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to download intraday data for {symbol} from Twelve Data: {exc}") from exc

        if isinstance(payload, dict) and payload.get("status") == "error":
            message = payload.get("message") or payload.get("note") or "unknown Twelve Data error"
            last_message = message
            if "apikey" in message.lower() and idx < len(candidate_keys) - 1:
                continue
            raise RuntimeError(f"Twelve Data error for {symbol}: {message}")

        values = payload.get("values") if isinstance(payload, dict) else None
        if not values:
            _LOGGER.warning("No intraday values returned for %s on %s", symbol, session_start.date())
            return pd.DataFrame()

        df = pd.DataFrame(values)
        if df.empty:
            return df

        if "datetime" not in df.columns:
            raise RuntimeError("Twelve Data response missing 'datetime' column")

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        for column in ("open", "high", "low", "close", "volume"):
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })

        df = df.set_index("datetime").sort_index()
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(NY_TZ)
        else:
            df.index = df.index.tz_convert(NY_TZ)

        df = df.loc[(df.index >= session_start) & (df.index <= session_end)]

        return df

    raise RuntimeError(f"Twelve Data error for {symbol}: {last_message or 'no valid API key candidates'}")


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

    for session_day in trading_sessions_between(start, end):
        path = _session_path(config, session_day)
        if path.exists():
            continue

        session_start = NY_TZ.localize(
            datetime.combine(session_day, datetime.min.time()) + timedelta(hours=9, minutes=30)
        )
        session_end = NY_TZ.localize(
            datetime.combine(session_day, datetime.min.time()) + timedelta(hours=15, minutes=45)
        )

        try:
            df = _download_intraday(symbol, session_start, session_end, interval)
        except RuntimeError as exc:  # noqa: BLE001
            message = str(exc).lower()
            if "market" in message and "closed" in message:
                _LOGGER.info(
                    "Skipping %s %s: market closed according to provider", symbol, session_day
                )
                continue
            if "no data" in message and "available" in message:
                _LOGGER.info(
                    "Skipping %s %s: provider returned no data", symbol, session_day
                )
                continue
            raise

        if not df.empty:
            _save_session(config, session_day, df)
            saved_paths.append(path)

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


def _resolve_twelve_data_keys() -> List[str]:
    """Collect possible Twelve Data API keys with deduplication."""

    candidates = [
        os.getenv("TWELVE_DATA_KEY"),
        os.getenv("TWELVEDATA_API_KEY"),
        APIKeysConfig().twelve_data_key,
    ]

    try:
        downloader_module = import_module("backend.data_download.twelvedata_downloader")
        candidates.append(getattr(downloader_module, "API_KEY", ""))
    except Exception:
        pass

    resolved: List[str] = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        stripped = candidate.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        resolved.append(stripped)
    return resolved


def _get_twelve_data_key() -> str:
    keys = _resolve_twelve_data_keys()
    if not keys:
        raise RuntimeError(
            "Twelve Data API key is not configured. Set TWELVE_DATA_KEY/TWELVEDATA_API_KEY in the environment, "
            "or update backend/data_download/twelvedata_downloader.py with a valid API_KEY."
        )
    return keys[0]


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

    expected_sessions = trading_sessions_between(start_date, end_date)
    expected_session_set = set(expected_sessions)

    data_from_sql = False
    if not force_download:
        start_ts = pd.Timestamp(start_date).tz_localize(NY_TZ).isoformat()
        end_ts = (pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)).tz_localize(NY_TZ).isoformat()
        sql_df = db_manager.get_intraday_data(symbol, interval=interval, start=start_ts, end=end_ts)
        if not sql_df.empty:
            data = sql_df.copy()
            data_from_sql = True

    if not data_from_sql and (force_download or not cached_sessions.issuperset(expected_session_set)):
        download_and_cache_sessions(symbol, start_date, end_date, interval=interval)

    if not data_from_sql:
        target_paths = [
            _session_path(config, session_day)
            for session_day in expected_sessions
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
    """Backward-compatible alias returning only trading sessions."""

    yield from trading_sessions_between(start, end)


def build_intraday_dataset(
    symbols: Tuple[str, str],
    interval: str = "15m",
    start: Optional[str | date | datetime] = None,
    end: Optional[str | date | datetime] = None
) -> pd.DataFrame:
    primary, benchmark = (_validate_intraday_symbol(sym) for sym in symbols)

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
