"""Utilities for importing legacy intraday CSV dumps into the new cache layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from data_download.intraday_loader import INTRADAY_ROOT, NY_TZ, SUPPORTED_INTERVALS
from database.db_manager import DatabaseManager


@dataclass
class ImportReport:
    symbol: str
    interval: str
    source_rows: int
    sessions_written: int
    output_root: Path
    rows_written_sql: int


def _ensure_interval_supported(interval: str) -> str:
    interval = interval.lower()
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {sorted(SUPPORTED_INTERVALS)}")
    return interval


def _prepare_dataframe(csv_path: Path, symbol: Optional[str]) -> tuple[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        raise ValueError("CSV must contain a 'datetime' column")

    detected_symbol = (symbol or df.get("symbol", pd.Series([""])).iloc[0]).strip()
    if not detected_symbol:
        raise ValueError("Symbol could not be inferred from CSV. Pass it explicitly.")

    dt_series = pd.to_datetime(df["datetime"], utc=False, errors="coerce")
    if dt_series.isna().any():
        raise ValueError("Failed to parse some datetime entries in CSV")

    dt_index = pd.DatetimeIndex(dt_series)
    if dt_index.tz is None:
        dt_index = dt_index.tz_localize(NY_TZ)
    else:
        dt_index = dt_index.tz_convert(NY_TZ)

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj_close": "Adj Close",
    })

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index(dt_index).sort_index()
    df = df.drop(columns=[c for c in ("datetime", "symbol", "month") if c in df.columns])
    present_cols = [col for col in numeric_cols if col in df.columns]
    if not present_cols:
        raise ValueError("CSV missing OHLCV columns after renaming")
    df = df[present_cols]
    df = df.dropna(how="any")

    if df.empty:
        raise ValueError("No valid OHLCV rows found in CSV")

    return detected_symbol.upper(), df


def _add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["session_date"] = enriched.index.tz_convert(NY_TZ).date

    minutes_from_midnight = (
        enriched.index - enriched.index.normalize()
    ).total_seconds() / 60.0
    minutes_from_midnight = pd.Series(minutes_from_midnight, index=enriched.index)
    enriched["minutes_from_open"] = (minutes_from_midnight - 570).clip(lower=0.0)

    grouped = enriched.groupby("session_date")
    enriched["bar_index"] = grouped.cumcount()
    last_index = grouped["bar_index"].transform("max").replace(0, 1)
    enriched["time_fraction"] = enriched["bar_index"] / last_index
    enriched["is_session_end"] = enriched["bar_index"] == last_index

    return enriched


def _write_sessions(df: pd.DataFrame, symbol: str, interval: str) -> Iterable[Path]:
    output_dir = INTRADAY_ROOT / symbol / interval
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths = []
    for session_date, session_df in df.groupby("session_date"):
        path = output_dir / f"{session_date}.csv"
        session_df.to_csv(path)
        written_paths.append(path)
    return written_paths


def import_rl_ready_intraday(csv_path: str | Path, symbol: Optional[str] = None, interval: str = "15m") -> ImportReport:
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    interval = _ensure_interval_supported(interval)
    detected_symbol, df = _prepare_dataframe(csv_path, symbol)
    enriched = _add_metadata(df)
    written_paths = list(_write_sessions(enriched, detected_symbol, interval))
    db_rows = DatabaseManager().save_intraday_data(detected_symbol, interval, enriched)

    return ImportReport(
        symbol=detected_symbol,
        interval=interval,
        source_rows=len(df),
        sessions_written=len(written_paths),
        output_root=INTRADAY_ROOT / detected_symbol / interval,
        rows_written_sql=db_rows,
    )


def import_many(csv_files: Iterable[str | Path], interval: str = "15m") -> list[ImportReport]:
    reports = []
    for csv in csv_files:
        reports.append(import_rl_ready_intraday(csv, interval=interval))
    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import legacy intraday CSV files into cache format.")
    parser.add_argument("paths", nargs="+", help="Paths to CSV files (e.g. TNA_RL_ready.csv)")
    parser.add_argument("--interval", default="15m", help="Bar interval (default: 15m)")
    parser.add_argument("--symbol", default=None, help="Override symbol inferred from CSV")

    args = parser.parse_args()

    for path in args.paths:
        report = import_rl_ready_intraday(path, symbol=args.symbol, interval=args.interval)
        print(
            f"Imported {path} -> {report.symbol} ({report.interval}) | "
            f"rows={report.source_rows}, sessions={report.sessions_written}, "
            f"sql_rows={report.rows_written_sql}, output={report.output_root}"
        )
