"""Quick intraday cache coverage summary."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.data_download.intraday_loader import (  # noqa: E402
    IntradayStoreConfig,
    list_cached_sessions,
)


def _symbol_coverage(symbol: str, interval: str) -> Tuple[str, int, date | None, date | None]:
    config = IntradayStoreConfig(symbol=symbol, interval=interval)
    sessions = list_cached_sessions(config)
    if not sessions:
        return symbol, 0, None, None
    return symbol, len(sessions), min(sessions), max(sessions)


def _format_row(symbol: str, count: int, earliest: date | None, latest: date | None) -> str:
    if earliest is None or latest is None:
        return f"{symbol:>5} | sessions: {count:5d} | coverage: <missing>"
    return (
        f"{symbol:>5} | sessions: {count:5d} | coverage: "
        f"{earliest.isoformat()} → {latest.isoformat()}"
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize cached intraday coverage")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["TNA", "IWM", "QQQ", "SPY", "UPRO", "TQQQ"],
        help="Symbols to inspect (default: core intraday set)",
    )
    parser.add_argument(
        "--interval",
        default="15m",
        help="Intraday interval to inspect (default: 15m)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    print(f"Interval: {args.interval}\n")
    for symbol in args.symbols:
        summary = _symbol_coverage(symbol, args.interval)
        print(_format_row(*summary))


if __name__ == "__main__":
    main()
