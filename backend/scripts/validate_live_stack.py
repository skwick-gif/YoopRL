"""File Note: CLI utility that validates live data stack readiness."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Tuple

import requests

from backend.utils.paths import default_database_path

DEFAULT_BRIDGE_URL = "http://localhost:5080"
DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_DB_PATH = default_database_path()


def check_bridge(base_url: str) -> Tuple[bool, str]:
    base = base_url.rstrip("/")
    try:
        health = requests.get(f"{base}/health", timeout=1)
        if health.status_code != 200:
            return False, f"health endpoint responded with {health.status_code}"

        account = requests.get(f"{base}/account", timeout=2)
        portfolio = requests.get(f"{base}/portfolio", timeout=2)
        if account.status_code != 200 or portfolio.status_code != 200:
            return False, "account/portfolio endpoints not available"

        return True, "bridge reachable and returning account & portfolio"
    except requests.RequestException as exc:
        return False, f"bridge unreachable ({exc})"


def check_backend(base_url: str) -> Tuple[bool, str]:
    base = base_url.rstrip("/")
    try:
        response = requests.get(f"{base}/health", timeout=1)
        if response.status_code != 200:
            return False, f"backend health returned {response.status_code}"
        return True, "backend API healthy"
    except requests.RequestException as exc:
        return False, f"backend unreachable ({exc})"


def check_database(db_path: Path) -> Tuple[bool, str]:
    if not db_path.exists():
        return False, f"missing file {db_path.as_posix()}"

    try:
        with sqlite3.connect(db_path, timeout=1) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_actions';")
            row = cursor.fetchone()
            if row is None:
                return False, "agent_actions table not initialized"
            cursor.execute("SELECT COUNT(1) FROM agent_actions;")
            count = cursor.fetchone()[0]
            return True, f"agent_actions table ready (rows={count})"
    except sqlite3.Error as exc:
        return False, f"database access failed ({exc})"


def check_pyqt6() -> Tuple[bool, str]:
    try:
        import PyQt6  # pylint: disable=unused-import

        return True, "PyQt6 import succeeded"
    except Exception as exc:  # pragma: no cover - import guard
        return False, f"PyQt6 not available ({exc})"


def _format_result(ok: bool, label: str, detail: str) -> str:
    status = "OK" if ok else "FAIL"
    return f"[{status}] {label}: {detail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate live trading stack readiness")
    parser.add_argument("--bridge-url", default=DEFAULT_BRIDGE_URL, help="InterReact bridge base URL")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="Backend API base URL")
    parser.add_argument(
        "--database",
        default=str(DEFAULT_DB_PATH),
        help="Path to SQLite trading database",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.database)

    checks = [
        ("Bridge",) + check_bridge(args.bridge_url),
        ("Backend",) + check_backend(args.backend_url),
        ("Database",) + check_database(db_path),
        ("PyQt6",) + check_pyqt6(),
    ]

    all_ok = True
    for label, ok, detail in checks:
        print(_format_result(ok, label, detail))
        all_ok &= ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
