"""Shared helpers for resolving project data directories.

Provides a single place to derive paths that were previously hard-coded to
an absolute drive (e.g., ``d:/YoopRL``). The helpers honor the optional
``YOOPRL_DATA_ROOT`` environment variable and otherwise fall back to the
repository's ``data`` folder. This allows the project to run on machines
without a dedicated ``D:`` drive.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def _project_root() -> Path:
    """Return the repository root directory (two levels up from utils)."""

    return Path(__file__).resolve().parents[2]


def get_data_root() -> Path:
    """Return the base directory for all persistent data.

    The lookup order is:
    1. ``YOOPRL_DATA_ROOT`` environment variable (expanded to an absolute Path).
    2. ``<repo>/data`` within the checked-out workspace.
    """

    env_override = os.getenv("YOOPRL_DATA_ROOT")
    if env_override:
        candidate = Path(env_override).expanduser()
    else:
        candidate = _project_root() / "data"

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate.resolve()


def ensure_data_subdir(*parts: Union[str, Path]) -> Path:
    """Return (and create if missing) a subdirectory under the data root."""

    root = get_data_root()
    target = root
    for part in parts:
        target = target / Path(part)
    target.mkdir(parents=True, exist_ok=True)
    return target


def default_database_path(filename: str = "trading.db") -> Path:
    """Return the full path to the primary SQLite database file."""

    return get_data_root() / filename
