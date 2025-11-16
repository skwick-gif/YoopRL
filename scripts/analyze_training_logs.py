"""Summarize recent actor/critic/entropy metrics plus drawdowns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCALAR_TAGS: Dict[str, str] = {
    "train/actor_loss": "Actor Loss",
    "train/critic_loss": "Critic Loss",
    "train/entropy_loss": "Entropy Loss",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize training logs and metadata")
    parser.add_argument(
        "--log-dir",
        default="backend/models/sac/tensorboard",
        help="Root TensorBoard directory (default: backend/models/sac/tensorboard)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of most recent runs to summarize (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of recent scalar points to average (default: 200)",
    )
    parser.add_argument(
        "--metadata",
        help="Optional explicit metadata JSON path for drawdown/return stats",
    )
    parser.add_argument(
        "--metadata-root",
        default="data/training",
        help="Directory to search for *_metadata.json when --metadata not provided (default: data/training)",
    )
    return parser.parse_args()


def _latest_metadata(metadata_root: Path) -> Optional[Path]:
    candidates = list(metadata_root.rglob("*_metadata.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_metadata(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "symbol": payload.get("symbol"),
        "timestamp": payload.get("timestamp"),
        "sharpe_ratio": float(payload.get("sharpe_ratio", payload.get("sharpe", 0.0))),
        "total_return": float(payload.get("total_return", payload.get("return", 0.0))),
        "max_drawdown": float(payload.get("max_drawdown", payload.get("drawdown", 0.0))),
    }


def _summarize_scalar(events, limit: int) -> Optional[Dict[str, float]]:
    if not events:
        return None
    tail = events[-limit:] if limit > 0 else events
    values = [item.value for item in tail]
    return {
        "latest": values[-1],
        "avg_last": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def _summarize_run(run_dir: Path, limit: int) -> Dict[str, Dict[str, float]]:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    summaries: Dict[str, Dict[str, float]] = {}
    for tag, label in SCALAR_TAGS.items():
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        stats = _summarize_scalar(accumulator.Scalars(tag), limit)
        if stats:
            summaries[label] = stats
    return summaries


def _iter_run_dirs(log_dir: Path) -> List[Path]:
    if not log_dir.exists():
        return []
    dirs = [path for path in log_dir.iterdir() if path.is_dir()]
    dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return dirs


def main() -> None:
    args = _parse_args()
    log_dir = Path(args.log_dir)
    run_dirs = _iter_run_dirs(log_dir)

    if not run_dirs:
        print(f"No TensorBoard runs found under {log_dir}")
    else:
        print(f"TensorBoard root: {log_dir}")
        for run_dir in run_dirs[: args.runs]:
            print(f"\nRun: {run_dir.name}")
            try:
                summaries = _summarize_run(run_dir, args.limit)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! Failed to read scalars: {exc}")
                continue
            if not summaries:
                print("  (no tracked scalars yet)")
                continue
            for label, stats in summaries.items():
                latest = stats["latest"]
                average = stats["avg_last"]
                print(
                    f"  {label:<13} latest={latest:+.4f} avg_last{stats['count']}={average:+.4f} "
                    f"range=({stats['min']:+.4f}, {stats['max']:+.4f})"
                )

    metadata_path: Optional[Path] = None
    if args.metadata:
        metadata_path = Path(args.metadata)
    else:
        metadata_path = _latest_metadata(Path(args.metadata_root))

    if metadata_path and metadata_path.exists():
        details = _load_metadata(metadata_path)
        print("\nLatest metadata snapshot:")
        print(f"  Path: {metadata_path}")
        print(
            "  Symbol: {symbol} | Sharpe: {sharpe:.2f} | Return: {ret:+.2f}% | Max DD: {dd:.2f}%".format(
                symbol=details.get("symbol", "?"),
                sharpe=details.get("sharpe_ratio", 0.0),
                ret=details.get("total_return", 0.0),
                dd=details.get("max_drawdown", 0.0),
            )
        )
    else:
        print("\nNo metadata file available to summarize.")
if __name__ == "__main__":
    main()
