"""CLI helper to run walk-forward training/evaluation cycles."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:  # pragma: no cover - CLI convenience
    sys.path.append(str(ROOT_DIR))

from config.training_config import TrainingConfig  # noqa: E402
from training.walk_forward import (  # noqa: E402
    WalkForwardWindow,
    generate_walk_forward_windows,
    run_walk_forward_training_pipeline,
)

DEFAULT_OUTPUT_DIR = Path("backend/evaluation/walk_forward_results")


def _load_training_config(
    config_path: Optional[str],
    *,
    symbol: str,
    agent_type: str,
) -> TrainingConfig:
    if config_path:
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        payload = path.read_text(encoding="utf-8")
        try:
            config_data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse config JSON: {exc}") from exc
        config = TrainingConfig.from_dict(config_data)
    else:
        config = TrainingConfig(agent_type=agent_type, symbol=symbol)

    if config.symbol.upper() != symbol.upper():
        config.symbol = symbol.upper()

    return config


def _load_windows_from_file(path: str) -> Sequence[WalkForwardWindow]:
    windows_path = Path(path).expanduser().resolve()
    if not windows_path.exists():
        raise FileNotFoundError(f"Walk-forward window file not found: {windows_path}")

    try:
        payload = json.loads(windows_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse windows JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("Walk-forward windows JSON must be a list of window objects")

    windows: List[WalkForwardWindow] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Walk-forward windows must be dictionaries with train/test spans")
        required_keys = {"train_start", "train_end", "test_start", "test_end"}
        missing = required_keys - set(entry.keys())
        if missing:
            raise ValueError(f"Window entry missing keys: {sorted(missing)}")
        windows.append(WalkForwardWindow(**entry))

    return windows


def _print_window_summary(windows: Sequence[WalkForwardWindow]) -> None:
    if not windows:
        print("No walk-forward windows resolved.")
        return

    print(f"Resolved {len(windows)} walk-forward windows:")
    for idx, window in enumerate(windows, start=1):
        train_days = (window.train_end - window.train_start).days + 1
        test_days = (window.test_end - window.test_start).days + 1
        print(
            f"[{idx}] Train {window.train_start.date()} → {window.train_end.date()}"
            f" ({train_days} days) | Test {window.test_start.date()} → {window.test_end.date()}"
            f" ({test_days} days)"
        )


def run_walk_forward_cli(args: argparse.Namespace) -> None:
    symbol = args.symbol.upper()
    agent_type = args.agent_type.upper()

    base_config = _load_training_config(args.config, symbol=symbol, agent_type=agent_type)

    if args.seed is not None:
        try:
            base_config.training_settings.random_seed = int(args.seed)
        except (TypeError, ValueError):
            raise ValueError("--seed must be an integer value")

    validation = base_config.validate()
    if not all(len(section_errors) == 0 for section_errors in validation.values()):
        raise ValueError(f"Training configuration failed validation: {validation}")

    windows: Sequence[WalkForwardWindow]
    if args.windows:
        windows = _load_windows_from_file(args.windows)
        auto_generate = False
    else:
        windows = generate_walk_forward_windows(
            symbol=symbol,
            benchmark_symbol=args.benchmark,
            interval=args.interval,
            train_years=args.train_years,
            test_years=args.test_years,
            allow_partial_final=args.allow_partial_final,
        )
        auto_generate = True

    if args.dry_run:
        _print_window_summary(windows)
        return

    output_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[WALK-FORWARD] Starting pipeline...")
    print(f"   Symbol: {symbol}")
    print(f"   Agent: {base_config.agent_type}")
    print(f"   Windows: {len(windows)}")
    if base_config.training_settings.random_seed not in (None, ""):
        print(f"   Random seed: {base_config.training_settings.random_seed}")
    print(f"   Output directory: {output_dir}")

    summary = run_walk_forward_training_pipeline(
        base_config=base_config,
        windows=windows,
        output_dir=str(output_dir),
        deterministic=not args.stochastic,
        auto_generate=auto_generate,
        train_years=args.train_years,
        test_years=args.test_years,
        benchmark_symbol=args.benchmark,
        verbose=not args.quiet,
        allow_partial_final=args.allow_partial_final,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_name = f"walk_forward_summary_{symbol}_{timestamp}.json"
    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[WALK-FORWARD] Pipeline finished.")
    print(f"   Status: {summary.get('status')} | Runs: {len(summary.get('runs', []))}")
    print(f"   Summary artifact: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run walk-forward training/evaluation pipeline for intraday agents.",
    )
    parser.add_argument("--symbol", required=True, help="Primary symbol to train/evaluate (e.g., TQQQ)")
    parser.add_argument(
        "--agent-type",
        default="SAC_INTRADAY_DSR",
        help="Agent type to instantiate when config is not supplied (default: SAC_INTRADAY_DSR)",
    )
    parser.add_argument("--config", help="Path to training config JSON payload")
    parser.add_argument("--windows", help="Path to JSON list of walk-forward window objects")
    parser.add_argument("--benchmark", help="Override benchmark symbol")
    parser.add_argument("--interval", default="15m", help="Intraday interval (default: 15m)")
    parser.add_argument("--train-years", type=int, default=2, help="Seed training span in years")
    parser.add_argument("--test-years", type=int, default=1, help="Evaluation span in years")
    parser.add_argument(
        "--no-partial-final",
        dest="allow_partial_final",
        action="store_false",
        help="Disallow truncated final evaluation window",
    )
    parser.add_argument("--output-dir", help="Directory for evaluation artifacts")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions during evaluation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print windows without training")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbose logging output")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible training runs")

    parser.set_defaults(allow_partial_final=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run_walk_forward_cli(args)
    except Exception as exc:  # pragma: no cover - CLI convenience
        parser.error(str(exc))


if __name__ == "__main__":
    main()
