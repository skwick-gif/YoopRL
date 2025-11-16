"""Walk-forward evaluation utilities for the intraday SAC + DSR pipeline."""

from __future__ import annotations

import csv
import json
import logging
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:  # pragma: no cover - CLI convenience
    sys.path.append(str(ROOT_DIR))

from config.training_config import TrainingConfig, TrainingSettings
from data_download.intraday_features import IntradayFeatureSpec, add_intraday_features
from data_download.intraday_loader import (
    build_intraday_dataset,
    ensure_intraday_data_up_to,
    get_intraday_date_bounds,
)
from environments.intraday_env import IntradayEquityEnv, IntradaySessionSampler
from evaluation.metrics import calculate_all_metrics
from models.model_manager import ModelManager
from training.rewards.dsr_wrapper import DSRConfig, DSRRewardWrapper
from training.train import apply_saved_normalizer, train_agent
from training.commission import resolve_commission_config, resolve_slippage_config


_LOGGER = logging.getLogger(__name__)


_HISTORY_KEYS = {"recent_actions", "performance", "position_history", "reward_history"}

_AGENT_TYPE_ALIASES = {
    "SAC_INTRADAY_DSR": "SAC",
    "SAC_INTRADAY": "SAC",
}


class ContinuousActionAdapter(gym.ActionWrapper):
    """Map SAC's continuous actions back onto the env's discrete interface."""

    SELL_THRESHOLD = 0.33
    BUY_THRESHOLD = 0.33
    FLAT_BUY_THRESHOLD = 0.2
    NEUTRAL_WINDOW = 0.3
    NEUTRAL_BIAS = 0.2

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._short_enabled = self._detect_short_support()

    def action(self, action):  # type: ignore[override]
        value = float(action[0]) if isinstance(action, (list, tuple, np.ndarray)) else float(action)
        holdings = self._current_holdings() or 0.0

        if value <= -self.SELL_THRESHOLD:
            if not self._short_enabled and holdings <= 0.0:
                return 0  # HOLD when flat and shorts disabled
            return 2  # SELL

        buy_threshold = self.BUY_THRESHOLD
        if not self._short_enabled and holdings <= 0.0:
            if abs(value) <= self.NEUTRAL_WINDOW:
                value += self.NEUTRAL_BIAS
            buy_threshold = self.FLAT_BUY_THRESHOLD

        if value >= buy_threshold:
            return 1  # BUY
        return 0  # HOLD

    def reverse_action(self, action):  # type: ignore[override]
        mapping = {0: 0.0, 1: 1.0, 2: -1.0}
        return np.array([mapping.get(int(action), 0.0)], dtype=np.float32)

    def _unwrap_env(self):
        env = self.env
        depth = 0
        while hasattr(env, 'env') and depth < 5:
            if hasattr(env, 'holdings') or hasattr(env, 'supports_shorting'):
                break
            env = getattr(env, 'env')
            depth += 1
        return env

    def _detect_short_support(self) -> bool:
        env = self._unwrap_env()
        return bool(getattr(env, 'supports_shorting', False))

    def _current_holdings(self) -> float:
        env = self._unwrap_env()
        return float(getattr(env, 'holdings', 0.0))

@dataclass
class WalkForwardWindow:
    """Single walk-forward train/test window definition."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        self.train_start = _normalize_timestamp(self.train_start)
        self.train_end = _normalize_timestamp(self.train_end)
        self.test_start = _normalize_timestamp(self.test_start)
        self.test_end = _normalize_timestamp(self.test_end)

        if self.train_start >= self.train_end:
            raise ValueError("train_start must be earlier than train_end")
        if self.test_start >= self.test_end:
            raise ValueError("test_start must be earlier than test_end")
        if self.train_end > self.test_start:
            raise ValueError("train_end must be on or before test_start")

    def to_dict(self) -> Dict[str, str]:
        return {
            "train_start": self.train_start.date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "test_start": self.test_start.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
        }

    def filename_stub(self, symbol: str) -> str:
        symbol_upper = symbol.upper()
        start_str = self.test_start.strftime("%Y%m%d")
        end_str = self.test_end.strftime("%Y%m%d")
        return f"{symbol_upper}_{start_str}_{end_str}_walkforward"


def generate_walk_forward_windows(
    symbol: str,
    benchmark_symbol: Optional[str],
    interval: str = "15m",
    train_years: int = 2,
    test_years: int = 1,
    *,
    allow_partial_final: bool = True,
    min_train_start: Optional[pd.Timestamp] = None,
    max_test_end: Optional[pd.Timestamp] = None,
) -> List[WalkForwardWindow]:
    """Construct cumulative walk-forward windows using available intraday data bounds.

    The initial training window spans ``train_years`` using the earliest overlapping data
    between symbol and benchmark. Each subsequent window extends the training end to the
    prior window's test end, ensuring cumulative learning. Test windows default to one year
    and shrink only if ``allow_partial_final`` is True and insufficient data remains.
    """

    benchmark = benchmark_symbol or _infer_benchmark_symbol(symbol)

    primary_bounds = get_intraday_date_bounds(symbol, interval)
    benchmark_bounds = get_intraday_date_bounds(benchmark, interval)

    earliest = max(primary_bounds[0], benchmark_bounds[0]).normalize()
    latest = min(primary_bounds[1], benchmark_bounds[1]).normalize()

    if min_train_start is not None:
        earliest = max(earliest, min_train_start.normalize())

    if max_test_end is not None:
        latest = min(latest, max_test_end.normalize())

    if earliest >= latest:
        raise ValueError(
            f"Insufficient overlapping intraday data for {symbol}/{benchmark} ({interval})"
        )

    train_span = pd.DateOffset(years=train_years)
    test_span = pd.DateOffset(years=test_years)

    first_train_end = (earliest + train_span) - pd.Timedelta(days=1)
    if first_train_end >= latest:
        raise ValueError(
            "Not enough intraday history to satisfy the minimum training span "
            f"({train_years}y) before the first evaluation window."
        )

    windows: List[WalkForwardWindow] = []

    train_start = earliest
    train_end = first_train_end

    while True:
        test_start = train_end + pd.Timedelta(days=1)
        if test_start > latest:
            break

        desired_test_end = (test_start + test_span) - pd.Timedelta(days=1)
        if desired_test_end > latest:
            if not allow_partial_final:
                _LOGGER.info(
                    "Skipping partial walk-forward window for %s; not enough data "
                    "for a %s-year evaluation span.",
                    symbol,
                    test_years,
                )
                break
            _LOGGER.warning(
                "Truncating final walk-forward window for %s to available data (%s → %s).",
                symbol,
                test_start.date(),
                latest.date(),
            )
            test_end = latest
        else:
            test_end = desired_test_end

        if test_end <= train_end:
            break

        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        if test_end >= latest:
            break

        train_end = test_end

    return windows


def run_walk_forward_evaluation(
    config: Union[TrainingConfig, Dict[str, Any]],
    windows: Optional[Sequence[Union[WalkForwardWindow, Dict[str, Any]]]] = None,
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
    output_dir: str = "backend/evaluation/walk_forward_results",
    deterministic: bool = True,
    auto_generate: bool = True,
    train_years: int = 2,
    test_years: int = 1,
    benchmark_symbol: Optional[str] = None,
    verbose: bool = True,
    allow_partial_final: bool = True,
) -> Dict[str, Any]:
    """Execute walk-forward evaluation across the requested windows.

    Args:
        config: Training configuration (dataclass or raw dict) used to train the model.
        windows: Iterable of window definitions (dataclass instances or dictionaries).
        model_id: Optional model identifier managed by ``ModelManager``.
        model_path: Optional direct filesystem path to a saved model.
        output_dir: Directory where JSON evaluation artifacts should be written.
        deterministic: Use deterministic actions when calling ``model.predict``.
        auto_generate: Derive walk-forward windows automatically when ``windows`` is ``None``.
        train_years: Minimum span (years) for the initial training window.
        test_years: Nominal span (years) for each evaluation window.
        benchmark_symbol: Override benchmark symbol (defaults to config/heuristic).
        verbose: Emit per-window summaries during evaluation.
        allow_partial_final: Allow the final evaluation window to shrink to remaining data
            instead of being discarded when insufficient history exists.

    Returns:
        Summary dictionary containing per-window evaluation results and metadata.
    """

    cfg = _coerce_training_config(config)

    model, metadata = _load_model(model_id=model_id, model_path=model_path)
    raw_agent_type = metadata.get("agent_type", cfg.agent_type)
    agent_type = _normalize_agent_type(raw_agent_type)
    if agent_type != "SAC":
        raise ValueError(
            "Walk-forward evaluation currently supports SAC agents only "
            f"(got {raw_agent_type})"
        )

    symbol = metadata.get("symbol", cfg.symbol).upper()
    training_settings = _resolve_training_settings(cfg, metadata)

    resolved_benchmark = (
        benchmark_symbol
        or training_settings.benchmark_symbol
        or _infer_benchmark_symbol(symbol)
    )
    interval = training_settings.interval or "15m"

    refresh_end_date: Optional[pd.Timestamp] = None
    min_train_start: Optional[pd.Timestamp] = None
    max_test_end: Optional[pd.Timestamp] = None

    if training_settings.start_date:
        try:
            min_train_start = pd.Timestamp(training_settings.start_date)
        except Exception:  # pragma: no cover
            min_train_start = None

    if training_settings.end_date:
        try:
            max_test_end = pd.Timestamp(training_settings.end_date)
        except Exception:  # pragma: no cover
            max_test_end = None

    if max_test_end is not None:
        refresh_end_date = max_test_end

    parsed_windows: Optional[List[WalkForwardWindow]] = None

    if windows:
        parsed_windows = [_coerce_window(window) for window in windows]
        refresh_candidates = [
            pd.Timestamp(window.test_end).normalize()
            for window in parsed_windows
            if window.test_end is not None
        ]
        if refresh_candidates:
            refresh_end_date = max(refresh_candidates)

    ensure_intraday_data_up_to(
        [symbol, resolved_benchmark],
        interval=interval,
        end=refresh_end_date.date() if refresh_end_date is not None else None,
    )

    if not windows:
        if not auto_generate:
            raise ValueError("Walk-forward windows must be provided when auto_generate is False")

        parsed_windows = generate_walk_forward_windows(
            symbol=symbol,
            benchmark_symbol=resolved_benchmark,
            interval=interval,
            train_years=train_years,
            test_years=test_years,
            allow_partial_final=allow_partial_final,
            min_train_start=min_train_start,
            max_test_end=max_test_end,
        )
    elif parsed_windows is None:
        parsed_windows = [_coerce_window(window) for window in windows]

    if not parsed_windows:
        raise ValueError("No walk-forward windows available for evaluation")

    features_payload = _features_payload(cfg)
    history_config = _extract_history_config(features_payload)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dsr_config = _resolve_dsr_config(cfg, metadata)
    reward_mode = (training_settings.reward_mode or "").lower()

    window_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    commission_config = resolve_commission_config(training_settings)
    slippage_config = resolve_slippage_config(training_settings)

    for window in parsed_windows:
        try:
            window_result = _evaluate_single_window(
                model=model,
                window=window,
                symbol=symbol,
                benchmark_symbol=resolved_benchmark,
                interval=interval,
                training_settings=training_settings,
                history_config=history_config,
                reward_mode=reward_mode,
                dsr_config=dsr_config,
                deterministic=deterministic,
                output_path=output_path,
                metadata=metadata,
                commission_config=commission_config,
                slippage_config=slippage_config,
            )
            window_results.append(window_result)

            if verbose:
                metrics = window_result.get("metrics", {})
                sharpe = float(metrics.get("sharpe_ratio", 0.0))
                ret = float(metrics.get("total_return", 0.0))
                max_dd = float(metrics.get("max_drawdown", 0.0))
                window_info = window.to_dict()
                print(
                    "[WALK-FORWARD][EVAL]"
                    f" Train {window_info['train_start']}->{window_info['train_end']}"
                    f" | Test {window_info['test_start']}->{window_info['test_end']}"
                    f" | Sharpe: {sharpe:.2f}"
                    f" | Return: {ret:+.2f}%"
                    f" | Max DD: {max_dd:.2f}%"
                )
        except Exception as exc:  # pragma: no cover - surfaced to caller
            errors.append({
                "window": window.to_dict(),
                "error": str(exc),
            })

    status = "completed" if not errors else ("partial" if window_results else "failed")

    return {
        "status": status,
        "symbol": symbol,
    "agent_type": raw_agent_type,
        "model_id": metadata.get("model_id", model_id),
        "windows_evaluated": len(window_results),
        "windows_failed": len(errors),
        "results": window_results,
        "errors": errors,
        "created_at": datetime.now(UTC).isoformat(),
    }


def run_walk_forward_training_pipeline(
    base_config: Union[TrainingConfig, Dict[str, Any]],
    windows: Optional[Sequence[Union[WalkForwardWindow, Dict[str, Any]]]] = None,
    output_dir: str = "backend/evaluation/walk_forward_results",
    deterministic: bool = True,
    auto_generate: bool = True,
    train_years: int = 2,
    test_years: int = 1,
    benchmark_symbol: Optional[str] = None,
    verbose: bool = True,
    allow_partial_final: bool = True,
) -> Dict[str, Any]:
    """Train and evaluate SAC models sequentially across walk-forward windows.

    Args:
        base_config: Seed training configuration for the walk-forward runs.
        windows: Optional explicit window definitions to use.
        output_dir: Directory where evaluation artifacts are stored.
        deterministic: Use deterministic actions for evaluation rollouts.
        auto_generate: Generate walk-forward windows automatically when ``windows`` is ``None``.
        train_years: Minimum span (years) for the initial training period.
        test_years: Nominal span (years) for each evaluation block.
        benchmark_symbol: Override benchmark symbol (defaults to config/heuristic).
        verbose: Emit logging information about window selection and run progress.
        allow_partial_final: Allow the final evaluation window to shrink to available data.

    Returns:
        Summary payload describing the training/evaluation runs.
    """

    cfg = _coerce_training_config(base_config)
    symbol = cfg.symbol.upper()
    base_settings = cfg.training_settings

    resolved_benchmark = (
        benchmark_symbol
        or base_settings.benchmark_symbol
        or _infer_benchmark_symbol(symbol)
    )
    interval = base_settings.interval or "15m"
    gates = _normalize_acceptance_gates(getattr(base_settings, "walk_forward_gates", {}))
    stop_on_fail = bool(getattr(base_settings, "walk_forward_stop_on_fail", True))
    warm_start_enabled = bool(getattr(base_settings, "walk_forward_warm_start", False))
    gate_grace_windows = max(0, int(getattr(base_settings, "walk_forward_gate_grace_windows", 0) or 0))
    warm_start_path: Optional[str] = None
    acceptance_records: List[Dict[str, Any]] = []
    gate_failed = False

    if windows is None:
        if not auto_generate:
            raise ValueError("Walk-forward windows must be provided when auto_generate is False")
        min_train_start = None
        if base_settings.start_date:
            try:
                min_train_start = pd.Timestamp(base_settings.start_date)
            except Exception:  # pragma: no cover - ignore malformed values
                min_train_start = None

        max_test_end = None
        if base_settings.end_date:
            try:
                max_test_end = pd.Timestamp(base_settings.end_date)
            except Exception:  # pragma: no cover
                max_test_end = None

        windows_obj = generate_walk_forward_windows(
            symbol=symbol,
            benchmark_symbol=resolved_benchmark,
            interval=interval,
            train_years=train_years,
            test_years=test_years,
            allow_partial_final=allow_partial_final,
            min_train_start=min_train_start,
            max_test_end=max_test_end,
        )
    else:
        windows_obj = [_coerce_window(window) for window in windows]

    if not windows_obj:
        raise ValueError("No walk-forward windows available for training pipeline")

    if verbose:
        print(f"[STAGE OK] Window setup complete ({len(windows_obj)} window(s))")
        for idx, window in enumerate(windows_obj, start=1):
            train_days = (window.train_end - window.train_start).days + 1
            test_days = (window.test_end - window.test_start).days + 1
            _LOGGER.info(
                "[WALK-FORWARD][WINDOW %s] Train %s→%s (%s days) | Test %s→%s (%s days)",
                idx,
                window.train_start.date(),
                window.train_end.date(),
                train_days,
                window.test_start.date(),
                window.test_end.date(),
                test_days,
            )

    runs: List[Dict[str, Any]] = []
    halt_pipeline = False

    for index, window in enumerate(windows_obj, start=1):
        window_config = deepcopy(cfg.to_dict())
        ts = window_config.setdefault("training_settings", {})
        ts["start_date"] = window.train_start.date().isoformat()
        ts["end_date"] = window.train_end.date().isoformat()
        ts["benchmark_symbol"] = resolved_benchmark
        ts["interval"] = interval
        ts["walk_forward_full_window"] = True
        ts["walk_forward_window"] = window.to_dict()

        frequency_flag = str(ts.get("data_frequency", "")).lower()
        if frequency_flag not in {"intraday", "15m", "15min"}:
            ts["data_frequency"] = "intraday"

        train_split = float(ts.get("train_split", 0.8))
        if train_split >= 1.0:
            ts["train_split"] = 0.95
        elif train_split < 0.5:
            ts["train_split"] = 0.5

        if warm_start_enabled:
            ts["walk_forward_warm_start"] = True
            if warm_start_path:
                ts["warm_start_model_path"] = warm_start_path
        else:
            ts.pop("warm_start_model_path", None)

        ts["walk_forward_stop_on_fail"] = stop_on_fail

        if verbose:
            print(
                f"[WALK-FORWARD][{index}/{len(windows_obj)}][TRAIN]"
                f" Train {window.train_start.date()}->{window.train_end.date()}"
                f" | Symbol {symbol} | Benchmark {resolved_benchmark}"
            )
            print(f"[STAGE] Training start ({index}/{len(windows_obj)})")

        train_result = train_agent(window_config)

        run_entry: Dict[str, Any] = {
            "window": window.to_dict(),
            "train_result": train_result,
        }

        if train_result.get("status") != "success":
            runs.append(run_entry)
            if verbose:
                print(f"[STAGE FAIL] Training failed ({index}/{len(windows_obj)})")
                print("[WALK-FORWARD][TRAIN] Training failed, skipping evaluation for this window.")
            continue

        if warm_start_enabled:
            warm_start_path = train_result.get("model_path") or warm_start_path

        if verbose:
            print(f"[STAGE OK] Training complete ({index}/{len(windows_obj)})")

        metadata = train_result.get("metadata", {})
        train_metrics = {
            "sharpe_ratio": float(metadata.get("sharpe_ratio", 0.0)),
            "total_return": float(metadata.get("total_return", 0.0)),
            "max_drawdown": float(metadata.get("max_drawdown", 0.0)),
        }

        if verbose:
            print(
                "[WALK-FORWARD][TRAIN]"
                f" Sharpe: {train_metrics['sharpe_ratio']:.2f}"
                f" | Return: {train_metrics['total_return']:+.2f}%"
                f" | Max DD: {train_metrics['max_drawdown']:.2f}%"
                f" | Model: {train_result.get('model_path')}"
            )

        if verbose:
            print(f"[STAGE] Evaluation start ({index}/{len(windows_obj)})")

        try:
            evaluation = run_walk_forward_evaluation(
                config=window_config,
                windows=[window],
                model_path=train_result.get("model_path"),
                output_dir=output_dir,
                deterministic=deterministic,
                auto_generate=False,
                benchmark_symbol=resolved_benchmark,
                verbose=verbose,
            )
        except Exception:
            if verbose:
                print(f"[STAGE FAIL] Evaluation error ({index}/{len(windows_obj)})")
            raise

        if verbose:
            print(f"[STAGE OK] Evaluation complete ({index}/{len(windows_obj)})")

        run_entry["evaluation_result"] = evaluation

        acceptance = _evaluate_acceptance_gates(
            evaluation=evaluation,
            gates=gates,
            window_index=index,
            train_metrics=train_metrics,
        )
        if acceptance is not None:
            if index <= gate_grace_windows:
                acceptance["passed"] = True
                acceptance.setdefault("reasons", []).append(
                    f"grace window {index}/{gate_grace_windows} bypass"
                )
                acceptance["grace_window"] = True
            run_entry["acceptance"] = acceptance
            acceptance_records.append(acceptance)
            if acceptance["passed"]:
                if verbose:
                    print(f"[WALK-FORWARD][GATE OK] Window {index} passed acceptance gates.")
            else:
                gate_failed = True
                reasons = acceptance.get("reasons") or ["unspecified"]
                joined_reasons = "; ".join(str(reason) for reason in reasons)
                if verbose:
                    print(
                        f"[WALK-FORWARD][GATE FAIL] Window {index} blocked by gates: {joined_reasons}"
                    )
                if stop_on_fail:
                    halt_pipeline = True
        elif gates:
            run_entry["acceptance"] = {
                "passed": True,
                "window": window.to_dict(),
                "reasons": [],
            }

        runs.append(run_entry)

        if halt_pipeline:
            if verbose:
                print(
                    f"[WALK-FORWARD][STOP] Halting pipeline after window {index} due to gate failure."
                )
            break

    successful_runs = [run for run in runs if run.get("train_result", {}).get("status") == "success"]

    if gate_failed:
        status = "failed"
    elif not successful_runs:
        status = "failed"
    elif len(successful_runs) == len(runs):
        status = "completed"
    else:
        status = "partial"

    acceptance_summary: Optional[Dict[str, str]] = None
    if acceptance_records:
        acceptance_summary = _write_acceptance_summary(
            output_dir=output_dir,
            symbol=symbol,
            records=acceptance_records,
            gates=gates,
            warm_start_enabled=warm_start_enabled,
            stop_on_fail=stop_on_fail,
            gate_grace_windows=gate_grace_windows,
        )

    summary = {
        "status": status,
        "symbol": symbol,
        "benchmark": resolved_benchmark,
        "interval": interval,
        "runs": runs,
        "gates": gates,
        "gate_failed": gate_failed,
        "warm_start_enabled": warm_start_enabled,
        "stop_on_fail": stop_on_fail,
        "gate_grace_windows": gate_grace_windows,
        "created_at": datetime.now(UTC).isoformat(),
    }

    if acceptance_summary:
        summary["acceptance_summary"] = acceptance_summary

    return summary


def _normalize_acceptance_gates(raw_gates: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Coerce acceptance gate thresholds into numeric values (or ``None``)."""

    default_keys = {"min_trades": None, "min_sharpe": None, "max_drawdown": None}
    if not raw_gates:
        return default_keys

    normalized: Dict[str, Optional[float]] = dict(default_keys)
    for key in default_keys:
        value = raw_gates.get(key)
        if value in (None, ""):
            normalized[key] = None
            continue
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            normalized[key] = None

    return normalized


def _evaluate_acceptance_gates(
    evaluation: Dict[str, Any],
    gates: Dict[str, Optional[float]],
    *,
    window_index: int,
    train_metrics: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """Evaluate a single window's results against configured acceptance gates."""

    if not gates or not any(value is not None for value in gates.values()):
        return None

    results = evaluation.get("results") or []
    window_payload = results[0] if results else {}
    metrics = window_payload.get("metrics", {}) or {}
    trades = window_payload.get("trades", []) or []
    window_info = window_payload.get("window") or evaluation.get("window") or {}

    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    total_return = float(metrics.get("total_return", 0.0))
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
    trade_count = len(trades)

    reasons: List[str] = []

    min_trades = gates.get("min_trades")
    if min_trades is not None and trade_count < int(min_trades):
        reasons.append(f"trades {trade_count} < {int(min_trades)}")

    min_sharpe = gates.get("min_sharpe")
    if min_sharpe is not None and sharpe < float(min_sharpe):
        reasons.append(f"sharpe {sharpe:.2f} < {float(min_sharpe):.2f}")

    max_dd_gate = gates.get("max_drawdown")
    if max_dd_gate is not None and max_drawdown > float(max_dd_gate):
        reasons.append(f"max_dd {max_drawdown:.2f} > {float(max_dd_gate):.2f}")

    return {
        "index": window_index,
        "window": window_info,
        "metrics": {
            "sharpe_ratio": sharpe,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
        },
        "train_metrics": train_metrics,
        "trades": trade_count,
        "gates": gates,
        "passed": not reasons,
        "reasons": reasons,
    }


def _write_acceptance_summary(
    *,
    output_dir: str,
    symbol: str,
    records: List[Dict[str, Any]],
    gates: Dict[str, Optional[float]],
    warm_start_enabled: bool,
    stop_on_fail: bool,
    gate_grace_windows: int,
) -> Dict[str, str]:
    """Persist JSON + CSV summaries for walk-forward acceptance gates."""

    if not records:
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base_name = f"{symbol}_walk_forward_acceptance_{timestamp}"

    summary_payload = {
        "symbol": symbol,
        "created_at": datetime.now(UTC).isoformat(),
        "gates": gates,
        "stop_on_fail": stop_on_fail,
        "warm_start_enabled": warm_start_enabled,
        "gate_grace_windows": gate_grace_windows,
        "windows": records,
    }

    json_path = output_path / f"{base_name}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    csv_path = output_path / f"{base_name}.csv"
    fieldnames = [
        "index",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "trades",
        "sharpe_ratio",
        "total_return",
        "max_drawdown",
        "train_sharpe",
        "train_total_return",
        "train_max_drawdown",
        "passed",
        "reasons",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            window_info = record.get("window", {}) or {}
            train_metrics = record.get("train_metrics", {}) or {}
            metrics = record.get("metrics", {}) or {}
            writer.writerow({
                "index": record.get("index"),
                "train_start": window_info.get("train_start"),
                "train_end": window_info.get("train_end"),
                "test_start": window_info.get("test_start"),
                "test_end": window_info.get("test_end"),
                "trades": record.get("trades"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "total_return": metrics.get("total_return"),
                "max_drawdown": metrics.get("max_drawdown"),
                "train_sharpe": train_metrics.get("sharpe_ratio"),
                "train_total_return": train_metrics.get("total_return"),
                "train_max_drawdown": train_metrics.get("max_drawdown"),
                "passed": record.get("passed"),
                "reasons": "; ".join(record.get("reasons") or []),
            })

    return {"json": str(json_path), "csv": str(csv_path)}


def _evaluate_single_window(
    model,
    window: WalkForwardWindow,
    symbol: str,
    benchmark_symbol: str,
    interval: str,
    training_settings: TrainingSettings,
    history_config: Dict[str, Any],
    reward_mode: str,
    dsr_config: Dict[str, Any],
    deterministic: bool,
    output_path: Path,
    metadata: Dict[str, Any],
    commission_config: Dict[str, float],
    slippage_config: Dict[str, float],
) -> Dict[str, Any]:
    test_df = _prepare_intraday_dataframe(
        symbol=symbol,
        benchmark_symbol=benchmark_symbol,
        interval=interval,
        window=window,
        metadata=metadata,
    )

    sampler = IntradaySessionSampler(shuffle=False, sequential=True)
    forced_exit_minutes = getattr(training_settings, "forced_exit_minutes", 375.0)
    forced_exit_tolerance = getattr(training_settings, "forced_exit_tolerance", 1.0)
    forced_exit_column = getattr(training_settings, "forced_exit_column", None)
    base_env = IntradayEquityEnv(
        df=test_df,
        initial_capital=training_settings.initial_capital,
        commission=commission_config,
        max_position_size=training_settings.max_position_size,
        normalize_obs=training_settings.normalize_obs,
        history_config=history_config,
        sampler=sampler,
        slippage_config=slippage_config,
        forced_exit_minutes=forced_exit_minutes,
        forced_exit_tolerance=forced_exit_tolerance,
        forced_exit_column=forced_exit_column,
    )

    env: gym.Env = ContinuousActionAdapter(base_env)

    if reward_mode == "dsr":
        env = DSRRewardWrapper(env, _build_dsr_config(dsr_config))

    equity_curve, trades, session_breakdown = _run_sequential_sessions(
        model=model,
        env=env,
        base_env=base_env,
        deterministic=deterministic,
        initial_capital=training_settings.initial_capital,
    )

    metrics = calculate_all_metrics(
        equity_curve=np.asarray(equity_curve, dtype=float),
        trades=trades,
        initial_balance=float(equity_curve[0]),
    )

    payload = {
        "window": window.to_dict(),
        "symbol": symbol,
        "benchmark": benchmark_symbol,
        "interval": interval,
        "metrics": metrics,
        "session_breakdown": session_breakdown,
        "equity_curve": [round(float(x), 6) for x in equity_curve],
        "trades": [round(float(x), 6) for x in trades],
        "model_metadata": {
            "model_id": metadata.get("model_id"),
            "model_path": metadata.get("model_path"),
            "version": metadata.get("version"),
        },
        "commission": {
            "type": commission_config.get("type"),
            "per_share": float(commission_config.get("per_share", 0.0)),
            "min_fee": float(commission_config.get("min_fee", 0.0)),
            "max_pct": float(commission_config.get("max_pct", 0.0)),
        },
        "slippage": {
            "buy_bps": float(slippage_config.get("buy_bps", 0.0)),
            "sell_bps": float(slippage_config.get("sell_bps", 0.0)),
            "buy_per_share": float(slippage_config.get("buy_per_share", 0.0)),
            "sell_per_share": float(slippage_config.get("sell_per_share", 0.0)),
        },
        "reward_mode": reward_mode,
        "created_at": datetime.now(UTC).isoformat(),
    }

    file_stub = window.filename_stub(symbol)
    file_path = output_path / f"{file_stub}.json"
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    payload["artifact_path"] = str(file_path)
    return payload


def _run_sequential_sessions(
    model,
    env: gym.Env,
    base_env: IntradayEquityEnv,
    deterministic: bool,
    initial_capital: float,
) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
    equity_curve: List[float] = [float(initial_capital)]
    trades: List[float] = []
    session_breakdown: List[Dict[str, Any]] = []

    carry_capital = float(initial_capital)

    for idx, session_date in enumerate(base_env.session_dates):
        base_env.initial_capital = carry_capital
        obs, _ = env.reset()

        session_initial = float(base_env.initial_capital)
        prev_value = session_initial
        steps = 0

        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            total_value = info.get("total_value", getattr(base_env, "total_value", prev_value))
            total_value = float(total_value)

            pnl = total_value - prev_value
            if abs(pnl) > 0.01:
                trades.append(float(pnl))

            equity_curve.append(total_value)
            prev_value = total_value
            steps += 1

        carry_capital = prev_value

        session_return = 0.0
        if session_initial != 0:
            session_return = (carry_capital - session_initial) / session_initial

        session_breakdown.append({
            "index": idx,
            "session_date": str(session_date),
            "steps": steps,
            "final_balance": round(carry_capital, 2),
            "session_return": round(session_return, 6),
        })

    return equity_curve, trades, session_breakdown


def _prepare_intraday_dataframe(
    symbol: str,
    benchmark_symbol: str,
    interval: str,
    window: WalkForwardWindow,
    metadata: Dict[str, Any],
) -> pd.DataFrame:
    dataset = build_intraday_dataset(
        symbols=(symbol, benchmark_symbol),
        interval=interval,
        start=window.train_start.date(),
        end=window.test_end.date(),
    )

    augmented = add_intraday_features(
        dataset,
        IntradayFeatureSpec(primary_symbol=symbol, benchmark_symbol=benchmark_symbol),
    )

    augmented = augmented.dropna().copy()
    augmented["session_date"] = pd.to_datetime(augmented["session_date"]).dt.date

    test_mask = (
        (augmented["session_date"] >= window.test_start.date()) &
        (augmented["session_date"] <= window.test_end.date())
    )

    test_df = augmented.loc[test_mask].copy()
    if test_df.empty:
        raise ValueError(
            "No intraday data available for validation window "
            f"{window.test_start.date()} -> {window.test_end.date()}"
        )

    # Evaluation uses the raw feature scale produced during training so we do
    # not reapply the saved normalizer (it would distort dollar-denominated
    # price columns and break portfolio accounting).
    feature_columns = metadata.get("features_used") or []
    if feature_columns:
        missing_features = [col for col in feature_columns if col not in test_df.columns]
        if missing_features:
            missing_display = ", ".join(sorted(missing_features))
            raise ValueError(
                "Prepared intraday frame is missing features required by the model: "
                f"{missing_display}"
            )

        # Preserve the training feature order while still carrying auxiliary columns
        ordered_columns = list(feature_columns)
        for column in test_df.columns:
            if column not in ordered_columns:
                ordered_columns.append(column)
        test_df = test_df.loc[:, ordered_columns]

    primary_prefix = symbol.lower()
    rename_map = {
        f"{primary_prefix}_close": "close",
        f"{primary_prefix}_open": "open",
        f"{primary_prefix}_high": "high",
        f"{primary_prefix}_low": "low",
        f"{primary_prefix}_volume": "volume",
    }
    test_df = test_df.rename(columns={k: v for k, v in rename_map.items() if k in test_df.columns})

    return test_df.sort_index()


def _coerce_training_config(config: Union[TrainingConfig, Dict[str, Any]]) -> TrainingConfig:
    if isinstance(config, TrainingConfig):
        return config
    if isinstance(config, dict):
        return TrainingConfig.from_dict(config)
    raise TypeError("config must be a TrainingConfig or dict payload")


def _resolve_training_settings(cfg: TrainingConfig, metadata: Dict[str, Any]) -> TrainingSettings:
    settings = cfg.training_settings
    meta_settings = metadata.get("training_settings")
    if isinstance(meta_settings, dict):
        # Overlay metadata settings in case training run mutated values (e.g., auto splits)
        for key, value in meta_settings.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
    return settings


def _resolve_dsr_config(cfg: TrainingConfig, metadata: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.training_settings.dsr_config:
        return dict(cfg.training_settings.dsr_config)
    meta_settings = metadata.get("training_settings", {})
    dsr_from_meta = meta_settings.get("dsr_config") if isinstance(meta_settings, dict) else None
    return dict(dsr_from_meta or {})


def _build_dsr_config(raw: Dict[str, Any]) -> DSRConfig:
    clip_raw = raw.get("clip_value")
    clip_value = None
    if clip_raw not in (None, "", "none", "None"):
        clip_value = float(clip_raw)

    return DSRConfig(
        decay=float(raw.get("decay", 0.94)),
        epsilon=float(raw.get("epsilon", 1e-9)),
        warmup_steps=int(raw.get("warmup_steps", 200)),
        clip_value=clip_value,
    )


def _extract_history_config(features: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(features, dict):
        return {}
    return {key: features[key] for key in _HISTORY_KEYS if key in features}


def _features_payload(cfg: TrainingConfig) -> Dict[str, Any]:
    features = cfg.features
    if hasattr(features, "to_payload"):
        return features.to_payload()
    if is_dataclass(features):
        return asdict(features)
    if isinstance(features, dict):
        return features
    return {}


def _normalize_agent_type(agent_type: Any) -> str:
    if agent_type is None:
        return ""
    normalized = str(agent_type).upper()
    return _AGENT_TYPE_ALIASES.get(normalized, normalized)


def _load_model(
    model_id: Optional[str],
    model_path: Optional[str],
) -> Tuple[Any, Dict[str, Any]]:
    manager = ModelManager()
    if model_id:
        return manager.load_model(model_id=model_id)
    if model_path:
        return manager.load_model(model_path=model_path)
    raise ValueError("Either model_id or model_path must be provided for evaluation")


def _infer_benchmark_symbol(symbol: str) -> str:
    mapping = {
        "TQQQ": "QQQ",
        "SQQQ": "QQQ",
        "UPRO": "SPY",
        "SPXL": "SPY",
        "TNA": "IWM",
        "TMF": "TLT",
    }
    return mapping.get(symbol.upper(), "SPY")


def _normalize_timestamp(value: Union[str, pd.Timestamp, datetime]) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _coerce_window(window: Union[WalkForwardWindow, Dict[str, Any]]) -> WalkForwardWindow:
    if isinstance(window, WalkForwardWindow):
        return window
    if isinstance(window, dict):
        required_keys = {"train_start", "train_end", "test_start", "test_end"}
        missing = required_keys - set(window.keys())
        if missing:
            raise ValueError(f"Walk-forward window missing keys: {sorted(missing)}")
        return WalkForwardWindow(**window)
    raise TypeError("walk-forward window must be a WalkForwardWindow or dict payload")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward utilities")
    parser.add_argument("symbol", help="Primary symbol for intraday data (e.g., TQQQ)")
    parser.add_argument("--benchmark", help="Override benchmark symbol", default=None)
    parser.add_argument("--interval", default="15m", help="Data interval (default: 15m)")
    parser.add_argument("--train-years", type=int, default=2, help="Seed training span in years")
    parser.add_argument("--test-years", type=int, default=1, help="Evaluation span in years")
    parser.add_argument(
        "--no-partial-final",
        dest="allow_partial",
        action="store_false",
        help="Disallow truncated final evaluation window",
    )

    args = parser.parse_args()

    try:
        windows = generate_walk_forward_windows(
            symbol=args.symbol,
            benchmark_symbol=args.benchmark,
            interval=args.interval,
            train_years=args.train_years,
            test_years=args.test_years,
            allow_partial_final=args.allow_partial,
        )
    except Exception as exc:  # pragma: no cover - CLI convenience
        parser.error(str(exc))

    if not windows:
        print("No walk-forward windows generated.")
    else:
        print(f"Resolved {len(windows)} walk-forward windows:")
        for idx, window in enumerate(windows, start=1):
            train_days = (window.train_end - window.train_start).days + 1
            test_days = (window.test_end - window.test_start).days + 1
            print(
                f"[{idx}] Train {window.train_start.date()} → {window.train_end.date()}"
                f" ({train_days} days) | Test {window.test_start.date()} → {window.test_end.date()}"
                f" ({test_days} days)"
            )
