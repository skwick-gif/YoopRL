"""Walk-forward evaluation utilities for the intraday SAC + DSR pipeline."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd

from config.training_config import TrainingConfig, TrainingSettings
from data_download.intraday_features import IntradayFeatureSpec, add_intraday_features
from data_download.intraday_loader import build_intraday_dataset, get_intraday_date_bounds
from environments.intraday_env import IntradayEquityEnv, IntradaySessionSampler
from evaluation.metrics import calculate_all_metrics
from models.model_manager import ModelManager
from training.rewards.dsr_wrapper import DSRConfig, DSRRewardWrapper
from training.train import train_agent
from training.commission import resolve_commission_config


_HISTORY_KEYS = {"recent_actions", "performance", "position_history", "reward_history"}


class ContinuousActionAdapter(gym.ActionWrapper):
    """Map SAC's continuous actions back onto the env's discrete interface."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, action):  # type: ignore[override]
        value = float(action[0]) if isinstance(action, (list, tuple, np.ndarray)) else float(action)
        if value <= -0.33:
            return 2  # SELL
        if value >= 0.33:
            return 1  # BUY
        return 0  # HOLD

    def reverse_action(self, action):  # type: ignore[override]
        mapping = {0: 0.0, 1: 1.0, 2: -1.0}
        return np.array([mapping.get(int(action), 0.0)], dtype=np.float32)


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
) -> List[WalkForwardWindow]:
    """Construct rolling windows using available intraday data bounds."""

    benchmark = benchmark_symbol or _infer_benchmark_symbol(symbol)

    primary_bounds = get_intraday_date_bounds(symbol, interval)
    benchmark_bounds = get_intraday_date_bounds(benchmark, interval)

    earliest = max(primary_bounds[0], benchmark_bounds[0])
    latest = min(primary_bounds[1], benchmark_bounds[1])

    if earliest >= latest:
        raise ValueError(
            f"Insufficient overlapping intraday data for {symbol}/{benchmark} ({interval})"
        )

    windows: List[WalkForwardWindow] = []
    step = pd.DateOffset(years=1)
    train_span = pd.DateOffset(years=train_years)
    test_span = pd.DateOffset(years=test_years)

    current_start = earliest

    while True:
        train_start = current_start.normalize()
        train_end = (train_start + train_span) - pd.Timedelta(days=1)

        if train_end > latest:
            break

        test_start = train_end + pd.Timedelta(days=1)
        if test_start > latest:
            break

        test_end_candidate = (test_start + test_span) - pd.Timedelta(days=1)
        test_end = min(test_end_candidate, latest)

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

        current_start = current_start + step

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
) -> Dict[str, Any]:
    """Execute walk-forward evaluation across the requested windows.

    Args:
        config: Training configuration (dataclass or raw dict) used to train the model.
        windows: Iterable of window definitions (dataclass instances or dictionaries).
        model_id: Optional model identifier managed by ``ModelManager``.
        model_path: Optional direct filesystem path to a saved model.
        output_dir: Directory where JSON evaluation artifacts should be written.
        deterministic: Use deterministic actions when calling ``model.predict``.

    Returns:
        Summary dictionary containing per-window evaluation results and metadata.
    """

    cfg = _coerce_training_config(config)

    model, metadata = _load_model(model_id=model_id, model_path=model_path)
    agent_type = metadata.get("agent_type", cfg.agent_type).upper()
    if agent_type != "SAC":
        raise ValueError(f"Walk-forward evaluation currently supports SAC agents only (got {agent_type})")

    symbol = metadata.get("symbol", cfg.symbol).upper()
    training_settings = _resolve_training_settings(cfg, metadata)

    resolved_benchmark = (
        benchmark_symbol
        or training_settings.benchmark_symbol
        or _infer_benchmark_symbol(symbol)
    )
    interval = training_settings.interval or "15m"

    if not windows:
        if not auto_generate:
            raise ValueError("Walk-forward windows must be provided when auto_generate is False")
        parsed_windows = generate_walk_forward_windows(
            symbol=symbol,
            benchmark_symbol=resolved_benchmark,
            interval=interval,
            train_years=train_years,
            test_years=test_years,
        )
    else:
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
        "agent_type": agent_type,
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
) -> Dict[str, Any]:
    """Train and evaluate SAC models sequentially across walk-forward windows."""

    cfg = _coerce_training_config(base_config)
    symbol = cfg.symbol.upper()
    base_settings = cfg.training_settings

    resolved_benchmark = (
        benchmark_symbol
        or base_settings.benchmark_symbol
        or _infer_benchmark_symbol(symbol)
    )
    interval = base_settings.interval or "15m"

    if windows is None:
        if not auto_generate:
            raise ValueError("Walk-forward windows must be provided when auto_generate is False")
        windows_obj = generate_walk_forward_windows(
            symbol=symbol,
            benchmark_symbol=resolved_benchmark,
            interval=interval,
            train_years=train_years,
            test_years=test_years,
        )
    else:
        windows_obj = [_coerce_window(window) for window in windows]

    if not windows_obj:
        raise ValueError("No walk-forward windows available for training pipeline")

    runs: List[Dict[str, Any]] = []

    for index, window in enumerate(windows_obj, start=1):
        window_config = deepcopy(cfg.to_dict())
        ts = window_config.setdefault("training_settings", {})
        ts["start_date"] = window.train_start.date().isoformat()
        ts["end_date"] = window.train_end.date().isoformat()
        ts["benchmark_symbol"] = resolved_benchmark
        ts["interval"] = interval

        frequency_flag = str(ts.get("data_frequency", "")).lower()
        if frequency_flag not in {"intraday", "15m", "15min"}:
            ts["data_frequency"] = "intraday"

        train_split = float(ts.get("train_split", 0.8))
        if train_split >= 1.0:
            ts["train_split"] = 0.95
        elif train_split < 0.5:
            ts["train_split"] = 0.5

        if verbose:
            print(
                f"[WALK-FORWARD][{index}/{len(windows_obj)}][TRAIN]"
                f" Train {window.train_start.date()}->{window.train_end.date()}"
                f" | Symbol {symbol} | Benchmark {resolved_benchmark}"
            )

        train_result = train_agent(window_config)

        run_entry: Dict[str, Any] = {
            "window": window.to_dict(),
            "train_result": train_result,
        }

        if train_result.get("status") != "success":
            runs.append(run_entry)
            if verbose:
                print("[WALK-FORWARD][TRAIN] Training failed, skipping evaluation for this window.")
            continue

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

        run_entry["evaluation_result"] = evaluation
        runs.append(run_entry)

    successful_runs = [run for run in runs if run.get("train_result", {}).get("status") == "success"]

    if not successful_runs:
        status = "failed"
    elif len(successful_runs) == len(runs):
        status = "completed"
    else:
        status = "partial"

    return {
        "status": status,
        "symbol": symbol,
        "benchmark": resolved_benchmark,
        "interval": interval,
        "runs": runs,
    "created_at": datetime.now(UTC).isoformat(),
    }


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
) -> Dict[str, Any]:
    test_df = _prepare_intraday_dataframe(
        symbol=symbol,
        benchmark_symbol=benchmark_symbol,
        interval=interval,
        window=window,
    )

    sampler = IntradaySessionSampler(shuffle=False, sequential=True)
    base_env = IntradayEquityEnv(
        df=test_df,
        initial_capital=training_settings.initial_capital,
        commission=commission_config,
        max_position_size=training_settings.max_position_size,
        normalize_obs=training_settings.normalize_obs,
        history_config=history_config,
        sampler=sampler,
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
