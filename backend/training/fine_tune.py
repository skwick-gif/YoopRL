"""Fine-tune previously trained SAC intraday models."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import sys

from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure sibling packages resolve when invoked as script
sys.path.append(str(Path(__file__).parent.parent))

from data_download.intraday_features import IntradayFeatureSpec, add_intraday_features
from data_download.intraday_loader import (
    METADATA_COLUMNS as INTRADAY_METADATA_COLUMNS,
    build_intraday_dataset,
    get_intraday_date_bounds,
)
from evaluation.backtester import evaluate_trained_model
from environments.intraday_env import IntradayEquityEnv, IntradaySessionSampler
from models.model_manager import ModelManager

from training.commission import resolve_commission_config, resolve_slippage_config
from training.rewards.dsr_wrapper import DSRConfig, DSRRewardWrapper
from training.train import ContinuousActionAdapter, _extract_history_config


def _coerce_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.Timestamp(value)
    return ts.normalize()


def _resolve_date_window(
    symbol: str,
    interval: str,
    days: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    requested_start = _coerce_date(start_date)
    requested_end = _coerce_date(end_date)

    earliest_available, latest_available = get_intraday_date_bounds(symbol, interval)
    earliest_available = earliest_available.normalize()
    latest_available = latest_available.normalize()

    end_ts = requested_end or latest_available
    days = max(1, int(days))
    start_ts = requested_start or (end_ts - timedelta(days=days - 1))

    if start_ts < earliest_available:
        start_ts = earliest_available
    if end_ts > latest_available:
        end_ts = latest_available
    if start_ts > end_ts:
        raise ValueError("Resolved start date is after end date for fine-tune window.")

    return start_ts, end_ts


def _prepare_intraday_frame(
    symbol: str,
    benchmark_symbol: str,
    interval: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    features_used: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], int]:
    dataset = build_intraday_dataset(
        (symbol, benchmark_symbol),
        interval=interval,
        start=window_start.date(),
        end=window_end.date(),
    )

    if dataset.empty:
        raise ValueError(f"No intraday data available for {symbol} between {window_start.date()} and {window_end.date()}.")

    dataset = dataset.sort_index()
    feature_spec = IntradayFeatureSpec(primary_symbol=symbol, benchmark_symbol=benchmark_symbol)
    dataset = add_intraday_features(dataset, feature_spec)

    numeric_cols = [col for col in dataset.columns if is_numeric_dtype(dataset[col])]
    if numeric_cols:
        dataset[numeric_cols] = (
            dataset[numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

    feature_frame = pd.DataFrame(index=dataset.index)
    missing_features: List[str] = []
    for column in features_used:
        if column in dataset.columns:
            feature_frame[column] = dataset[column].astype(np.float32, copy=False)
        else:
            missing_features.append(column)
            feature_frame[column] = np.zeros(len(dataset), dtype=np.float32)

    metadata_columns = [col for col in INTRADAY_METADATA_COLUMNS if col in dataset.columns]
    for meta_col in metadata_columns:
        feature_frame[meta_col] = dataset[meta_col]

    primary_prefix = symbol.lower()
    close_candidates: Iterable[str] = (
        f"{primary_prefix}_close",
        f"{primary_prefix}_price",
    )
    close_source = next((col for col in close_candidates if col in dataset.columns), None)
    if close_source is None:
        raise ValueError(f"Prepared dataset missing primary close column for {symbol}.")

    feature_frame['close'] = dataset[close_source].astype(np.float32, copy=False)

    if 'session_date' not in feature_frame.columns:
        raise ValueError("Prepared dataset missing 'session_date' column required for session sampling.")

    feature_frame = feature_frame.reset_index(drop=True)
    session_count = feature_frame['session_date'].nunique()

    return feature_frame, missing_features, session_count


def _build_env(
    frame: pd.DataFrame,
    training_settings: Dict[str, Any],
    history_config: Dict[str, Any],
    commission_config: Dict[str, float],
    slippage_config: Dict[str, float],
    reward_mode: str,
    dsr_config: Optional[DSRConfig],
    *,
    shuffle: bool,
) -> ContinuousActionAdapter:
    sampler = IntradaySessionSampler(shuffle=shuffle, sequential=not shuffle)

    forced_exit_minutes = training_settings.get('forced_exit_minutes', 375.0)
    forced_exit_tolerance = training_settings.get('forced_exit_tolerance', 1.0)
    forced_exit_column = training_settings.get('forced_exit_column')

    initial_capital = training_settings.get('initial_capital')
    if initial_capital is None:
        initial_capital = training_settings.get('initial_cash', 100000.0)

    base_env = IntradayEquityEnv(
        df=frame.copy(deep=True),
        initial_capital=float(initial_capital),
        commission=commission_config,
        max_position_size=float(training_settings.get('max_position_size', 1.0)),
        normalize_obs=bool(training_settings.get('normalize_obs', True)),
        history_config=history_config,
        sampler=sampler,
        slippage_config=slippage_config,
        forced_exit_minutes=forced_exit_minutes,
        forced_exit_tolerance=float(forced_exit_tolerance),
        forced_exit_column=forced_exit_column,
    )

    wrapped_env = base_env
    if reward_mode == 'dsr' and dsr_config is not None:
        wrapped_env = DSRRewardWrapper(base_env, config=dsr_config)

    return ContinuousActionAdapter(wrapped_env)


def _resolve_dsr_config(payload: Dict[str, Any]) -> Optional[DSRConfig]:
    if not payload:
        return None

    try:
        clip_raw = payload.get('clip_value')
        clip_value = None if clip_raw in (None, '', 'none', 'None') else float(clip_raw)
        config = DSRConfig(
            decay=float(payload.get('decay', 0.94)),
            epsilon=float(payload.get('epsilon', 1e-9)),
            warmup_steps=int(payload.get('warmup_steps', 200)),
            clip_value=clip_value,
        )
        config.validate()
        return config
    except Exception as exc:  # pragma: no cover - defensive safety
        raise ValueError(f"Invalid DSR configuration: {exc}") from exc


def fine_tune_model(
    *,
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
    days: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timesteps: int = 5000,
    eval_episodes: int = 5,
    dry_run: bool = False,
    clear_buffer: bool = True,
    save_artifacts: bool = True,
    version_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    manager = ModelManager(base_dir='backend/models')
    model, metadata = manager.load_model(model_id=model_id, model_path=model_path)

    agent_type = (metadata.get('canonical_agent_type') or metadata.get('agent_type') or '').upper()
    if agent_type != 'SAC':
        raise ValueError("Fine-tune workflow currently supports SAC-based intraday models only.")

    symbol = metadata.get('symbol')
    if not symbol:
        raise ValueError("Model metadata missing symbol; cannot fine-tune.")

    training_settings = deepcopy(metadata.get('training_settings') or {})
    benchmark_symbol = (
        metadata.get('benchmark_symbol')
        or training_settings.get('benchmark_symbol')
        or 'SPY'
    )
    interval = str(metadata.get('interval') or training_settings.get('interval') or '15m')

    features_used = metadata.get('features_used') or []
    if not features_used:
        raise ValueError("Model metadata missing 'features_used'; cannot reconstruct observation space.")

    history_config = _extract_history_config(metadata.get('features') or {})
    commission_config = resolve_commission_config(training_settings)
    slippage_config = resolve_slippage_config(training_settings)

    reward_mode = str(
        metadata.get('reward_mode')
        or training_settings.get('reward_mode')
        or 'dsr'
    ).lower()
    dsr_payload = metadata.get('dsr_config') or training_settings.get('dsr_config') or {}
    dsr_config = _resolve_dsr_config(dsr_payload) if reward_mode == 'dsr' else None

    window_start, window_end = _resolve_date_window(
        symbol=symbol,
        interval=interval,
        days=days,
        start_date=start_date,
        end_date=end_date,
    )

    frame, missing_features, session_count = _prepare_intraday_frame(
        symbol=symbol,
        benchmark_symbol=benchmark_symbol,
        interval=interval,
        window_start=window_start,
        window_end=window_end,
        features_used=features_used,
    )

    window_start_iso = window_start.date().isoformat()
    window_end_iso = window_end.date().isoformat()
    training_settings['fine_tune_window_start'] = window_start_iso
    training_settings['fine_tune_window_end'] = window_end_iso
    end_candidate = training_settings.get('end_date')
    if not end_candidate or window_end_iso > str(end_candidate):
        training_settings['end_date'] = window_end_iso

    bar_count = len(frame)
    if bar_count == 0:
        raise ValueError("Prepared intraday frame is empty; nothing to fine-tune.")

    print("\n[INFO] Fine-tune configuration")
    print(f"   Model: {metadata.get('model_id', 'unknown')} ({agent_type})")
    print(f"   Symbol: {symbol} | Benchmark: {benchmark_symbol} | Interval: {interval}")
    print(f"   Window: {window_start.date()} → {window_end.date()} ({session_count} sessions, {bar_count} bars)")
    print(f"   Timesteps: {timesteps:,} | Eval episodes: {eval_episodes}")
    if missing_features:
        missing_preview = ', '.join(missing_features[:5])
        if len(missing_features) > 5:
            missing_preview += ', ...'
        print(f"   Missing features ({len(missing_features)}): {missing_preview}")

    if dry_run:
        print("\n[DRY-RUN] Skipping training and artifact updates.")
        return {
            'status': 'dry_run',
            'sessions': session_count,
            'bars': bar_count,
            'window_start': window_start_iso,
            'window_end': window_end_iso,
            'missing_features': missing_features,
        }

    def _make_training_env() -> ContinuousActionAdapter:
        return _build_env(
            frame=frame,
            training_settings=training_settings,
            history_config=history_config,
            commission_config=commission_config,
            slippage_config=slippage_config,
            reward_mode=reward_mode,
            dsr_config=dsr_config,
            shuffle=True,
        )

    vec_env = DummyVecEnv([_make_training_env])
    model.set_env(vec_env)

    if clear_buffer and hasattr(model, 'replay_buffer') and model.replay_buffer is not None:
        model.replay_buffer.reset()
        print("[INFO] Cleared existing replay buffer before fine-tune.")

    print("\n[TRAIN] Starting fine-tune...")
    model.learn(total_timesteps=int(timesteps), reset_num_timesteps=False, progress_bar=True)
    vec_env.close()
    print("[OK] Fine-tune complete.")

    def _make_eval_env() -> ContinuousActionAdapter:
        return _build_env(
            frame=frame,
            training_settings=training_settings,
            history_config=history_config,
            commission_config=commission_config,
            slippage_config=slippage_config,
            reward_mode=reward_mode,
            dsr_config=dsr_config,
            shuffle=False,
        )

    eval_env = _make_eval_env()
    initial_capital = training_settings.get('initial_capital')
    if initial_capital is None:
        initial_capital = training_settings.get('initial_cash', 100000.0)

    eval_results = evaluate_trained_model(
        model=model,
        test_env=eval_env,
        n_eval_episodes=max(1, int(eval_episodes)),
        initial_capital=float(initial_capital),
        deterministic=False,
    )
    eval_env.close()

    now = datetime.utcnow()
    fine_entry = {
        'timestamp': now.isoformat() + 'Z',
        'window_start': window_start_iso,
        'window_end': window_end_iso,
        'sessions': int(session_count),
        'bars': int(bar_count),
        'timesteps': int(timesteps),
        'eval_episodes': int(eval_episodes),
        'metrics': eval_results.get('metrics', {}),
    }

    total_timesteps = getattr(model, 'num_timesteps', None)
    if total_timesteps is not None:
        fine_entry['total_timesteps'] = int(total_timesteps)

    result = {
        'status': 'success',
        'fine_tune': fine_entry,
        'missing_features': missing_features,
        'eval_results': eval_results,
    }

    if not save_artifacts:
        print("[WARN] --no-save requested; skipping model persistence.")
        return result

    version_prefix = now.strftime('v%Y%m%d_%H%M%S')
    if version_suffix:
        version = f"{version_prefix}_{version_suffix}"
    else:
        version = f"{version_prefix}_ft"

    agent_type_lower = agent_type.lower()
    model_dir = Path(manager.base_dir) / agent_type_lower
    model_path_resolved = model_dir / f"{agent_type_lower}_{symbol.upper()}_{version}.zip"

    base_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {'model_id', 'file_path', 'created_at', 'version'}
    }

    base_metadata['created'] = now.isoformat() + 'Z'
    base_metadata['symbol'] = symbol
    base_metadata['benchmark_symbol'] = benchmark_symbol
    base_metadata['training_settings'] = training_settings
    base_metadata['fine_tuned_from_model_id'] = metadata.get('model_id')
    history = list(metadata.get('fine_tune_history', []))
    history.append(fine_entry)
    base_metadata['fine_tune_history'] = history
    base_metadata['fine_tune_last_run'] = fine_entry
    base_metadata['model_path'] = str(model_path_resolved)
    base_metadata['eval_results'] = eval_results

    if total_timesteps is not None:
        base_metadata['total_timesteps'] = int(total_timesteps)

    saved_model_id = manager.save_model(
        model=model,
        agent_type=agent_type,
        symbol=symbol,
        metadata=base_metadata,
        version=version,
    )

    result['model_id'] = saved_model_id
    result['version'] = version
    result['model_path'] = str(model_path_resolved)

    print(f"\n[SAVE] Fine-tuned model saved as {saved_model_id}")
    print(f"       Path: {model_path_resolved}")

    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SAC intraday models with recent data.")
    parser.add_argument('--model-id', help="Identifier of the model to fine-tune.")
    parser.add_argument('--model-path', help="Explicit path to model artifact (optional alternative).")
    parser.add_argument('--days', type=int, default=30, help="Number of calendar days to include when deriving the fine-tune window.")
    parser.add_argument('--start-date', help="Optional explicit start date (YYYY-MM-DD).")
    parser.add_argument('--end-date', help="Optional explicit end date (YYYY-MM-DD).")
    parser.add_argument('--timesteps', type=int, default=5000, help="Fine-tune budget in timesteps.")
    parser.add_argument('--eval-episodes', type=int, default=5, help="Number of evaluation episodes after fine-tuning.")
    parser.add_argument('--dry-run', action='store_true', help="Prepare data and print summary without training or saving.")
    parser.add_argument('--no-clear-buffer', dest='clear_buffer', action='store_false', help="Do not reset the replay buffer before fine-tuning.")
    parser.add_argument('--no-save', action='store_true', help="Run fine-tune but skip saving the updated model.")
    parser.add_argument('--version-suffix', help="Optional suffix appended to the generated version tag (e.g., 'beta').")

    parser.set_defaults(clear_buffer=True)

    args = parser.parse_args(argv)

    if not args.model_id and not args.model_path:
        parser.error("Either --model-id or --model-path must be provided.")

    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    fine_tune_model(
        model_id=args.model_id,
        model_path=args.model_path,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        dry_run=args.dry_run,
        clear_buffer=args.clear_buffer,
        save_artifacts=not args.no_save,
        version_suffix=args.version_suffix,
    )


if __name__ == '__main__':
    main()
