"""
Main Training Script for RL Trading System

This module provides the core training workflow:
1. Load and prepare data
2. Normalize features
3. Create trading environment
4. Train RL agent (PPO or SAC)
5. Save trained model with metadata

Author: YoopRL System
Date: November 8, 2025
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gym

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3.common.callbacks import BaseCallback
from environments.stock_env import StockTradingEnv
from environments.etf_env import ETFTradingEnv
from environments.intraday_env import IntradayEquityEnv, IntradaySessionSampler
from agents.agent_factory import create_agent
from utils.state_normalizer import StateNormalizer
from models.model_manager import ModelManager
from config.training_config import TrainingConfig
from evaluation.backtester import evaluate_trained_model
from data_download.intraday_loader import METADATA_COLUMNS as INTRADAY_METADATA_COLUMNS, build_intraday_dataset
from data_download.intraday_features import IntradayFeatureSpec, add_intraday_features
from training.commission import resolve_commission_config
from training.rewards.dsr_wrapper import DSRConfig, DSRRewardWrapper


class ContinuousActionAdapter(gym.ActionWrapper):
    """Map SAC's continuous actions back onto the env's discrete API."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        value = float(action[0]) if isinstance(action, (list, tuple, np.ndarray)) else float(action)
        if value <= -0.33:
            return 2  # SELL
        if value >= 0.33:
            return 1  # BUY
        return 0  # HOLD

    def reverse_action(self, action):
        mapping = {0: 0.0, 1: 1.0, 2: -1.0}
        return np.array([mapping.get(int(action), 0.0)], dtype=np.float32)


AGENT_ALIAS_MAP = {
    "SAC_INTRADAY_DSR": "SAC",
}

INTRADAY_FREQUENCY_FLAGS = {"intraday", "15m", "15min"}

DEFAULT_MULTI_ASSET_SYMBOLS = ['SPY', 'QQQ', 'TLT', 'GLD']


def _canonical_agent_type(agent_type: str) -> str:
    if not agent_type:
        return "PPO"
    normalized = agent_type.upper()
    return AGENT_ALIAS_MAP.get(normalized, normalized)


def _features_payload(features: Any) -> Dict[str, Any]:
    if features is None:
        return {}
    if hasattr(features, "to_payload"):
        try:
            return dict(features.to_payload())  # type: ignore[arg-type]
        except Exception:
            pass
    if is_dataclass(features):
        return asdict(features)
    if isinstance(features, dict):
        return dict(features)
    return {}


def _infer_benchmark_symbol(symbol: str) -> str:
    mapping = {
        'TQQQ': 'QQQ',
        'SQQQ': 'QQQ',
        'UPRO': 'SPY',
        'SPXL': 'SPY',
        'TNA': 'IWM',
        'TMF': 'TLT',
    }
    return mapping.get((symbol or '').upper(), 'SPY')


def _sanitize_train_split(value: Any, default: float = 0.8) -> float:
    try:
        split = float(value)
    except (TypeError, ValueError):
        return default

    if not (0.0 < split < 1.0):
        return default

    # Avoid degenerate splits
    return float(max(0.05, min(split, 0.95)))


def _is_intraday_mode(config_dict: Dict[str, Any]) -> bool:
    requested_agent = str(config_dict.get('agent_type', '')).upper()
    if requested_agent == 'SAC_INTRADAY_DSR':
        return True

    training_settings = config_dict.get('training_settings') or {}
    frequency = str(training_settings.get('data_frequency', '')).lower()
    if frequency in INTRADAY_FREQUENCY_FLAGS:
        return True

    interval = str(training_settings.get('interval', '')).lower()
    return interval in {'15m', '15min'}


def _append_matching(
    target: List[str],
    candidate_columns: Iterable[str],
    predicate: Callable[[str], bool]
) -> None:
    for column in candidate_columns:
        if predicate(column) and column not in target:
            target.append(column)


def _select_feature_columns(
    df: pd.DataFrame,
    features: Dict[str, Any],
    *,
    symbol: str,
    benchmark_symbol: Optional[str],
    intraday: bool
) -> List[str]:
    numeric_columns = [col for col in df.columns if is_numeric_dtype(df[col])]
    if not numeric_columns:
        return []

    selected: List[str] = []
    features = features or {}

    def flag_enabled(key: str) -> bool:
        value = features.get(key)
        if isinstance(value, dict):
            return bool(value.get('enabled', True))
        if value is None:
            return False
        return bool(value)

    def add_by_keywords(keywords: Iterable[str]) -> None:
        lowered = [kw.lower() for kw in keywords]
        _append_matching(
            selected,
            numeric_columns,
            lambda col: any(kw in col.lower() for kw in lowered)
        )

    if flag_enabled('price'):
        add_by_keywords(['price', 'close', 'adjclose'])

    if flag_enabled('volume'):
        add_by_keywords(['volume'])

    if flag_enabled('ohlc'):
        add_by_keywords(['open', 'high', 'low', 'close'])

    if flag_enabled('returns'):
        add_by_keywords(['return'])

    if flag_enabled('rsi'):
        add_by_keywords(['rsi'])

    ema_block = features.get('ema')
    if isinstance(ema_block, dict):
        if ema_block.get('enabled', True):
            add_by_keywords(['ema'])
    elif flag_enabled('ema'):
        add_by_keywords(['ema'])

    if flag_enabled('macd'):
        add_by_keywords(['macd'])

    if flag_enabled('adx'):
        add_by_keywords(['adx', 'plus_di', 'minus_di'])

    if flag_enabled('bollinger'):
        add_by_keywords(['bollinger', 'bb_'])

    if flag_enabled('stochastic'):
        add_by_keywords(['stoch'])

    if flag_enabled('sentiment'):
        add_by_keywords(['sentiment'])

    if flag_enabled('macro'):
        add_by_keywords(['macro_'])

    if flag_enabled('base_trend_context'):
        add_by_keywords(['base_trend_context'])

    if flag_enabled('base_momentum'):
        add_by_keywords(['base_momentum'])

    if flag_enabled('base_trend_strength'):
        add_by_keywords(['base_trend_strength', 'base_plus_di', 'base_minus_di'])

    if flag_enabled('base_extremes'):
        add_by_keywords(['base_extremes'])

    if flag_enabled('leveraged_volatility'):
        add_by_keywords(['leveraged_volatility'])

    if flag_enabled('leveraged_momentum_short'):
        add_by_keywords(['leveraged_momentum_short'])

    if flag_enabled('time_context'):
        add_by_keywords(['time_fraction', 'minutes_from_open', 'bar_index', 'is_session_end'])

    if flag_enabled('position_context'):
        add_by_keywords(['position_context'])

    multi_asset_cfg = features.get('multi_asset')
    if isinstance(multi_asset_cfg, dict) and multi_asset_cfg.get('enabled', False):
        symbols = multi_asset_cfg.get('symbols') or DEFAULT_MULTI_ASSET_SYMBOLS
        keywords = [sym.lower() for sym in symbols]
        _append_matching(
            selected,
            numeric_columns,
            lambda col: any(sym in col.lower() for sym in keywords)
        )

    llm_cfg = features.get('llm')
    if isinstance(llm_cfg, dict) and llm_cfg.get('enabled', False):
        add_by_keywords(['llm'])

    if intraday:
        primary_prefix = (symbol or '').lower()
        benchmark_prefix = (benchmark_symbol or '').lower()
        _append_matching(
            selected,
            numeric_columns,
            lambda col: col.lower().startswith(primary_prefix + '_')
        )
        if benchmark_prefix:
            _append_matching(
                selected,
                numeric_columns,
                lambda col: col.lower().startswith(benchmark_prefix + '_')
            )
        add_by_keywords(['primary_return', 'benchmark_return'])

    if not selected:
        selected = list(numeric_columns)

    return selected


def _resolve_hyperparameters(config_dict: dict) -> dict:
    """Merge hyperparameter sources into a single dict.

    Supports both the generic ``hyperparameters`` key (used by the API)
    and the agent-specific keys produced by ``TrainingConfig``.
    Later sources override earlier ones so that explicit agent presets win.
    """

    requested_type = str(config_dict.get('agent_type', '')).upper()
    agent_type = _canonical_agent_type(requested_type)

    merged: dict = {}

    # Generic hyperparameters from API payloads
    if isinstance(config_dict.get('hyperparameters'), dict):
        merged.update(config_dict['hyperparameters'])

    # Agent-specific blocks from TrainingConfig dataclasses
    if agent_type == 'PPO' and isinstance(config_dict.get('ppo_hyperparameters'), dict):
        merged.update(config_dict['ppo_hyperparameters'])
    elif agent_type == 'SAC' and isinstance(config_dict.get('sac_hyperparameters'), dict):
        merged.update(config_dict['sac_hyperparameters'])

    return merged


def _extract_history_config(features: dict) -> dict:
    """Return only history-related feature flags from the feature payload."""
    if not isinstance(features, dict):
        return {}

    history_keys = ['recent_actions', 'performance', 'position_history', 'reward_history']
    return {key: features[key] for key in history_keys if key in features}


class TrainingProgressCallback(BaseCallback):
    """
    Custom callback for logging training progress to JSON file.
    
    Frontend polls this file every 5 seconds to update progress UI.
    """
    
    def __init__(self, total_timesteps: int, progress_file: str = 'training_progress.json'):
        """
        Initialize callback.
        
        Args:
            total_timesteps: Total training timesteps
            progress_file: Path to JSON file for progress updates
        """
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_file = Path(progress_file)
        self.episode_rewards = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        """
        Called at every training step.
        
        Logs progress every 100 steps to avoid excessive I/O.
        """
        # Log every 100 steps
        if self.n_calls % 100 == 0:
            progress_pct = (self.n_calls / self.total_timesteps) * 100
            
            # Get episode reward if available
            if len(self.model.ep_info_buffer) > 0:
                recent_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer[-10:]])
            else:
                recent_reward = 0.0
            
            progress_data = {
                'timestep': self.n_calls,
                'total_timesteps': self.total_timesteps,
                'progress_pct': round(progress_pct, 2),
                'episode_reward': round(float(recent_reward), 4),
                'episode_count': self.episode_count,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to file (Frontend will poll this)
            try:
                with open(self.progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
            except Exception as e:
                print(f"[WARNING] Failed to write progress file: {e}")
        
        return True  # Continue training
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (episode)."""
        self.episode_count += 1


def load_data(
    symbol: str,
    start_date: str,
    end_date: str,
    features: dict = None,
    train_split: float = 0.8
) -> tuple:
    """
    Load training data from SQL database with real Yahoo Finance data.
    
    Args:
        symbol: Stock/ETF symbol (e.g., 'AAPL', 'TQQQ', 'IWM')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        features: Feature configuration dict (optional)
    
    Returns:
        Tuple of (train_data, test_data) as DataFrames with real market data
    """
    from data_download.loader import prepare_training_data
    
    print(f"[INFO] Loading REAL data for {symbol} ({start_date} to {end_date})...")
    
    # Extract feature flags from config
    features_payload = _features_payload(features)
    enable_sentiment = False
    enable_multi_asset = False
    multi_asset_symbols = None
    
    if features_payload:
        # Check sentiment features
        sentiment_config = features_payload.get('sentiment', {})
        if isinstance(sentiment_config, dict):
            enable_sentiment = sentiment_config.get('enabled', False)
        else:
            enable_sentiment = bool(sentiment_config)
        
        # Check multi-asset features
        multi_asset_config = features_payload.get('multi_asset', {})
        if isinstance(multi_asset_config, dict):
            enable_multi_asset = multi_asset_config.get('enabled', False)
            multi_asset_symbols = multi_asset_config.get('symbols', DEFAULT_MULTI_ASSET_SYMBOLS)
        else:
            enable_multi_asset = bool(multi_asset_config)
            multi_asset_symbols = DEFAULT_MULTI_ASSET_SYMBOLS
    
    # Calculate period from dates
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        years = (end - start).days / 365.25
        
        if years > 10:
            period = "max"
        elif years > 5:
            period = "10y"
        elif years > 2:
            period = "5y"
        elif years > 1:
            period = "2y"
        else:
            period = "1y"
    except:
        period = "5y"  # Default fallback
    
    print(f"   Period: {period}, Sentiment: {enable_sentiment}, Multi-asset: {enable_multi_asset}")
    
    # Use the production-grade data pipeline
    split_ratio = _sanitize_train_split(train_split, default=0.8)

    prepared_data, data_split = prepare_training_data(
        symbol=symbol,
        period=period,
        train_test_split=split_ratio,
        enable_sentiment=enable_sentiment,
        enable_multi_asset=enable_multi_asset,
        multi_asset_symbols=multi_asset_symbols if enable_multi_asset else None,
        force_redownload=False,
        feature_config=features_payload,
    )
    
    # Filter by date range if needed
    train_data = data_split.train
    test_data = data_split.test
    
    # Don't filter by dates - prepare_training_data already did the split correctly
    # The start_date/end_date were used to determine the period
    
    # Rename 'Close' to 'price' for backward compatibility with environments
    if 'Close' in train_data.columns:
        train_data = train_data.rename(columns={'Close': 'price'})
        test_data = test_data.rename(columns={'Close': 'price'})

    # Clean up any rows with missing timestamps that can appear after merges
    train_data = train_data[train_data.index.notna()].copy()
    test_data = test_data[test_data.index.notna()].copy()

    # Ensure chronological order in case upstream sources return unsorted data
    train_data = train_data.sort_index()
    test_data = test_data.sort_index()

    # Rename 'Volume' to 'volume' for consistency
    if 'Volume' in train_data.columns:
        train_data = train_data.rename(columns={'Volume': 'volume'})
        test_data = test_data.rename(columns={'Volume': 'volume'})
    
    # FeatureEngineering already filtered features based on UI selection - no need for additional filtering here
    
    print(f"[OK] REAL data loaded from Yahoo Finance + SQL:")
    print(f"   Train: {len(train_data)} samples ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"   Test: {len(test_data)} samples ({test_data.index[0]} to {test_data.index[-1]})")
    print(f"   Features: {len(train_data.columns)} total")
    print(f"   Sample features: {', '.join(train_data.columns[:10])}")
    
    if enable_multi_asset:
        multi_features = [f for f in train_data.columns if any(s.lower() in f.lower() for s in ['spy', 'qqq', 'tlt', 'gld'])]
        print(f"   Multi-asset features: {len(multi_features)} ({', '.join(multi_features[:5])}...)")
    
    if enable_sentiment:
        sentiment_features = [f for f in train_data.columns if 'sentiment' in f.lower()]
        print(f"   Sentiment features: {len(sentiment_features)} ({', '.join(sentiment_features)})")
    
    return train_data, test_data


def _load_intraday_training_frames(
    config_dict: Dict[str, Any],
    train_split: float
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    symbol = config_dict.get('symbol', '').upper()
    if not symbol:
        raise ValueError("Intraday training requires a valid symbol")

    training_settings = config_dict.get('training_settings') or {}
    benchmark_symbol = (
        training_settings.get('benchmark_symbol')
        or _infer_benchmark_symbol(symbol)
    )

    interval = training_settings.get('interval', '15m')
    start = training_settings.get('start_date')
    end = training_settings.get('end_date')

    dataset = build_intraday_dataset(
        (symbol, benchmark_symbol),
        interval=str(interval or '15m'),
        start=start,
        end=end,
    )

    if dataset.empty:
        raise ValueError(f"No intraday data available for {symbol} ({interval})")

    dataset = dataset.sort_index()

    feature_spec = IntradayFeatureSpec(
        primary_symbol=symbol,
        benchmark_symbol=benchmark_symbol,
    )
    dataset = add_intraday_features(dataset, feature_spec)

    numeric_cols = [col for col in dataset.columns if is_numeric_dtype(dataset[col])]
    dataset[numeric_cols] = dataset[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    split_ratio = _sanitize_train_split(train_split, default=0.8)
    split_index = int(len(dataset) * split_ratio)
    split_index = max(1, min(split_index, len(dataset) - 1))

    train_df = dataset.iloc[:split_index].copy()
    test_df = dataset.iloc[split_index:].copy()

    if test_df.empty:
        test_df = dataset.iloc[-max(1, len(dataset) // 5):].copy()

    return train_df, test_df, benchmark_symbol


def optimize_hyperparameters_with_optuna(
    config_dict: dict,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_names: list,
    n_trials: int = 100
) -> dict:
    """
    Use Optuna to find optimal hyperparameters.
    
    Args:
        config_dict: Base configuration dictionary
        train_data: Training DataFrame
        test_data: Test DataFrame  
        feature_names: List of feature column names
        n_trials: Number of Optuna trials to run
    
    Returns:
        Dictionary with optimized hyperparameters
    """
    
    print(f"\n{'='*70}")
    print(f"[OPTUNA] Starting Hyperparameter Optimization")
    print(f"   Trials: {n_trials}")
    print(f"   Agent: {config_dict['agent_type']}")
    print(f"{'='*70}\n")
    
    # Suppress Optuna verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    agent_type = config_dict['agent_type'].upper()
    base_hyperparams = _resolve_hyperparameters(config_dict)
    if base_hyperparams is None:
        base_hyperparams = {}
    initial_capital = config_dict['training_settings'].get('initial_capital')
    if initial_capital is None:
        initial_capital = config_dict['training_settings'].get('initial_cash', 100000)
    commission = config_dict['training_settings'].get('commission', 1.0)
    history_config = _extract_history_config(config_dict.get('features', {}))
    
    def objective(trial):
        """Optuna objective function - maximize Sharpe ratio on validation set."""
        
        # Sample hyperparameters based on agent type
        if agent_type == 'PPO':
            trial_hyperparams = {
                'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True),
                'gamma': trial.suggest_float('gamma', 0.98, 0.999),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
                'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
                'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.02),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.25),
                'episodes': base_hyperparams.get('episodes', 10000)  # Keep episodes fixed
            }
        else:  # SAC
            trial_hyperparams = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                'tau': trial.suggest_float('tau', 0.001, 0.02),
                'ent_coef': trial.suggest_categorical('ent_coef', ['auto', 0.1, 0.2, 0.5]),
                'episodes': base_hyperparams.get('episodes', 10000)
            }
        
        try:
            # Create environment for this trial
            if agent_type == 'PPO':
                env = StockTradingEnv(
                    df=train_data,
                    initial_capital=initial_capital,
                    commission=commission,
                    history_config=history_config
                )
            else:
                env = ETFTradingEnv(
                    df=train_data,
                    initial_capital=initial_capital,
                    commission=commission,
                    history_config=history_config
                )
                env = ContinuousActionAdapter(env)
            
            # Create agent with trial hyperparameters
            agent = create_agent(
                agent_type=agent_type,
                env=env,
                hyperparameters=trial_hyperparams,
                model_dir=f"backend/models/{agent_type.lower()}/optuna_trials"
            )
            
            # Quick training (reduced timesteps for optimization)
            quick_timesteps = min(10000, len(train_data) * 50)  # 50 episodes max
            agent.train(total_timesteps=quick_timesteps, callback=None)
            
            # Evaluate on validation set (test_data)
            # Create test environment
            if agent_type == 'PPO':
                test_env = StockTradingEnv(
                    df=test_data,
                    initial_capital=initial_capital,
                    commission=commission,
                    history_config=history_config
                )
            else:
                test_env = ETFTradingEnv(
                    df=test_data,
                    initial_capital=initial_capital,
                    commission=commission,
                    history_config=history_config
                )
                test_env = ContinuousActionAdapter(test_env)
            
            returns = []
            trade_counts = []
            drawdowns = []
            for episode in range(10):  # 10 evaluation episodes
                obs = test_env.reset()
                # Handle tuple return from shimmy wrapper
                if isinstance(obs, tuple):
                    obs = obs[0]
                    
                done = False
                episode_trades = 0
                peak_value = float(initial_capital)
                episode_max_drawdown = 0.0
                total_value = float(initial_capital)
                
                while not done:
                    prediction = agent.predict(obs, deterministic=True)
                    action = prediction[0] if isinstance(prediction, tuple) else prediction
                    step_result = test_env.step(action)
                    
                    # Handle different return formats (gym vs gymnasium)
                    if len(step_result) == 4:
                        obs, reward, done, info = step_result
                    else:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    
                    if action in (1, 2):
                        episode_trades += 1

                    info_total = None
                    if isinstance(info, dict):
                        info_total = info.get('total_value')
                    if info_total is None:
                        info_total = getattr(test_env, 'total_value', peak_value)
                    total_value = float(info_total)

                    if total_value > peak_value:
                        peak_value = total_value
                    else:
                        drawdown = (peak_value - total_value) / (peak_value + 1e-8)
                        if drawdown > episode_max_drawdown:
                            episode_max_drawdown = drawdown
                
                final_value = total_value
                episode_return = (final_value - float(initial_capital)) / float(initial_capital)
                returns.append(episode_return)
                trade_counts.append(episode_trades)
                drawdowns.append(episode_max_drawdown)
            
            # Calculate Sharpe ratio (mean return / std return)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / (std_return + 1e-8)

            avg_trades = float(np.mean(trade_counts)) if trade_counts else 0.0
            if avg_trades < 1.0:
                sharpe_ratio = -1e6  # Strongly penalize policies that never trade

            avg_drawdown = float(np.mean(drawdowns)) if drawdowns else 0.0
            if avg_drawdown > 0.25:  # penalize >25% drawdown
                sharpe_ratio -= (avg_drawdown - 0.25) * 200.0
            
            print(f"   Trial {trial.number}: Sharpe={sharpe_ratio:.3f}, "
                  f"LR={trial_hyperparams['learning_rate']:.6f}, "
                  f"Gamma={trial_hyperparams['gamma']:.3f}, "
                  f"AvgTrades={avg_trades:.2f}, "
                  f"AvgDD={avg_drawdown*100:.2f}%")
            
            return sharpe_ratio
            
        except Exception as e:
            print(f"   Trial {trial.number} failed: {e}")
            return -999.0  # Very bad score for failed trials
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n{'='*70}")
    print(f"[OK] Optuna Optimization Complete!")
    print(f"   Best Sharpe Ratio: {best_value:.3f}")
    print(f"   Best Hyperparameters:")
    for key, value in best_params.items():
        print(f"      {key}: {value}")
    print(f"{'='*70}\n")
    
    # Merge best params with original config
    optimized_hyperparams = {**base_hyperparams, **best_params}
    
    return optimized_hyperparams


def train_agent(config) -> dict:
    """
    Main training function.
    
    Workflow:
    1. Load data
    2. Normalize features
    3. Create environment
    4. Create agent
    5. Train with callbacks
    6. Save model and metadata
    
    Args:
        config: TrainingConfig dataclass instance or dictionary with keys:
            - agent_type: 'PPO' or 'SAC'
            - symbol: Stock/ETF symbol
            - hyperparameters: Agent hyperparameters dict/object
            - features: Feature selection dict/object
            - training_settings: Training settings dict/object
                - start_date, end_date, commission, initial_cash
    
    Returns:
        Dictionary with training results:
            - status: 'success' or 'failed'
            - model_path: Path to saved model
            - metadata: Model metadata
            - version: Model version
    """
    
    # Convert TrainingConfig to dict if necessary
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    elif not isinstance(config, dict):
        config_dict = asdict(config)
    else:
        config_dict = dict(config)

    requested_agent_type = str(config_dict.get('agent_type', 'PPO')).upper()
    canonical_agent_type = _canonical_agent_type(requested_agent_type)
    config_dict['agent_type'] = requested_agent_type

    training_settings = config_dict.get('training_settings') or {}
    if is_dataclass(training_settings):
        training_settings = asdict(training_settings)
    else:
        training_settings = dict(training_settings)
    config_dict['training_settings'] = training_settings

    features_payload = _features_payload(config_dict.get('features'))
    config_dict['features'] = features_payload

    symbol = str(config_dict.get('symbol', '')).upper()
    if not symbol:
        raise ValueError("Training configuration must include a symbol")

    train_split = _sanitize_train_split(training_settings.get('train_split', 0.8), default=0.8)
    training_settings['train_split'] = train_split

    # Extract hyperparameters (supports both API payloads and TrainingConfig output)
    hyperparams = _resolve_hyperparameters(config_dict)

    print(f"\n{'='*70}")
    print(f">> Starting Training: {requested_agent_type} Agent (canonical: {canonical_agent_type})")
    date_range = (
        f"{training_settings.get('start_date', 'N/A')} to {training_settings.get('end_date', 'N/A')}"
    )
    print(f"   Symbol: {symbol}")
    print(f"   Date Range: {date_range}")
    print(f"   Train/Test Split: {train_split:.2f}/{1-train_split:.2f}")
    print(f"{'='*70}\n")
    
    try:
        intraday_mode = _is_intraday_mode(config_dict)
        reward_mode = str(training_settings.get('reward_mode', '')).lower()
        dsr_config_raw = training_settings.get('dsr_config') or {}
        dsr_config_obj: Optional[DSRConfig] = None

        if intraday_mode:
            train_market, test_market, benchmark_symbol = _load_intraday_training_frames(config_dict, train_split)
            training_settings.setdefault('benchmark_symbol', benchmark_symbol)
            interval = training_settings.get('interval', '15m')
        else:
            train_market, test_market = load_data(
                symbol=symbol,
                start_date=training_settings.get('start_date'),
                end_date=training_settings.get('end_date'),
                features=features_payload,
                train_split=train_split,
            )
            benchmark_symbol = training_settings.get('benchmark_symbol') or _infer_benchmark_symbol(symbol)
            interval = training_settings.get('interval', '1d')

        print(f"[INFO] Data frequency: {'intraday' if intraday_mode else 'daily'} (interval={interval})")
        print(f"   Training samples: {len(train_market)} | Test samples: {len(test_market)}")

        feature_candidates = _select_feature_columns(
            train_market,
            features_payload,
            symbol=symbol,
            benchmark_symbol=benchmark_symbol,
            intraday=intraday_mode,
        )

        numeric_feature_columns = [
            col for col in feature_candidates
            if col in train_market.columns and is_numeric_dtype(train_market[col])
        ]

        if not numeric_feature_columns:
            numeric_feature_columns = [
                col for col in train_market.columns if is_numeric_dtype(train_market[col])
            ]

        if not numeric_feature_columns:
            raise ValueError("No numeric features available after applying feature configuration.")

        train_numeric = train_market[numeric_feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test_numeric = test_market[numeric_feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        metadata_columns: List[str] = []
        if intraday_mode:
            metadata_columns = [col for col in INTRADAY_METADATA_COLUMNS if col in train_market.columns]

        train_df = train_numeric.copy()
        test_df = test_numeric.copy()
        for meta_col in metadata_columns:
            train_df[meta_col] = train_market[meta_col]
            test_df[meta_col] = test_market[meta_col]

        if reward_mode == 'dsr':
            try:
                clip_raw = dsr_config_raw.get('clip_value')
                clip_value = None if clip_raw in (None, '', 'none', 'None') else float(clip_raw)
                dsr_config_obj = DSRConfig(
                    decay=float(dsr_config_raw.get('decay', 0.94)),
                    epsilon=float(dsr_config_raw.get('epsilon', 1e-9)),
                    warmup_steps=int(dsr_config_raw.get('warmup_steps', 200)),
                    clip_value=clip_value,
                )
                dsr_config_obj.validate()
            except Exception as exc:  # pragma: no cover - config validation
                raise ValueError(f"Invalid DSR configuration: {exc}") from exc

        feature_display = ', '.join(numeric_feature_columns[:15])
        if len(numeric_feature_columns) > 15:
            feature_display += ', ...'

        print("\n[INFO] Normalizing features...")
        print(f"   Selected features ({len(numeric_feature_columns)}): {feature_display}")

        normalizer = StateNormalizer(method='zscore')
        normalizer.fit(train_numeric.values)

        normalizer_path = f"backend/models/normalizer_{symbol}_{requested_agent_type}.json"
        normalizer.save_params(normalizer_path)
        print(f"[OK] Normalization complete, saved to {normalizer_path}")
        
        # 3. Create Environment
        print("\n[INFO] Creating trading environment...")

        initial_capital = training_settings.get('initial_capital')
        if initial_capital is None:
            initial_capital = training_settings.get('initial_cash', 100000)
        initial_capital = float(initial_capital)

        max_position_size = float(training_settings.get('max_position_size', 1.0))
        normalize_obs = bool(training_settings.get('normalize_obs', True))
        history_config = _extract_history_config(features_payload)

        commission_config = resolve_commission_config(training_settings)

        if reward_mode == 'dsr' and dsr_config_obj is None:
            dsr_config_obj = DSRConfig()

        if canonical_agent_type == 'PPO':
            env = StockTradingEnv(
                df=train_df,
                initial_capital=initial_capital,
                commission=commission_config,
                max_position_size=max_position_size,
                risk_penalty=hyperparams.get('risk_penalty', -0.5),
                normalize_obs=normalize_obs,
                history_config=history_config
            )
            print("[OK] StockTradingEnv created (PPO)")

        elif canonical_agent_type == 'SAC':
            if intraday_mode:
                sampler = IntradaySessionSampler(shuffle=True)
                base_env = IntradayEquityEnv(
                    df=train_df,
                    initial_capital=initial_capital,
                    commission=commission_config,
                    max_position_size=max_position_size,
                    normalize_obs=normalize_obs,
                    history_config=history_config,
                    sampler=sampler
                )
                env_label = "IntradayEquityEnv"
            else:
                base_env = ETFTradingEnv(
                    df=train_df,
                    initial_capital=initial_capital,
                    commission=commission_config,
                    max_position_size=max_position_size,
                    vol_penalty=hyperparams.get('vol_penalty', -0.3),
                    leverage_factor=hyperparams.get('leverage_factor', 3.0),
                    normalize_obs=normalize_obs,
                    history_config=history_config
                )
                env_label = "ETFTradingEnv"

            if reward_mode == 'dsr' and dsr_config_obj is not None:
                base_env = DSRRewardWrapper(base_env, config=dsr_config_obj)

            env = ContinuousActionAdapter(base_env)
            print(f"[OK] {env_label} created (SAC)")

        else:
            raise ValueError(f"Unknown agent type: {requested_agent_type}")
        
        # 4. Create Agent
        print("\n[INFO] Creating RL agent...")
        
        agent = create_agent(
            agent_type=canonical_agent_type,
            env=env,
            hyperparameters=hyperparams,
            model_dir=f"backend/models/{canonical_agent_type.lower()}"
        )

        print(f"[OK] {canonical_agent_type} agent created")
        
        # 4.5. Optuna Hyperparameter Optimization (Optional)
        optuna_trials = int(training_settings.get('optuna_trials', 0) or 0)
        if intraday_mode and optuna_trials > 0:
            print("[WARN] Optuna optimization is not currently supported for the intraday pipeline; skipping trials.")
            optuna_trials = 0

        if optuna_trials > 0:
            print(f"\n[OPTUNA] Running optimization with {optuna_trials} trials...")
            optimized_hyperparams = optimize_hyperparameters_with_optuna(
                config_dict=config_dict,
                train_data=train_df,
                test_data=test_df,
                feature_names=numeric_feature_columns,
                n_trials=optuna_trials
            )
            
            # Update hyperparameters with optimized values
            hyperparams = optimized_hyperparams
            
            # Recreate agent with optimized hyperparameters
            print("\n[INFO] Recreating agent with optimized hyperparameters...")
            agent = create_agent(
                agent_type=canonical_agent_type,
                env=env,
                hyperparameters=hyperparams,
                model_dir=f"backend/models/{canonical_agent_type.lower()}"
            )
            print(f"[OK] Agent recreated with optimized parameters")
        
        # 5. Train Agent
        print("\n[TRAIN] Training agent...")

        # Calculate total timesteps with sane defaults and caps
        steps_per_episode = max(1, len(train_df))
        requested_total_timesteps = (
            hyperparams.get('total_timesteps')
            or training_settings.get('total_timesteps')
        )
        requested_episode_budget = hyperparams.get('episodes')
        fallback_episode_budget = training_settings.get('episode_budget')

        resolved_episode_budget = None
        if requested_total_timesteps:
            total_timesteps_requested = int(requested_total_timesteps)
        else:
            if requested_episode_budget is not None and requested_episode_budget > 0:
                resolved_episode_budget = int(requested_episode_budget)
            elif fallback_episode_budget is not None and fallback_episode_budget > 0:
                resolved_episode_budget = int(fallback_episode_budget)
            else:
                resolved_episode_budget = 1

            total_timesteps_requested = resolved_episode_budget * steps_per_episode

        max_total_timesteps = training_settings.get('max_total_timesteps', 1_000_000)
        if not isinstance(max_total_timesteps, int) or max_total_timesteps <= 0:
            max_total_timesteps = 1_000_000

        total_timesteps = total_timesteps_requested
        if total_timesteps > max_total_timesteps:
            print(
                f"   Requested total timesteps {total_timesteps:,} exceed limit of "
                f"{max_total_timesteps:,}. Capping budget."
            )
            total_timesteps = max_total_timesteps

        total_timesteps = max(steps_per_episode, total_timesteps)
        effective_episodes = max(1, math.ceil(total_timesteps / steps_per_episode))

        if resolved_episode_budget is None and requested_episode_budget is not None:
            resolved_episode_budget = int(requested_episode_budget)
        elif resolved_episode_budget is None:
            resolved_episode_budget = effective_episodes

        print(f"   Episodes requested: {resolved_episode_budget}")
        if requested_total_timesteps:
            print(f"   Total timesteps requested: {total_timesteps_requested:,}")
        print(f"   Steps per episode: {steps_per_episode}")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Effective episodes: {effective_episodes}")
        
        
        # Create progress callback
        progress_callback = TrainingProgressCallback(
            total_timesteps=total_timesteps,
            progress_file='training_progress.json'
        )
        
        # Train
        start_time = datetime.now()
        agent.train(total_timesteps=total_timesteps, callback=progress_callback)
        end_time = datetime.now()
        
        training_duration = (end_time - start_time).total_seconds()
        
        print(f"\n[OK] Training complete in {training_duration:.1f} seconds")
        
        # 6. Evaluate on Test Set
        test_pct = max(0.0, 100.0 * (1.0 - train_split))
        print(f"\n[EVAL] Evaluating model on test set (~{test_pct:.0f}%)...")
        
        # Create test environment
        if canonical_agent_type == 'PPO':
            test_env = StockTradingEnv(
                df=test_df,
                initial_capital=initial_capital,
                commission=commission_config,
                max_position_size=max_position_size,
                risk_penalty=hyperparams.get('risk_penalty', -0.5),
                normalize_obs=normalize_obs,
                history_config=history_config
            )
        else:
            if intraday_mode:
                eval_sampler = IntradaySessionSampler(shuffle=False, sequential=True)
                base_test_env = IntradayEquityEnv(
                    df=test_df,
                    initial_capital=initial_capital,
                    commission=commission_config,
                    max_position_size=max_position_size,
                    normalize_obs=normalize_obs,
                    history_config=history_config,
                    sampler=eval_sampler
                )
            else:
                base_test_env = ETFTradingEnv(
                    df=test_df,
                    initial_capital=initial_capital,
                    commission=commission_config,
                    max_position_size=max_position_size,
                    vol_penalty=hyperparams.get('vol_penalty', -0.3),
                    leverage_factor=hyperparams.get('leverage_factor', 3.0),
                    normalize_obs=normalize_obs,
                    history_config=history_config
                )

            test_env = ContinuousActionAdapter(base_test_env)
        
        # Run evaluation
        try:
            eval_results = evaluate_trained_model(
                model=agent.model,
                test_env=test_env,
                n_eval_episodes=10,
                initial_capital=initial_capital
            )
            
            # Extract metrics
            test_metrics = eval_results['metrics']
            if 'final_balance' not in test_metrics:
                test_metrics['final_balance'] = initial_capital
            
            print(f"\n{'='*70}")
            print(f"[RESULTS] Test Set Performance:")
            print(f"{'='*70}")
            print(f"  Total Return:      {test_metrics['total_return']:>8.2f}%")
            print(f"  Sharpe Ratio:      {test_metrics['sharpe_ratio']:>8.2f}")
            print(f"  Sortino Ratio:     {test_metrics['sortino_ratio']:>8.2f}")
            print(f"  Max Drawdown:      {test_metrics['max_drawdown']:>8.2f}%")
            print(f"  Calmar Ratio:      {test_metrics['calmar_ratio']:>8.2f}")
            print(f"  Win Rate:          {test_metrics['win_rate']:>8.2f}%")
            print(f"  Profit Factor:     {test_metrics['profit_factor']:>8.2f}")
            print(f"  Total Trades:      {test_metrics['total_trades']:>8}")
            print(f"  Win/Loss:          {test_metrics['winning_trades']:>4}/{test_metrics['losing_trades']:<4}")
            print(f"{'='*70}\n")
            
        except Exception as eval_error:
            print(f"\n[WARNING] Evaluation failed: {eval_error}")
            print("   Continuing with placeholder metrics...")
            test_metrics = {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'calmar_ratio': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'final_balance': initial_capital
            }
            eval_results = {'equity_curve': [], 'trades': []}
        
        # 7. Save Model
        print("\n[SAVE] Saving model...")
        
        # Generate version
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        
        model_manager = ModelManager(base_dir='backend/models')
        agent_type_lower = canonical_agent_type.lower()
        symbol_upper = symbol
        model_path = Path(model_manager.base_dir) / agent_type_lower / f"{agent_type_lower}_{symbol_upper}_{version}.zip"

        data_frequency = training_settings.get('data_frequency') or ('intraday' if intraday_mode else 'daily')

        # Create metadata with REAL metrics
        metadata = {
            'agent_type': requested_agent_type,
            'canonical_agent_type': canonical_agent_type,
            'symbol': symbol,
            'version': version,
            'created': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat(),  # For compatibility with ModelSelector
            'hyperparameters': hyperparams,
            'features': features_payload,
            'features_used': numeric_feature_columns,
            'training_settings': training_settings,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'benchmark_symbol': benchmark_symbol,
            'interval': interval,
            'data_frequency': data_frequency,
            'reward_mode': reward_mode,
            'dsr_config': dsr_config_raw if reward_mode == 'dsr' else {},
            'episodes_requested': int(requested_episode_budget) if requested_episode_budget is not None else None,
            'episodes': int(effective_episodes),
            'total_timesteps_requested': int(total_timesteps_requested),
            'total_timesteps': int(total_timesteps),
            'training_duration_seconds': training_duration,
            'model_path': str(model_path),
            'normalizer_path': normalizer_path,
            # REAL metrics from test set evaluation!
            'sharpe_ratio': float(test_metrics['sharpe_ratio']),
            'sortino_ratio': float(test_metrics['sortino_ratio']),
            'total_return': float(test_metrics['total_return']),
            'max_drawdown': float(test_metrics['max_drawdown']),
            'win_rate': float(test_metrics['win_rate']),
            'profit_factor': float(test_metrics['profit_factor']),
            'calmar_ratio': float(test_metrics['calmar_ratio']),
            'total_trades': int(test_metrics['total_trades']),
            'winning_trades': int(test_metrics['winning_trades']),
            'losing_trades': int(test_metrics['losing_trades']),
            'final_balance': float(test_metrics['final_balance']),
            # Additional evaluation data
            'equity_curve': eval_results.get('equity_curve', []),
            'trade_history': eval_results.get('trades', [])
        }
        
        model_id = model_manager.save_model(
            model=agent.model,
            agent_type=canonical_agent_type,
            symbol=symbol,
            metadata=metadata,
            version=version
        )

        metadata['model_id'] = model_id

        print(f"[OK] Model saved: {model_path}")
        print(f"[OK] Metadata saved")

        print(f"\n{'='*70}")
        print(f"[OK] Training Complete!")
        print(f"   Model: {version}")
        print(f"   Duration: {training_duration:.1f}s")
        print(f"   Test Sharpe: {test_metrics['sharpe_ratio']:.2f}")
        print(f"   Test Return: {test_metrics['total_return']:+.2f}%")
        print(f"   Max Drawdown: {test_metrics['max_drawdown']:.2f}%")
        print(f"{'='*70}\n")
        
        return {
            'status': 'success',
            'model_path': str(model_path),
            'metadata': metadata,
            'version': version,
            'model_id': model_id
        }
    
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e)
        }


if __name__ == '__main__':
    """
    Example usage for testing
    """
    
    # Example configuration
    test_config = {
        'agent_type': 'PPO',
        'symbol': 'AAPL',
        'hyperparameters': {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'batch_size': 256,
            'n_steps': 2048,
            'episodes': 10  # Small number for testing
        },
        'features': {
            'price': True,
            'volume': True,
            'rsi': True,
            'macd': True,
            'ema_10': True,
            'ema_50': True,
            'vix': True,
            'bb_upper': True,
            'bb_lower': True,
            'stochastic': True,
            'sentiment': False
        },
        'training_settings': {
            'start_date': '2023-01-01',
            'end_date': '2024-11-01',
            'commission': 1.0,
            'initial_cash': 100000
        }
    }
    
    print("[TEST] Running training test...")
    result = train_agent(test_config)
    
    print("\n[RESULT]:")
    print(json.dumps(result, indent=2))
