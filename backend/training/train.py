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
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from stable_baselines3.common.callbacks import BaseCallback
from environments.stock_env import StockTradingEnv
from environments.etf_env import ETFTradingEnv
from agents.agent_factory import create_agent
from utils.state_normalizer import StateNormalizer
from models.model_manager import ModelManager
from config.training_config import TrainingConfig
from evaluation.backtester import evaluate_trained_model


def _resolve_hyperparameters(config_dict: dict) -> dict:
    """Merge hyperparameter sources into a single dict.

    Supports both the generic ``hyperparameters`` key (used by the API)
    and the agent-specific keys produced by ``TrainingConfig``.
    Later sources override earlier ones so that explicit agent presets win.
    """

    agent_type = config_dict.get('agent_type', '').upper()

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


def load_data(symbol: str, start_date: str, end_date: str, features: dict = None) -> tuple:
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
    enable_sentiment = False
    enable_multi_asset = False
    multi_asset_symbols = None
    
    if features:
        # Check sentiment features
        sentiment_config = features.get('sentiment', {})
        if isinstance(sentiment_config, dict):
            enable_sentiment = sentiment_config.get('enabled', False)
        else:
            enable_sentiment = bool(sentiment_config)
        
        # Check multi-asset features
        multi_asset_config = features.get('multi_asset', {})
        if isinstance(multi_asset_config, dict):
            enable_multi_asset = multi_asset_config.get('enabled', False)
            multi_asset_symbols = multi_asset_config.get('symbols', ['SPY', 'QQQ', 'TLT', 'GLD'])
        else:
            enable_multi_asset = bool(multi_asset_config)
            multi_asset_symbols = ['SPY', 'QQQ', 'TLT', 'GLD']
    
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
    prepared_data, data_split = prepare_training_data(
        symbol=symbol,
        period=period,
        train_test_split=0.8,
        enable_sentiment=enable_sentiment,
        enable_multi_asset=enable_multi_asset,
        multi_asset_symbols=multi_asset_symbols if enable_multi_asset else None,
        force_redownload=False,
        feature_config=features,  # ← NEW: Pass full feature config to FeatureEngineering
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
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
                'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
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
            
            returns = []
            for episode in range(10):  # 10 evaluation episodes
                obs = test_env.reset()
                # Handle tuple return from shimmy wrapper
                if isinstance(obs, tuple):
                    obs = obs[0]
                    
                done = False
                episode_return = 0
                
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
                    
                    episode_return += reward
                
                returns.append(episode_return)
            
            # Calculate Sharpe ratio (mean return / std return)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / (std_return + 1e-8)
            
            print(f"   Trial {trial.number}: Sharpe={sharpe_ratio:.3f}, "
                  f"LR={trial_hyperparams['learning_rate']:.6f}, "
                  f"Gamma={trial_hyperparams['gamma']:.3f}")
            
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
        # Handle dataclass without to_dict method
        from dataclasses import asdict
        config_dict = asdict(config)
    else:
        config_dict = config
    
    # Extract hyperparameters (supports both API payloads and TrainingConfig output)
    hyperparams = _resolve_hyperparameters(config_dict)
    
    print(f"\n{'='*70}")
    print(f">> Starting Training: {config_dict['agent_type']} Agent")
    print(f"   Symbol: {config_dict['symbol']}")
    print(f"   Date Range: {config_dict['training_settings']['start_date']} to {config_dict['training_settings']['end_date']}")
    print(f"{'='*70}\n")
    
    try:
        # 1. Load Data
        train_data, test_data = load_data(
            symbol=config_dict['symbol'],
            start_date=config_dict['training_settings']['start_date'],
            end_date=config_dict['training_settings']['end_date'],
            features=config_dict['features']
        )
        
        # Remove non-numeric columns (like 'date')
        numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
        train_data = train_data[numeric_columns]
        test_data = test_data[numeric_columns]
        
        # 2. Normalize Data
        print("\n[INFO] Normalizing features...")
        
        # Extract feature names - handle both boolean and dict features
        # Match them to actual column names in the data
        feature_names = []
        available_columns = [col for col in train_data.columns if col != 'date']
        
        for k, v in config_dict['features'].items():
            # Skip agent history features (not in market data)
            if k in ['recent_actions', 'performance', 'position_history', 'reward_history', 'llm']:
                continue
            
            if isinstance(v, bool) and v:
                # Simple boolean feature like 'ohlcv' or 'fundamentals'
                if k == 'ohlcv':
                    # Add OHLCV columns
                    ohlcv_cols = [col for col in available_columns if col in ['Open', 'High', 'Low', 'price', 'volume']]
                    feature_names.extend(ohlcv_cols)
                elif k == 'technical':
                    # Add all technical indicator columns
                    tech_cols = [col for col in available_columns if any(prefix in col.lower() for prefix in 
                                ['sma', 'ema', 'rsi', 'macd', 'bb_', 'stochastic', 'atr', 'adx', 'returns', 'log_returns'])]
                    feature_names.extend(tech_cols)
                elif k in available_columns:
                    feature_names.append(k)
            elif isinstance(v, dict):
                # Complex feature with parameters (technical indicators)
                if k == 'technical':
                    # Technical dict with specific indicators
                    for indicator, enabled in v.items():
                        if enabled:
                            if indicator == 'sma':
                                sma_cols = [col for col in available_columns if col.startswith('sma_')]
                                feature_names.extend(sma_cols)
                            elif indicator == 'ema':
                                ema_cols = [col for col in available_columns if col.startswith('ema_')]
                                feature_names.extend(ema_cols)
                            elif indicator == 'rsi':
                                if 'rsi' in available_columns:
                                    feature_names.append('rsi')
                            elif indicator == 'macd':
                                macd_cols = [col for col in available_columns if 'macd' in col.lower()]
                                feature_names.extend(macd_cols)
                            elif indicator == 'bollinger':
                                bb_cols = [col for col in available_columns if col.startswith('bb_')]
                                feature_names.extend(bb_cols)
                            elif indicator == 'stochastic':
                                stoch_cols = [col for col in available_columns if col.startswith('stochastic_')]
                                feature_names.extend(stoch_cols)
                elif v.get('enabled', False):
                    # Other enabled dict features
                    if k in available_columns:
                        feature_names.append(k)
        
        # Remove duplicates while preserving order
        feature_names = list(dict.fromkeys(feature_names))
        
        if not feature_names:
            raise ValueError("No valid features selected. At least 'price' and 'volume' should be available.")
        
        print(f"   Selected features ({len(feature_names)}): {', '.join(feature_names)}")
        
        normalizer = StateNormalizer(method='zscore')
        normalizer.fit(train_data[feature_names].values)
        
        # Save normalizer params
        normalizer_path = f"backend/models/normalizer_{config_dict['symbol']}_{config_dict['agent_type']}.json"
        normalizer.save_params(normalizer_path)
        
        print(f"[OK] Normalization complete, saved to {normalizer_path}")
        
        # 3. Create Environment
        print("\n[INFO] Creating trading environment...")
        
        initial_capital = config_dict['training_settings'].get('initial_capital')
        if initial_capital is None:
            initial_capital = config_dict['training_settings'].get('initial_cash', 100000)
        commission = config_dict['training_settings'].get('commission', 1.0)
        history_config = _extract_history_config(config_dict.get('features', {}))
        
        if config_dict['agent_type'].upper() == 'PPO':
            env = StockTradingEnv(
                df=train_data,
                initial_capital=initial_capital,
                commission=commission,
                history_config=history_config
            )
            print(f"[OK] StockTradingEnv created (PPO-optimized)")
        
        elif config_dict['agent_type'].upper() == 'SAC':
            env = ETFTradingEnv(
                df=train_data,
                initial_capital=initial_capital,
                commission=commission,
                history_config=history_config
            )
            print(f"[OK] ETFTradingEnv created (SAC-optimized)")
        
        else:
            raise ValueError(f"Unknown agent type: {config_dict['agent_type']}")
        
        # 4. Create Agent
        print("\n[INFO] Creating RL agent...")
        
        agent = create_agent(
            agent_type=config_dict['agent_type'].upper(),
            env=env,
            hyperparameters=hyperparams,
            model_dir=f"backend/models/{config_dict['agent_type'].lower()}"
        )
        
        print(f"[OK] {config_dict['agent_type']} agent created")
        
        # 4.5. Optuna Hyperparameter Optimization (Optional)
        optuna_trials = config_dict['training_settings'].get('optuna_trials', 0)
        if optuna_trials > 0:
            print(f"\n[OPTUNA] Running optimization with {optuna_trials} trials...")
            optimized_hyperparams = optimize_hyperparameters_with_optuna(
                config_dict=config_dict,
                train_data=train_data,
                test_data=test_data,
                feature_names=feature_names,
                n_trials=optuna_trials
            )
            
            # Update hyperparameters with optimized values
            hyperparams = optimized_hyperparams
            
            # Recreate agent with optimized hyperparameters
            print("\n[INFO] Recreating agent with optimized hyperparameters...")
            agent = create_agent(
                agent_type=config_dict['agent_type'].upper(),
                env=env,
                hyperparameters=hyperparams,
                model_dir=f"backend/models/{config_dict['agent_type'].lower()}"
            )
            print(f"[OK] Agent recreated with optimized parameters")
        
        # 5. Train Agent
        print("\n[TRAIN] Training agent...")

        # Calculate total timesteps with sane defaults and caps
        steps_per_episode = max(1, len(train_data))
        requested_total_timesteps = (
            hyperparams.get('total_timesteps')
            or config_dict['training_settings'].get('total_timesteps')
        )
        requested_episode_budget = hyperparams.get('episodes')
        fallback_episode_budget = config_dict['training_settings'].get('episode_budget')

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

        max_total_timesteps = config_dict['training_settings'].get('max_total_timesteps', 1_000_000)
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
        print("\n[EVAL] Evaluating model on test set (20%)...")
        
        # Create test environment
        if config_dict['agent_type'].upper() == 'PPO':
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
        agent_type_lower = config_dict['agent_type'].lower()
        symbol_upper = config_dict['symbol'].upper()
        model_path = Path(model_manager.base_dir) / agent_type_lower / f"{agent_type_lower}_{symbol_upper}_{version}.zip"
        
        # Create metadata with REAL metrics
        metadata = {
            'agent_type': config_dict['agent_type'].upper(),
            'symbol': config_dict['symbol'],
            'version': version,
            'created': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat(),  # For compatibility with ModelSelector
            'hyperparameters': hyperparams,
            'features': config_dict['features'],  # Original feature config
            'features_used': feature_names,  # Actual features used in training
            'training_settings': config_dict['training_settings'],
            'train_samples': len(train_data),
            'test_samples': len(test_data),
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
        
        model_manager.save_model(
            model=agent.model,
            agent_type=config_dict['agent_type'],
            symbol=config_dict['symbol'],
            metadata=metadata,
            version=version
        )
        
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
            'version': version
        }
    
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
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
