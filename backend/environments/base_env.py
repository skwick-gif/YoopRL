"""
base_env.py
Base Trading Environment for RL Agents

Purpose:
- Abstract base class for all trading environments
- Implements Gym interface (compatible with Stable-Baselines3)
- Defines state space, action space, and reward structure
- Provides normalization and common utilities

Why separate file:
- Shared logic for StockEnv and ETFEnv
- Enforces consistent interface across environments
- Easy to extend for new asset types
- Single source of truth for environment structure

State Space (observation):
- Portfolio state: cash, holdings, total_value, position_size
- Price features: current_price, returns, volatility
- Technical indicators: RSI, MACD, EMA, etc.
- Agent history: recent actions, performance metrics

Action Space:
- Discrete(3): [HOLD, BUY, SELL]
- Continuous alternative: action in [-1, 1] (sell to buy)

Wiring:
- Subclasses (StockEnv, ETFEnv) implement _calculate_reward()
- Used by RL agents (PPO, SAC) during training
- Receives data from feature_engineering.py
"""

import gym
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union


class BaseTradingEnv(gym.Env, ABC):
    """
    Abstract base environment for stock/ETF trading.
    
    Follows OpenAI Gym interface for compatibility with Stable-Baselines3.
    """
    
    metadata = {'render.modes': ['human']}
    
    DEFAULT_HISTORY_FLAGS = {
        'recent_actions': True,
        'performance': True,
        'position_history': True,
        'reward_history': False,
    }

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000.0,
    commission: Union[float, Dict[str, float]] = 1.0,
        max_position_size: float = 1.0,
        normalize_obs: bool = True,
        history_config: Optional[Dict] = None
    ):
        """
        Initialize base trading environment.
        
        Args:
            df: DataFrame with OHLCV + features (from feature_engineering.py)
            initial_capital: Starting portfolio value ($)
            commission: Trading fee per transaction ($)
            max_position_size: Max fraction of portfolio per trade (0-1)
            normalize_obs: Whether to normalize observations
        """
        super(BaseTradingEnv, self).__init__()
        
        # Data
        self.df = df.reset_index(drop=True)
        self.n_steps = len(df)
        
        # Trading parameters
        self.initial_capital = initial_capital
        self._commission_config: Optional[Dict[str, float]] = None
        self._commission_value: float = 0.0

        if isinstance(commission, dict):
            self._commission_config = {**commission}
            self.commission = self._commission_config  # Preserve attribute for external access
        else:
            self._commission_value = float(commission)
            self.commission = self._commission_value
        self.max_position_size = max_position_size
        self.normalize_obs = normalize_obs
        
        # Portfolio state
        self.current_step = 0
        self.cash = initial_capital
        self.holdings = 0  # Number of shares/units held
        self.total_value = initial_capital
        self.position_value = 0
        
        # History tracking
        self.action_history = []
        self.portfolio_history = []
        self.position_history = []
        self.reward_history = []
        self._prev_holdings = 0

        # History feature configuration
        self.history_config = history_config or {}
        self.history_settings = self._build_history_settings()
        
        # Define action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = gym.spaces.Discrete(3)
        
        # Define observation space (will be set by subclass based on features)
        self._set_observation_space()
        
        # Normalization parameters (calculated on first reset)
        self.obs_mean = None
        self.obs_std = None
    
    def _set_observation_space(self):
        """
        Set observation space dimensions based on DataFrame features.
        
        Observation includes:
        - Portfolio state (4): cash_ratio, holdings_ratio, position_ratio, total_value_normalized
        - Price features (N): close, returns, volatility, etc.
        - Technical indicators (M): RSI, MACD, EMA, etc.
        - Agent history (K): recent actions, performance
        """
        # Count feature columns (exclude 'date' and basic OHLCV if needed)
        n_features = len(self.df.columns)
        n_portfolio_state = 4
        history_dim = 0

        if self.history_settings['recent_actions']['enabled']:
            history_dim += self.history_settings['recent_actions']['length']

        if self.history_settings['performance']['enabled']:
            history_dim += self.history_settings['performance']['length']

        if self.history_settings['position_history']['enabled']:
            history_dim += self.history_settings['position_history']['length']

        if self.history_settings['reward_history']['enabled']:
            history_dim += self.history_settings['reward_history']['length']

        obs_dim = n_portfolio_state + n_features + history_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _history_enabled(self, feature: str) -> bool:
        """Determine if a history-based feature should be included."""
        default = self.DEFAULT_HISTORY_FLAGS.get(feature, False)
        value = self.history_config.get(feature, default)

        if isinstance(value, dict):
            return value.get('enabled', default)

        if value is None:
            return default

        return bool(value)

    def _history_length(self, feature: str, default: int) -> int:
        """Extract desired window length for sequence-based history features."""
        length = default
        value = self.history_config.get(feature)

        if isinstance(value, dict):
            for key in ('length', 'window', 'steps'):
                if key in value and isinstance(value[key], (int, float)):
                    length = int(value[key])
                    break
        elif isinstance(value, (int, float)):
            length = int(value)

        return max(1, min(length, 100))

    def _parse_performance_window(self) -> int:
        """Parse performance window (in steps) from config."""
        default_window = 30
        value = self.history_config.get('performance', {})

        if isinstance(value, dict):
            raw_period = value.get('period')
        else:
            raw_period = value

        if isinstance(raw_period, str):
            digits = ''.join(ch for ch in raw_period if ch.isdigit())
            if digits:
                default_window = int(digits)
        elif isinstance(raw_period, (int, float)) and raw_period > 0:
            default_window = int(raw_period)

        return max(2, min(default_window, 252))

    def _build_history_settings(self) -> Dict[str, Dict[str, int]]:
        """Construct normalized history feature configuration."""
        performance_enabled = self._history_enabled('performance')

        settings = {
            'recent_actions': {
                'enabled': self._history_enabled('recent_actions'),
                'length': self._history_length('recent_actions', 5)
            },
            'performance': {
                'enabled': performance_enabled,
                'window': self._parse_performance_window() if performance_enabled else 2,
                'length': 4  # cumulative_return, avg_return, volatility, max_drawdown
            },
            'position_history': {
                'enabled': self._history_enabled('position_history'),
                'length': self._history_length('position_history', 5)
            },
            'reward_history': {
                'enabled': self._history_enabled('reward_history'),
                'length': self._history_length('reward_history', 5)
            }
        }

        return settings

    def _get_history_features(self) -> np.ndarray:
        """Assemble history-based features according to configuration."""
        segments = []

        recent_cfg = self.history_settings['recent_actions']
        if recent_cfg['enabled']:
            length = recent_cfg['length']
            actions = [(a - 1) for a in self.action_history[-length:]]  # Map to [-1, 0, 1]
            action_segment = np.zeros(length, dtype=np.float32)
            if actions:
                action_segment[-len(actions):] = actions
            segments.append(action_segment)

        perf_cfg = self.history_settings['performance']
        if perf_cfg['enabled']:
            performance_metrics = self._compute_performance_metrics(perf_cfg['window'])
            segments.append(performance_metrics.astype(np.float32))

        position_cfg = self.history_settings['position_history']
        if position_cfg['enabled']:
            length = position_cfg['length']
            positions = self.position_history[-length:]
            position_segment = np.zeros(length, dtype=np.float32)
            if positions:
                position_segment[-len(positions):] = positions
            segments.append(position_segment)

        reward_cfg = self.history_settings['reward_history']
        if reward_cfg['enabled']:
            length = reward_cfg['length']
            rewards = self.reward_history[-length:]
            reward_segment = np.zeros(length, dtype=np.float32)
            if rewards:
                reward_segment[-len(rewards):] = rewards
            segments.append(reward_segment)

        if not segments:
            return np.array([], dtype=np.float32)

        return np.concatenate(segments).astype(np.float32)

    def _compute_performance_metrics(self, window: int) -> np.ndarray:
        """Compute rolling performance summary metrics."""
        metrics = np.zeros(self.history_settings['performance']['length'], dtype=np.float32)

        if window <= 1 or len(self.portfolio_history) < 2:
            return metrics

        sample = self.portfolio_history[-window:]
        if len(sample) < 2:
            return metrics

        values = np.array(sample, dtype=np.float32)
        prev_values = values[:-1]
        next_values = values[1:]

        # Avoid division by zero
        prev_values = np.where(prev_values == 0, 1e-8, prev_values)
        returns = (next_values - prev_values) / prev_values

        if returns.size == 0:
            return metrics

        cumulative_return = values[-1] / values[0] - 1.0 if values[0] != 0 else 0.0
        avg_return = returns.mean()
        volatility = returns.std()

        peak = np.maximum.accumulate(values)
        peak = np.where(peak == 0, 1e-8, peak)
        drawdowns = (peak - values) / peak
        max_drawdown = float(drawdowns.max()) if drawdowns.size > 0 else 0.0

        metrics[:] = [
            float(cumulative_return),
            float(avg_return),
            float(volatility),
            max_drawdown
        ]

        return metrics
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        self.cash = self.initial_capital
        self.holdings = 0
        self.total_value = self.initial_capital
        self.position_value = 0
        self._prev_holdings = 0
        
        self.action_history = []
        self.portfolio_history = [self.initial_capital]
        self.position_history = [0.0]
        self.reward_history = []
        
        # Calculate normalization parameters on first reset
        if self.normalize_obs and self.obs_mean is None:
            self._calculate_normalization_params()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            
        Returns:
            observation: Current state
            reward: Reward for this step
            done: Whether episode is complete
            info: Additional information
        """
        # Get current price
        # Support both 'close' and 'price' column names
        if 'close' in self.df.columns:
            current_price = self.df.loc[self.current_step, 'close']
        elif 'price' in self.df.columns:
            current_price = self.df.loc[self.current_step, 'price']
        else:
            raise ValueError("DataFrame must have either 'close' or 'price' column")

        prev_holdings = self.holdings

        # Execute action
        self._execute_action(action, current_price)

        # Track previous holdings for reward shaping
        self._prev_holdings = prev_holdings
        
        # Update portfolio value
        self.position_value = self.holdings * current_price
        self.total_value = self.cash + self.position_value
        
        # Calculate reward (implemented by subclass)
        reward = self._calculate_reward(action, current_price)
        
        # Track history
        self.action_history.append(action)
        self.portfolio_history.append(self.total_value)
        position_ratio = self.position_value / self.total_value if self.total_value > 0 else 0.0
        self.position_history.append(position_ratio)
        self.reward_history.append(reward)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # Get next observation
        obs = self._get_observation()
        
        # Build info dict
        info = {
            'total_value': self.total_value,
            'cash': self.cash,
            'holdings': self.holdings,
            'position_value': self.position_value,
            'current_price': current_price,
            'action': action
        }
        
        return obs, reward, done, info
    
    def _execute_action(self, action: int, current_price: float):
        """
        Execute trading action.
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            current_price: Current asset price
        """
        if action == 1:  # BUY
            spendable_cash = min(self.cash, self.cash * self.max_position_size)
            if current_price <= 0 or spendable_cash <= 0:
                return

            max_shares = int(spendable_cash / current_price)
            while max_shares > 0:
                fee = self._calculate_commission(max_shares, current_price)
                cost = max_shares * current_price + fee
                if cost <= spendable_cash and cost <= self.cash:
                    break
                max_shares -= 1

            if max_shares > 0:
                fee = self._calculate_commission(max_shares, current_price)
                total_cost = max_shares * current_price + fee
                if total_cost <= self.cash:
                    self.cash -= total_cost
                    self.holdings += max_shares
        
        elif action == 2:  # SELL
            if self.holdings > 0 and current_price > 0:
                fee = self._calculate_commission(self.holdings, current_price)
                gross = self.holdings * current_price
                fee = min(fee, gross)
                revenue = gross - fee
                self.cash += revenue
                self.holdings = 0
        
        # action == 0 (HOLD) does nothing

    def _calculate_commission(self, shares: int, price: float) -> float:
        if shares <= 0 or price <= 0:
            return 0.0

        if self._commission_config:
            rate = float(self._commission_config.get('per_share', 0.005))
            min_fee = float(self._commission_config.get('min_fee', 1.0))
            max_pct = float(self._commission_config.get('max_pct', 0.01))

            fee = shares * rate
            fee = max(fee, min_fee)
            trade_value = shares * price
            if max_pct > 0:
                fee = min(fee, trade_value * max_pct)
            fee = min(fee, trade_value)
            return float(fee)

        fee = self._commission_value
        return float(fee) if fee > 0 else 0.0
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).
        
        Returns:
            Normalized observation vector
        """
        if self.current_step >= len(self.df):
            # Return zeros if we're past the end
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Portfolio state
        cash_ratio = self.cash / self.initial_capital
        holdings_ratio = self.position_value / self.initial_capital if self.initial_capital > 0 else 0
        position_ratio = self.holdings / 1000 if self.holdings > 0 else 0  # Normalized by 1000 shares
        value_ratio = self.total_value / self.initial_capital
        
        portfolio_state = np.array([
            cash_ratio,
            holdings_ratio,
            position_ratio,
            value_ratio
        ], dtype=np.float32)
        
        # Market features (from DataFrame)
        row = self.df.iloc[self.current_step].copy()
        if 'position_context' in row.index:
            position_ratio = self.position_value / self.total_value if self.total_value > 0 else 0.0
            row['position_context'] = float(position_ratio)
        market_features = row.values.astype(np.float32)
        
        history_features = self._get_history_features()

        components = [portfolio_state, market_features]
        if history_features.size > 0:
            components.append(history_features.astype(np.float32))

        obs = np.concatenate(components)
        
        # Normalize if enabled
        if self.normalize_obs and self.obs_mean is not None:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        
        return obs
    
    def _calculate_normalization_params(self):
        """
        Calculate mean and std for observation normalization.
        Uses all available data for stable normalization.
        """
        all_obs = []
        
        for step in range(min(1000, len(self.df))):  # Sample first 1000 steps
            self.current_step = step
            obs = self._get_observation()
            all_obs.append(obs)

        all_obs = np.array(all_obs)
        self.obs_mean = np.mean(all_obs, axis=0)
        self.obs_std = np.std(all_obs, axis=0)

        self.current_step = 0  # Reset after calculation
    
    @abstractmethod
    def _calculate_reward(self, action: int, current_price: float) -> float:
        """
        Calculate reward for current step.
        
        Must be implemented by subclass (StockEnv, ETFEnv).
        Different reward functions for different trading strategies.
        
        Args:
            action: Action taken this step
            current_price: Current asset price
            
        Returns:
            Reward value (float)
        """
        pass
    
    def render(self, mode='human'):
        """
        Render environment state (for debugging).
        """
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.n_steps}")
            print(f"Cash: ${self.cash:.2f}")
            print(f"Holdings: {self.holdings} shares")
            print(f"Position Value: ${self.position_value:.2f}")
            print(f"Total Value: ${self.total_value:.2f}")
            print(f"Return: {((self.total_value / self.initial_capital - 1) * 100):.2f}%")
            print("-" * 50)
    
    def get_total_return(self) -> float:
        """
        Calculate total return percentage.
        
        Returns:
            Return as percentage
        """
        return (self.total_value / self.initial_capital - 1) * 100
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio of portfolio returns.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Sharpe ratio
        """
        if len(self.portfolio_history) < 2:
            return 0.0
        
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
