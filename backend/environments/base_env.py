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
from typing import Dict, Tuple, Optional


class BaseTradingEnv(gym.Env, ABC):
    """
    Abstract base environment for stock/ETF trading.
    
    Follows OpenAI Gym interface for compatibility with Stable-Baselines3.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000.0,
        commission: float = 1.0,
        max_position_size: float = 1.0,
        normalize_obs: bool = True
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
        self.commission = commission
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
        self.reward_history = []
        
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
        n_history = 5  # Last 5 actions
        
        obs_dim = n_portfolio_state + n_features + n_history
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
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
        
        self.action_history = []
        self.portfolio_history = []
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
        
        # Execute action
        self._execute_action(action, current_price)
        
        # Update portfolio value
        self.position_value = self.holdings * current_price
        self.total_value = self.cash + self.position_value
        
        # Calculate reward (implemented by subclass)
        reward = self._calculate_reward(action, current_price)
        
        # Track history
        self.action_history.append(action)
        self.portfolio_history.append(self.total_value)
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
            # Calculate max shares we can buy
            max_buy_value = self.cash * self.max_position_size
            max_shares = int((max_buy_value - self.commission) / current_price)
            
            if max_shares > 0:
                cost = max_shares * current_price + self.commission
                self.cash -= cost
                self.holdings += max_shares
        
        elif action == 2:  # SELL
            # Sell all holdings
            if self.holdings > 0:
                revenue = self.holdings * current_price - self.commission
                self.cash += revenue
                self.holdings = 0
        
        # action == 0 (HOLD) does nothing
    
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
        market_features = self.df.iloc[self.current_step].values.astype(np.float32)
        
        # Recent action history (last 5 actions, one-hot encoded)
        recent_actions = self.action_history[-5:] if len(self.action_history) >= 5 else [0] * 5
        action_features = np.array(recent_actions, dtype=np.float32)
        
        # Concatenate all features
        obs = np.concatenate([portfolio_state, market_features, action_features])
        
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
