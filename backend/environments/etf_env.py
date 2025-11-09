"""
etf_env.py
Leveraged ETF Trading Environment (SAC-Optimized)

Purpose:
- Trading environment for leveraged ETFs (TNA, TQQQ, SPXL, etc.)
- Optimized for SAC agent with volatility-adjusted reward function
- Aggressive strategy focused on high returns with controlled volatility

Why separate file:
- Different reward structure than stock trading
- SAC works best with entropy-regularized rewards
- ETF-specific trading logic (leveraged positions)
- Higher risk tolerance than stock environment

Reward Function (SAC-specific):
- Portfolio returns (main component)
- Volatility penalty (less aggressive than stock env)
- Entropy bonus (encourages exploration)
- Position sizing reward (optimal leverage usage)

Action Space:
- 0: HOLD - maintain current position
- 1: BUY - purchase units (up to max_position_size)
- 2: SELL - liquidate all holdings

Key Differences from StockEnv:
- Lower risk penalty (leverage is expected)
- Focus on capturing volatility opportunities
- Faster trading frequency encouraged
- Higher position sizing allowed

Wiring:
- Extends BaseTradingEnv
- Used by SAC agent during training
- Receives data from feature_engineering.py
- Called by train.py training loop
"""

import numpy as np
import pandas as pd
from typing import Optional
from environments.base_env import BaseTradingEnv


class ETFTradingEnv(BaseTradingEnv):
    """
    Trading environment for leveraged ETFs.
    
    Optimized for SAC with volatility-adjusted rewards.
    Focus on high returns with controlled risk.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000.0,
        commission: float = 1.0,
        max_position_size: float = 1.0,
        vol_penalty: float = -0.3,
        leverage_factor: float = 3.0,
        normalize_obs: bool = True,
        history_config: Optional[dict] = None
    ):
        """
        Initialize ETF trading environment.
        
        Args:
            df: DataFrame with OHLCV + features
            initial_capital: Starting portfolio value ($)
            commission: Trading fee per transaction ($)
            max_position_size: Max fraction of portfolio per trade (0-1)
            vol_penalty: Penalty coefficient for volatility (less negative than stock)
            leverage_factor: ETF leverage multiplier (3x for TNA/TQQQ)
            normalize_obs: Whether to normalize observations
        """
        super().__init__(
            df,
            initial_capital,
            commission,
            max_position_size,
            normalize_obs,
            history_config=history_config
        )
        
        self.vol_penalty = vol_penalty
        self.leverage_factor = leverage_factor
        
        # Track for volatility metrics
        self.previous_value = initial_capital
        self.returns_window = []  # Rolling window for volatility
        self.window_size = 10  # Shorter window for faster adaptation
        
        # Volatility regime tracking
        self.high_vol_threshold = 0.02  # 2% daily volatility threshold
        self.low_vol_threshold = 0.005  # 0.5% daily volatility threshold
    
    def _calculate_reward(self, action: int, current_price: float) -> float:
        """
        Calculate SAC-optimized reward.
        
        Reward components:
        1. Portfolio return (scaled for leverage)
        2. Volatility penalty (lighter than stock env)
        3. Momentum reward (capture trending moves)
        4. Position sizing reward (optimal leverage usage)
        5. Action diversity bonus (encourages exploration)
        
        Args:
            action: Action taken this step
            current_price: Current ETF price
            
        Returns:
            Total reward (float)
        """
        # 1. Portfolio return component
        portfolio_return = (self.total_value - self.previous_value) / self.previous_value
        return_reward = portfolio_return * 100  # Scale up
        
        # 2. Volatility penalty (lighter than stock env)
        self.returns_window.append(portfolio_return)
        if len(self.returns_window) > self.window_size:
            self.returns_window.pop(0)
        
        volatility = 0.0
        vol_penalty_value = 0.0
        if len(self.returns_window) >= 3:
            volatility = np.std(self.returns_window)
            vol_penalty_value = self.vol_penalty * volatility * 100
        
        # 3. Momentum reward (capture strong trends)
        momentum_reward = 0.0
        if len(self.returns_window) >= 3:
            recent_trend = np.mean(self.returns_window[-3:])
            if abs(recent_trend) > 0.01:  # Strong momentum
                if action == 1 and recent_trend > 0:  # Buying uptrend
                    momentum_reward = 0.5
                elif action == 2 and recent_trend < 0:  # Selling downtrend
                    momentum_reward = 0.5
        
        # 4. Position sizing reward (encourage optimal leverage usage)
        position_ratio = self.position_value / self.total_value if self.total_value > 0 else 0
        position_reward = 0.0
        
        # Reward for being positioned (ETFs benefit from staying invested)
        if position_ratio > 0.8:  # Near full position
            position_reward = 0.3
        elif position_ratio < 0.2 and action == 0:  # Staying out when appropriate
            # Check if volatility is too high
            if volatility > self.high_vol_threshold:
                position_reward = 0.2  # Good to stay out during high vol
        
        # 5. Action diversity bonus (SAC benefits from exploration)
        action_diversity_bonus = 0.0
        if len(self.action_history) >= 5:
            unique_actions = len(set(self.action_history[-5:]))
            if unique_actions >= 2:  # Using different actions
                action_diversity_bonus = 0.1
        
        # 6. Volatility regime adaptation
        regime_reward = 0.0
        if volatility > 0:
            if volatility > self.high_vol_threshold:
                # High volatility - reward risk management
                if action == 2 and self.holdings > 0:  # Reducing exposure
                    regime_reward = 0.3
            elif volatility < self.low_vol_threshold:
                # Low volatility - reward staying invested
                if action == 1 or (action == 0 and self.holdings > 0):
                    regime_reward = 0.2
        
        # Total reward
        total_reward = (
            return_reward +
            vol_penalty_value +
            momentum_reward +
            position_reward +
            action_diversity_bonus +
            regime_reward
        )
        
        # Update previous value for next step
        self.previous_value = self.total_value
        
        return total_reward
    
    def reset(self) -> np.ndarray:
        """
        Reset environment and tracking variables.
        
        Returns:
            Initial observation
        """
        obs = super().reset()
        
        # Reset ETF-specific tracking
        self.previous_value = self.initial_capital
        self.returns_window = []
        
        return obs
    
    def get_etf_metrics(self) -> dict:
        """
        Calculate ETF-specific performance metrics.
        
        Returns:
            Dictionary with ETF metrics
        """
        if len(self.portfolio_history) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'avg_position_ratio': 0.0,
                'num_trades': 0
            }
        
        # Calculate returns
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Total return
        total_return = (self.total_value / self.initial_capital - 1) * 100
        
        # Sharpe ratio
        sharpe = self.get_sharpe_ratio()
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino = np.mean(returns) / downside_std * np.sqrt(252)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) * 100
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) * 100
        
        # Average position ratio (how much capital was deployed)
        position_ratios = []
        for value in portfolio_values:
            position_value = self.holdings * self.df.loc[min(len(self.df)-1, len(position_ratios)), 'close']
            position_ratio = position_value / value if value > 0 else 0
            position_ratios.append(position_ratio)
        avg_position_ratio = np.mean(position_ratios) * 100
        
        # Number of trades
        num_trades = sum(1 for action in self.action_history if action != 0)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'avg_position_ratio': avg_position_ratio,
            'num_trades': num_trades,
            'leverage_factor': self.leverage_factor
        }


def create_etf_env(
    symbol: str,
    df: pd.DataFrame,
    config: dict
) -> ETFTradingEnv:
    """
    Factory function to create ETFTradingEnv with configuration.
    
    Args:
        symbol: ETF ticker (e.g., 'TNA', 'TQQQ')
        df: DataFrame with features
        config: Configuration dictionary from training_config.py
        
    Returns:
        Configured ETFTradingEnv instance
    """
    # Detect leverage factor from symbol
    leverage_factor = 3.0  # Default 3x
    if 'UPRO' in symbol or 'SPXL' in symbol:
        leverage_factor = 3.0
    elif 'TQQQ' in symbol or 'TNA' in symbol:
        leverage_factor = 3.0
    elif 'SSO' in symbol or 'QLD' in symbol:
        leverage_factor = 2.0
    
    return ETFTradingEnv(
        df=df,
        initial_capital=config.get('initial_capital', 100000.0),
        commission=config.get('commission', 1.0),
        max_position_size=config.get('max_position_size', 1.0),
        vol_penalty=config.get('vol_penalty', -0.3),
        leverage_factor=leverage_factor,
        normalize_obs=config.get('normalize_obs', True)
    )
