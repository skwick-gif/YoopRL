"""
stock_env.py
Stock Trading Environment (PPO-Optimized)

Purpose:
- Trading environment for regular stocks (AAPL, GOOGL, TSLA, etc.)
- Optimized for PPO agent with risk-adjusted reward function
- Conservative strategy focused on steady returns with low risk

Why separate file:
- Different reward structure than ETF trading
- PPO works best with risk-penalized rewards
- Stock-specific trading logic (no leverage)
- Clear separation between asset types

Reward Function (PPO-specific):
- Portfolio returns (main component)
- Risk penalty for volatility
- Drawdown penalty for large losses
- Encourages consistent, stable gains

Action Space:
- 0: HOLD - maintain current position
- 1: BUY - purchase shares (up to max_position_size)
- 2: SELL - liquidate all holdings

Wiring:
- Extends BaseTradingEnv
- Used by PPO agent during training
- Receives data from feature_engineering.py
- Called by train.py training loop
"""

import numpy as np
import pandas as pd
from environments.base_env import BaseTradingEnv


class StockTradingEnv(BaseTradingEnv):
    """
    Trading environment for regular stocks.
    
    Optimized for PPO with risk-adjusted rewards.
    Focus on steady, low-risk returns.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000.0,
        commission: float = 1.0,
        max_position_size: float = 1.0,
        risk_penalty: float = -0.5,
        normalize_obs: bool = True
    ):
        """
        Initialize stock trading environment.
        
        Args:
            df: DataFrame with OHLCV + features
            initial_capital: Starting portfolio value ($)
            commission: Trading fee per transaction ($)
            max_position_size: Max fraction of portfolio per trade (0-1)
            risk_penalty: Penalty coefficient for volatility (negative)
            normalize_obs: Whether to normalize observations
        """
        super().__init__(df, initial_capital, commission, max_position_size, normalize_obs)
        
        self.risk_penalty = risk_penalty
        
        # Track for risk metrics
        self.previous_value = initial_capital
        self.peak_value = initial_capital
        self.returns_window = []  # Rolling window for volatility calculation
        self.window_size = 20  # 20-step rolling window
    
    def _calculate_reward(self, action: int, current_price: float) -> float:
        """
        Calculate PPO-optimized reward.
        
        Reward components:
        1. Portfolio return (main component)
        2. Risk penalty (volatility of returns)
        3. Drawdown penalty (distance from peak)
        4. Action penalty (discourage excessive trading)
        
        Args:
            action: Action taken this step
            current_price: Current stock price
            
        Returns:
            Total reward (float)
        """
        # 1. Portfolio return component
        portfolio_return = (self.total_value - self.previous_value) / self.previous_value
        return_reward = portfolio_return * 100  # Scale up for better learning signal
        
        # 2. Risk penalty (volatility of recent returns)
        self.returns_window.append(portfolio_return)
        if len(self.returns_window) > self.window_size:
            self.returns_window.pop(0)
        
        if len(self.returns_window) >= 5:  # Need minimum samples
            volatility = np.std(self.returns_window)
            risk_penalty_value = self.risk_penalty * volatility * 100
        else:
            risk_penalty_value = 0.0
        
        # 3. Drawdown penalty (how far from peak)
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
        
        drawdown = (self.peak_value - self.total_value) / self.peak_value
        drawdown_penalty = -10.0 * drawdown if drawdown > 0.1 else 0.0  # Penalty if >10% drawdown
        
        # 4. Action penalty (discourage excessive trading)
        action_penalty = 0.0
        if action in [1, 2]:  # BUY or SELL
            # Check if we're trading too frequently
            recent_trades = sum(1 for a in self.action_history[-5:] if a != 0)
            if recent_trades > 3:  # More than 3 trades in last 5 steps
                action_penalty = -0.5
        
        # 5. Holding reward (encourage staying invested when profitable)
        holding_reward = 0.0
        if action == 0 and self.holdings > 0 and portfolio_return > 0:
            holding_reward = 0.2  # Small bonus for holding during gains
        
        # Total reward
        total_reward = (
            return_reward +
            risk_penalty_value +
            drawdown_penalty +
            action_penalty +
            holding_reward
        )
        
        # Update previous value for next step
        self.previous_value = self.total_value
        
        return total_reward
    
    def reset(self) -> np.ndarray:
        """
        Reset environment and risk tracking.
        
        Returns:
            Initial observation
        """
        obs = super().reset()
        
        # Reset risk tracking
        self.previous_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.returns_window = []
        
        return obs
    
    def get_risk_metrics(self) -> dict:
        """
        Calculate risk-adjusted performance metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        if len(self.portfolio_history) < 2:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'calmar_ratio': 0.0
            }
        
        # Calculate returns
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Total return
        total_return = (self.total_value / self.initial_capital - 1) * 100
        
        # Sharpe ratio
        sharpe = self.get_sharpe_ratio()
        
        # Sortino ratio (only penalizes downside volatility)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino = np.mean(returns) / downside_std * np.sqrt(252)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) * 100
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252) * 100
        
        # Calmar ratio (return / max drawdown)
        calmar = total_return / max_drawdown if max_drawdown > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'calmar_ratio': calmar
        }


def create_stock_env(
    symbol: str,
    df: pd.DataFrame,
    config: dict
) -> StockTradingEnv:
    """
    Factory function to create StockTradingEnv with configuration.
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        df: DataFrame with features
        config: Configuration dictionary from training_config.py
        
    Returns:
        Configured StockTradingEnv instance
    """
    return StockTradingEnv(
        df=df,
        initial_capital=config.get('initial_capital', 100000.0),
        commission=config.get('commission', 1.0),
        max_position_size=config.get('max_position_size', 1.0),
        risk_penalty=config.get('risk_penalty', -0.5),
        normalize_obs=config.get('normalize_obs', True)
    )
