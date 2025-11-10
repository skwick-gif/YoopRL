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
from typing import Dict, Optional, Tuple
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
        normalize_obs: bool = True,
        history_config: Optional[dict] = None
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
        super().__init__(
            df,
            initial_capital,
            commission,
            max_position_size,
            normalize_obs,
            history_config=history_config
        )
        
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
        
        # 5. Opportunity cost / market alignment reward
        market_return = 0.0
        if self.current_step > 0:
            if 'returns' in self.df.columns:
                raw_return = self.df.loc[self.current_step, 'returns']
                market_return = float(raw_return) if np.isfinite(raw_return) else 0.0
            else:
                price_col = 'price' if 'price' in self.df.columns else 'close'
                prev_price = self.df.loc[self.current_step - 1, price_col]
                denominator = prev_price if prev_price != 0 else 1e-8
                market_return = float((current_price - prev_price) / denominator)

        opportunity_reward = 0.0
        if market_return > 0.0:
            capped = np.clip(market_return, 0.0, 0.03)
            if self.holdings > 0:
                opportunity_reward = capped * 20.0
            else:
                opportunity_reward = -capped * 10.0
        elif market_return < 0.0 and self.holdings > 0:
            capped = np.clip(abs(market_return), 0.0, 0.03)
            opportunity_reward = -capped * 20.0

        # 6. Trade incentive / inactivity penalty
        trade_bonus = 0.0
        prev_holdings = getattr(self, '_prev_holdings', 0)
        if action == 1 and prev_holdings == 0 and self.holdings > 0:
            trade_bonus = 0.3  # Encourage opening new positions when flat
        elif action == 2 and prev_holdings == 0:
            trade_bonus = -0.2  # Discourage selling when nothing is held

        trade_penalty = 0.0
        if action in (1, 2):
            trade_penalty = -0.02  # light damping of excessive trades

        inactivity_penalty = 0.0
        if action == 0 and self.holdings == 0:
            inactivity_penalty = -0.01

        holding_reward = 0.0
        if action == 0 and self.holdings > 0 and portfolio_return > 0:
            holding_reward = 0.2  # Small bonus for holding during gains
        
        # Total reward
        total_reward = (
            return_reward +
            risk_penalty_value +
            drawdown_penalty +
            action_penalty +
            holding_reward +
            opportunity_reward +
            trade_bonus +
            trade_penalty +
            inactivity_penalty
        )
        
        # Update previous value for next step
        self.previous_value = self.total_value
        
        return total_reward
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment-specific tracking before starting an episode."""
        obs, info = super().reset(seed=seed, options=options)

        self.previous_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.returns_window = []

        return obs, info
    
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
