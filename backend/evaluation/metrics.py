"""
Performance Metrics for RL Trading System

Provides comprehensive performance metrics for backtesting:
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted return)
- Maximum Drawdown (peak-to-trough decline)
- Win Rate (percentage of winning trades)
- Profit Factor (gross profit / gross loss)
- Total Return (percentage gain/loss)

Author: YoopRL System
Date: November 8, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe Ratio (annualized).
    
    Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev of Returns
    
    Args:
        returns: Array of returns (daily)
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    # Adjust risk-free rate to daily (assume 252 trading days)
    daily_rf = risk_free_rate / 252
    
    # Calculate Sharpe
    sharpe = (mean_return - daily_rf) / std_return
    
    # Annualize (sqrt of 252 trading days)
    sharpe_annual = sharpe * np.sqrt(252)
    
    return float(sharpe_annual)


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino Ratio (annualized).
    
    Sortino Ratio = (Mean Return - Risk Free Rate) / Downside Deviation
    Only considers downside volatility (negative returns).
    
    Args:
        returns: Array of returns (daily)
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = np.mean(returns)
    
    # Downside deviation (only negative returns)
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) == 0:
        return 0.0  # No downside
    
    downside_std = np.std(negative_returns)
    
    if downside_std == 0:
        return 0.0
    
    # Adjust risk-free rate to daily
    daily_rf = risk_free_rate / 252
    
    # Calculate Sortino
    sortino = (mean_return - daily_rf) / downside_std
    
    # Annualize
    sortino_annual = sortino * np.sqrt(252)
    
    return float(sortino_annual)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate Maximum Drawdown.
    
    Max Drawdown = Maximum peak-to-trough decline (as percentage)
    
    Args:
        equity_curve: Array of portfolio values over time
    
    Returns:
        Max drawdown as negative percentage (e.g., -15.5 for 15.5% drawdown)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown at each point
    drawdown = (equity_curve - running_max) / running_max
    
    # Get maximum drawdown (most negative value)
    max_dd = np.min(drawdown)
    
    return float(max_dd * 100)  # Convert to percentage


def calculate_win_rate(trades: List[float]) -> float:
    """
    Calculate Win Rate.
    
    Win Rate = (Number of Winning Trades) / (Total Trades) * 100
    
    Args:
        trades: List of trade P&Ls (positive = win, negative = loss)
    
    Returns:
        Win rate as percentage (0-100)
    """
    if len(trades) == 0:
        return 0.0
    
    winning_trades = [t for t in trades if t > 0]
    
    win_rate = (len(winning_trades) / len(trades)) * 100
    
    return float(win_rate)


def calculate_profit_factor(trades: List[float]) -> float:
    """
    Calculate Profit Factor.
    
    Profit Factor = Gross Profit / Gross Loss
    
    A value > 1 indicates profitable trading overall.
    
    Args:
        trades: List of trade P&Ls
    
    Returns:
        Profit factor (>1 is profitable, <1 is losing)
    """
    if len(trades) == 0:
        return 0.0
    
    gross_profit = sum([t for t in trades if t > 0])
    gross_loss = abs(sum([t for t in trades if t < 0]))
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    profit_factor = gross_profit / gross_loss
    
    return float(profit_factor)


def calculate_total_return(initial_balance: float, final_balance: float) -> float:
    """
    Calculate Total Return.
    
    Total Return = (Final - Initial) / Initial * 100
    
    Args:
        initial_balance: Starting capital
        final_balance: Ending capital
    
    Returns:
        Total return as percentage
    """
    if initial_balance == 0:
        return 0.0
    
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    
    return float(total_return)


def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """
    Calculate Calmar Ratio.
    
    Calmar Ratio = Annual Return / |Maximum Drawdown|
    
    Args:
        total_return: Total return percentage
        max_drawdown: Maximum drawdown percentage (negative)
    
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return 0.0
    
    calmar = total_return / abs(max_drawdown)
    
    return float(calmar)


def calculate_all_metrics(
    equity_curve: np.ndarray,
    trades: List[float],
    initial_balance: float,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate all performance metrics.
    
    Args:
        equity_curve: Portfolio values over time
        trades: List of individual trade P&Ls
        initial_balance: Starting capital
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Dictionary with all metrics:
        {
            'sharpe_ratio': float,
            'sortino_ratio': float,
            'max_drawdown': float (negative %),
            'win_rate': float (%),
            'profit_factor': float,
            'total_return': float (%),
            'calmar_ratio': float,
            'final_balance': float,
            'total_trades': int,
            'winning_trades': int,
            'losing_trades': int,
            'avg_win': float,
            'avg_loss': float
        }
    """
    # Calculate returns from equity curve
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
    else:
        returns = np.array([])
    
    # Calculate basic metrics
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino = calculate_sortino_ratio(returns, risk_free_rate)
    max_dd = calculate_max_drawdown(equity_curve)
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    
    final_balance = equity_curve[-1] if len(equity_curve) > 0 else initial_balance
    total_return = calculate_total_return(initial_balance, final_balance)
    calmar = calculate_calmar_ratio(total_return, max_dd)
    
    # Trade statistics
    winning_trades = [t for t in trades if t > 0]
    losing_trades = [t for t in trades if t < 0]
    
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
    avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
    
    metrics = {
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'max_drawdown': round(max_dd, 2),
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'total_return': round(total_return, 2),
        'calmar_ratio': round(calmar, 2),
        'final_balance': round(final_balance, 2),
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'avg_win': round(float(avg_win), 2),
        'avg_loss': round(float(avg_loss), 2)
    }
    
    return metrics


if __name__ == '__main__':
    """
    Test metrics calculation
    """
    
    # Example data
    initial = 100000
    equity = np.array([100000, 102000, 101000, 105000, 103000, 108000, 110000, 107000, 112000, 115000])
    trades = [2000, -1000, 4000, -2000, 5000, 2000, -3000, 5000, 3000]
    
    print("🧪 Testing Metrics Calculation...\n")
    
    metrics = calculate_all_metrics(equity, trades, initial)
    
    print("📊 Performance Metrics:")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:     {metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown:      {metrics['max_drawdown']:.2f}%")
    print(f"  Win Rate:          {metrics['win_rate']:.2f}%")
    print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
    print(f"  Total Return:      {metrics['total_return']:.2f}%")
    print(f"  Calmar Ratio:      {metrics['calmar_ratio']:.2f}")
    print(f"  Final Balance:     ${metrics['final_balance']:,.2f}")
    print(f"  Total Trades:      {metrics['total_trades']}")
    print(f"  Win/Loss:          {metrics['winning_trades']}/{metrics['losing_trades']}")
    print(f"  Avg Win:           ${metrics['avg_win']:,.2f}")
    print(f"  Avg Loss:          ${metrics['avg_loss']:,.2f}")
