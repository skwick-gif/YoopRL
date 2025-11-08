"""
Evaluation Module for RL Trading System

Provides backtesting and performance evaluation:
- Performance metrics (Sharpe, Sortino, Drawdown, etc.)
- Backtesting framework for trained models
- Results analysis and comparison

Author: YoopRL System
Date: November 8, 2025
"""

from .metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_total_return,
    calculate_calmar_ratio,
    calculate_all_metrics
)

from .backtester import Backtester, run_backtest

__all__ = [
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_total_return',
    'calculate_calmar_ratio',
    'calculate_all_metrics',
    'Backtester',
    'run_backtest'
]
