"""
Backtesting Framework for RL Trading System

Runs trained RL agents on test data and evaluates performance:
- Load trained model
- Execute trading strategy on test data
- Track equity curve and trades
- Calculate performance metrics
- Compare to Buy & Hold benchmark
- Save results to JSON

Author: YoopRL System
Date: November 8, 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from environments.stock_env import StockTradingEnv
from environments.etf_env import ETFTradingEnv
from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from evaluation.metrics import calculate_all_metrics


class Backtester:
    """
    Backtesting framework for trained RL agents.
    
    Runs a trained model on test data and calculates comprehensive
    performance metrics including comparison to Buy & Hold benchmark.
    """
    
    def __init__(self, model_path: str, test_data: pd.DataFrame, agent_type: str = 'PPO'):
        """
        Initialize backtester.
        
        Args:
            model_path: Path to trained model file (.zip)
            test_data: Test data DataFrame with OHLCV + features
            agent_type: 'PPO' or 'SAC'
        """
        self.model_path = Path(model_path)
        self.test_data = test_data
        self.agent_type = agent_type.upper()
        
        # Create environment
        if self.agent_type == 'PPO':
            self.env = StockTradingEnv(df=test_data)
            self.agent = PPOAgent(env=self.env, hyperparameters={})
        elif self.agent_type == 'SAC':
            self.env = ETFTradingEnv(df=test_data)
            self.agent = SACAgent(env=self.env, hyperparameters={})
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Load trained model
        try:
            self.agent.load(str(self.model_path))
            print(f"✅ Model loaded: {self.model_path.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def run(self) -> Dict:
        """
        Run backtest on test data.
        
        Executes the trained agent on test data, tracking all trades
        and portfolio values.
        
        Returns:
            Dictionary with:
            {
                'metrics': {...},
                'equity_curve': [...],
                'trades': [...],
                'actions_log': [...]
            }
        """
        
        print(f"\n{'='*70}")
        print(f"📊 Running Backtest: {self.agent_type} Agent")
        print(f"   Test samples: {len(self.test_data)}")
        print(f"{'='*70}\n")
        
        # Reset environment
        obs = self.env.reset()
        done = False
        
        # Tracking variables
        equity_curve = [self.env.initial_capital]
        trades = []
        actions_log = []
        
        step = 0
        prev_position = 0
        
        # Run episode
        while not done and step < len(self.test_data):
            # Predict action (deterministic for backtesting)
            prediction = self.agent.predict(obs, deterministic=True)
            action = prediction[0] if isinstance(prediction, tuple) else prediction
            
            # Execute action
            obs, reward, done, info = self.env.step(action)
            
            # Track equity
            current_equity = info.get('total_value', self.env.total_value)
            equity_curve.append(current_equity)
            
            # Track trades (when position changes)
            current_position = info.get('holdings', self.env.holdings)
            
            if current_position != prev_position:
                # Position changed - record trade
                if prev_position > 0 and current_position == 0:
                    # Sell - calculate P&L
                    pnl = equity_curve[-1] - equity_curve[-2] if len(equity_curve) > 1 else 0
                    if abs(pnl) > 0.01:  # Ignore tiny fluctuations
                        trades.append(pnl)
            
            prev_position = current_position
            
            # Log action
            if step < 100:  # Only log first 100 actions to save memory
                actions_log.append({
                    'step': step,
                    'action': int(action) if isinstance(action, np.ndarray) else action,
                    'price': float(self.test_data.iloc[step].get('price', 0)),
                    'equity': round(current_equity, 2),
                    'holdings': int(current_position)
                })
            
            step += 1
        
        print(f"✅ Backtest complete: {step} steps executed\n")
        
        # Calculate metrics
        print("📈 Calculating performance metrics...")
        
        metrics = calculate_all_metrics(
            equity_curve=np.array(equity_curve),
            trades=trades,
            initial_balance=self.env.initial_capital
        )
        
        # Calculate Buy & Hold benchmark
        buy_and_hold_return = self._calculate_buy_and_hold()
        metrics['buy_and_hold_return'] = buy_and_hold_return
        metrics['alpha'] = metrics['total_return'] - buy_and_hold_return
        
        # Print results
        self._print_results(metrics)
        
        return {
            'metrics': metrics,
            'equity_curve': [float(x) for x in equity_curve],
            'trades': [float(t) for t in trades],
            'actions_log': actions_log,
            'backtest_info': {
                'model_path': str(self.model_path),
                'agent_type': self.agent_type,
                'test_samples': len(self.test_data),
                'total_steps': step,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _calculate_buy_and_hold(self) -> float:
        """
        Calculate Buy & Hold benchmark return.
        
        Returns:
            Total return percentage for buying at start and holding until end
        """
        if 'price' not in self.test_data.columns or len(self.test_data) < 2:
            return 0.0
        
        initial_price = self.test_data.iloc[0]['price']
        final_price = self.test_data.iloc[-1]['price']
        
        buy_and_hold_return = ((final_price - initial_price) / initial_price) * 100
        
        return round(float(buy_and_hold_return), 2)
    
    def _print_results(self, metrics: Dict) -> None:
        """Print formatted backtest results."""
        
        print(f"\n{'='*70}")
        print(f"📊 Backtest Results:")
        print(f"{'='*70}")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:     {metrics['sortino_ratio']:>8.2f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown']:>8.2f}%")
        print(f"  Win Rate:          {metrics['win_rate']:>8.2f}%")
        print(f"  Profit Factor:     {metrics['profit_factor']:>8.2f}")
        print(f"  Calmar Ratio:      {metrics['calmar_ratio']:>8.2f}")
        print(f"")
        print(f"  Total Return:      {metrics['total_return']:>8.2f}%")
        print(f"  Buy & Hold:        {metrics['buy_and_hold_return']:>8.2f}%")
        print(f"  Alpha:             {metrics['alpha']:>8.2f}%")
        print(f"")
        print(f"  Final Balance:     ${metrics['final_balance']:>12,.2f}")
        print(f"  Total Trades:      {metrics['total_trades']:>8}")
        print(f"  Win/Loss:          {metrics['winning_trades']:>4}/{metrics['losing_trades']:<4}")
        print(f"  Avg Win:           ${metrics['avg_win']:>10,.2f}")
        print(f"  Avg Loss:          ${metrics['avg_loss']:>10,.2f}")
        print(f"{'='*70}\n")
    
    def save_results(self, results: Dict, filepath: str) -> None:
        """
        Save backtest results to JSON file.
        
        Args:
            results: Results dictionary from run()
            filepath: Output file path
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved: {output_path}")


def evaluate_trained_model(
    model,
    test_env,
    n_eval_episodes: int = 10,
    initial_capital: float = 100000
) -> Dict:
    """
    Evaluate a trained model on test environment.
    
    This is the main function used by train.py for automatic evaluation.
    
    Args:
        model: Trained stable-baselines3 model
        test_env: Test environment (StockTradingEnv or ETFTradingEnv)
        n_eval_episodes: Number of episodes to run
        initial_capital: Starting capital
    
    Returns:
        Dictionary with metrics, equity_curve, trade_history
    """
    print(f"\n📊 Evaluating model on test set...")
    print(f"   Episodes: {n_eval_episodes}")
    print(f"   Initial Capital: ${initial_capital:,.0f}")
    
    all_equity_curves = []
    all_trades = []
    
    for episode in range(n_eval_episodes):
        # Reset environment
        obs = test_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        done = False
        episode_equity = [initial_capital]
        episode_trades = []
        episode_actions = []
        
        step = 0
        prev_portfolio_value = initial_capital
        
        while not done:
            # Predict action (deterministic)
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            step_result = test_env.step(action)
            
            # Handle different gym versions
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            # Track portfolio value
            current_portfolio_value = info.get('portfolio_value', info.get('total_value', prev_portfolio_value))
            episode_equity.append(current_portfolio_value)
            
            # Track trades (when portfolio changes significantly)
            pnl = current_portfolio_value - prev_portfolio_value
            if abs(pnl) > 0.01:  # Track if change > $0.01
                episode_trades.append(float(pnl))
            
            prev_portfolio_value = current_portfolio_value
            step += 1
        
        all_equity_curves.append(episode_equity)
        all_trades.extend(episode_trades)
    
    # Average equity curve
    min_length = min(len(eq) for eq in all_equity_curves)
    truncated_curves = [eq[:min_length] for eq in all_equity_curves]
    avg_equity_curve = np.mean(truncated_curves, axis=0)
    
    # Calculate metrics
    metrics = calculate_all_metrics(
        equity_curve=avg_equity_curve,
        trades=all_trades,
        initial_balance=initial_capital
    )
    
    print(f"\n✅ Evaluation Complete!")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Total Return: {metrics['total_return']:+.2f}%")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"   Win Rate: {metrics['win_rate']:.1f}%")
    print(f"   Total Trades: {metrics['total_trades']}")
    
    return {
        'metrics': metrics,
        'equity_curve': [float(x) for x in avg_equity_curve],
        'trades': [float(t) for t in all_trades]
    }


def run_backtest(
    model_path: str,
    test_data: pd.DataFrame,
    agent_type: str = 'PPO',
    save_path: str = None
) -> Dict:
    """
    Convenience function to run backtest from saved model file.
    
    Args:
        model_path: Path to trained model
        test_data: Test data DataFrame
        agent_type: 'PPO' or 'SAC'
        save_path: Optional path to save results JSON
    
    Returns:
        Backtest results dictionary
    """
    backtester = Backtester(model_path, test_data, agent_type)
    results = backtester.run()
    
    if save_path:
        backtester.save_results(results, save_path)
    
    return results


if __name__ == '__main__':
    """
    Test backtester
    """
    
    # Create dummy test data
    print("🧪 Testing Backtester...\n")
    
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
    n_samples = len(dates)
    
    test_data = pd.DataFrame({
        'date': dates,
        'price': 100 + np.random.randn(n_samples).cumsum(),
        'volume': np.random.randint(1000000, 10000000, n_samples),
        'rsi': np.random.uniform(30, 70, n_samples),
        'macd': np.random.randn(n_samples),
        'ema_10': 100 + np.random.randn(n_samples).cumsum() * 0.5,
        'ema_50': 100 + np.random.randn(n_samples).cumsum() * 0.3,
        'vix': np.random.uniform(10, 30, n_samples),
    })
    
    print("⚠️ Note: This is a test with dummy data.")
    print("   Replace with actual trained model and test data.\n")
