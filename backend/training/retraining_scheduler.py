"""
Automatic Retraining Scheduler for RL Trading System

Provides automatic retraining workflow with:
- Scheduled retraining (Daily/Weekly/Monthly)
- Data merging (historical + live logs)
- Performance validation
- Automatic deployment of better models
- Rollback capability

Author: YoopRL System
Date: November 8, 2025
"""

import schedule
import time
from pathlib import Path
import sys
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.train import train_agent, load_data
from evaluation.backtester import run_backtest
from models.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """
    Automatic retraining scheduler for RL trading agents.
    
    Workflow:
    1. Merge historical data with recent live logs
    2. Train new model with updated data
    3. Backtest new model on test set
    4. Compare with current best model
    5. Deploy if performance improves
    6. Archive old models
    """
    
    def __init__(
        self,
        config: dict,
        frequency: str = 'Weekly',
    performance_threshold: float = 0.2,
    min_sharpe: float = 1.5
    ):
        """
        Initialize retraining scheduler.
        
        Args:
            config: Training configuration dictionary
            frequency: 'Daily', 'Weekly', or 'Monthly'
            performance_threshold: Minimum improvement required (Sharpe diff)
            min_sharpe: Minimum Sharpe ratio to deploy
        """
        self.config = config
        self.frequency = frequency
        self.performance_threshold = performance_threshold
        self.min_sharpe = min_sharpe
        
        self.model_manager = ModelManager()
        
        logger.info(
            f"RetrainingScheduler initialized: {frequency}, "
            f"threshold={performance_threshold}, min_sharpe={min_sharpe}"
        )
    
    def retrain(self) -> dict:
        """
        Execute full retraining workflow.
        
        Returns:
            Dictionary with retraining results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 Starting Automatic Retraining")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Symbol: {self.config['symbol']}")
        logger.info(f"   Agent: {self.config['agent_type']}")
        logger.info(f"{'='*70}\n")
        
        try:
            # 1. Load and merge data
            logger.info("📥 Loading training data...")
            
            train_data, test_data = load_data(
                symbol=self.config['symbol'],
                start_date=self.config['training_settings']['start_date'],
                end_date=datetime.now().strftime('%Y-%m-%d'),  # Up to today
                features=self.config.get('features')
            )
            
            logger.info(f"✅ Data loaded: {len(train_data)} train, {len(test_data)} test\n")
            
            # 2. Train new model
            logger.info("🏋️ Training new model...")
            
            training_result = train_agent(self.config)
            
            if training_result['status'] != 'success':
                logger.error(f"❌ Training failed: {training_result}")
                return {
                    'status': 'failed',
                    'reason': 'training_failed',
                    'error': training_result.get('error')
                }
            
            new_model_path = training_result['model_path']
            new_version = training_result['version']
            
            logger.info(f"✅ New model trained: {new_version}\n")
            
            # 3. Backtest new model
            logger.info("📊 Backtesting new model...")
            
            backtest_results = run_backtest(
                model_path=new_model_path,
                test_data=test_data,
                agent_type=self.config['agent_type'],
                commission_settings=self.config.get('training_settings')
            )
            
            new_sharpe = backtest_results['metrics']['sharpe_ratio']
            new_return = backtest_results['metrics']['total_return']
            
            logger.info(f"📈 New model performance:")
            logger.info(f"   Sharpe: {new_sharpe:.2f}")
            logger.info(f"   Return: {new_return:.2f}%\n")
            
            # 4. Compare with previous best model
            logger.info("🔍 Comparing with previous best model...")
            
            prev_best = self.model_manager.get_best_model(
                agent_type=self.config['agent_type'],
                symbol=self.config['symbol'],
                metric='sharpe_ratio'
            )
            
            should_deploy = False
            reason = ""
            
            if prev_best is None:
                logger.info("ℹ️ No previous model found")
                should_deploy = True
                reason = "first_model"
            else:
                prev_sharpe = prev_best.get('metrics', {}).get('sharpe_ratio', 0.0)
                sharpe_improvement = new_sharpe - prev_sharpe
                
                logger.info(f"📊 Previous best Sharpe: {prev_sharpe:.2f}")
                logger.info(f"📊 Improvement: {sharpe_improvement:+.2f}")
                
                # Check deployment criteria
                if new_sharpe < self.min_sharpe:
                    reason = f"sharpe_too_low (< {self.min_sharpe})"
                    logger.warning(f"⚠️ New model Sharpe too low: {new_sharpe:.2f}")
                
                elif sharpe_improvement < self.performance_threshold:
                    reason = f"insufficient_improvement (< {self.performance_threshold})"
                    logger.warning(f"⚠️ Insufficient improvement: {sharpe_improvement:+.2f}")
                
                else:
                    should_deploy = True
                    reason = f"improved (Sharpe +{sharpe_improvement:.2f})"
                    logger.info("✅ New model meets deployment criteria")
            
            # 5. Deploy or discard
            if should_deploy:
                logger.info(f"\n✅ Deploying new model: {new_version}")
                logger.info(f"   Reason: {reason}")
                
                # Update metadata to mark as deployed
                metadata_update = {
                    'deployed': True,
                    'deployed_at': datetime.now().isoformat(),
                    'deployment_reason': reason,
                    'metrics': backtest_results['metrics']
                }

                self.model_manager.update_model_metadata(
                    agent_type=self.config['agent_type'],
                    symbol=self.config['symbol'],
                    version=new_version,
                    updates=metadata_update
                )
                
                # Cleanup old models (keep last 10)
                self.model_manager.cleanup_old_models(
                    agent_type=self.config['agent_type'],
                    symbol=self.config['symbol'],
                    keep_last=10
                )
                
                logger.info(f"\n{'='*70}")
                logger.info("✅ Retraining Complete: New model deployed!")
                logger.info(f"{'='*70}\n")
                
                return {
                    'status': 'success',
                    'action': 'deployed',
                    'model_version': new_version,
                    'model_path': new_model_path,
                    'metrics': backtest_results['metrics'],
                    'reason': reason
                }
            
            else:
                logger.info(f"\n⚠️ New model NOT deployed")
                logger.info(f"   Reason: {reason}")
                logger.info("   Keeping previous model\n")
                
                # Archive the new model (not good enough)
                model_id = f"{self.config['agent_type'].lower()}_{self.config['symbol']}_{new_version}"
                self.model_manager.archive_model(model_id)
                
                return {
                    'status': 'success',
                    'action': 'not_deployed',
                    'model_version': new_version,
                    'metrics': backtest_results['metrics'],
                    'reason': reason
                }
        
        except Exception as e:
            logger.error(f"\n❌ Retraining failed: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'reason': 'exception',
                'error': str(e)
            }
    
    def schedule(self) -> None:
        """
        Schedule retraining based on frequency.
        
        Runs in an infinite loop. Press Ctrl+C to stop.
        """
        if self.frequency == 'Daily':
            schedule.every().day.at("02:00").do(self.retrain)
            logger.info("📅 Retraining scheduled: Daily at 02:00")
        
        elif self.frequency == 'Weekly':
            schedule.every().monday.at("02:00").do(self.retrain)
            logger.info("📅 Retraining scheduled: Weekly on Monday at 02:00")
        
        elif self.frequency == 'Monthly':
            # Run on 1st of each month at 02:00
            schedule.every().month.at("02:00").do(self.retrain)
            logger.info("📅 Retraining scheduled: Monthly on 1st at 02:00")
        
        else:
            logger.error(f"❌ Unknown frequency: {self.frequency}")
            return
        
        logger.info("🔄 Scheduler started. Press Ctrl+C to stop.\n")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\n⚠️ Scheduler stopped by user")


if __name__ == '__main__':
    """
    Example usage
    """
    
    # Example configuration
    config = {
        'agent_type': 'PPO',
        'symbol': 'AAPL',
        'hyperparameters': {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'batch_size': 256,
            'n_steps': 2048,
            'episodes': 10000
        },
        'features': {
            'price': True,
            'volume': True,
            'rsi': True,
            'macd': True,
            'ema_10': True,
            'ema_50': True,
            'vix': True
        },
        'training_settings': {
            'start_date': '2023-01-01',
            'end_date': '2024-11-01',
            'commission': {
                'per_share': 0.01,
                'min_fee': 2.5,
                'max_pct': 0.01,
            },
            'commission_per_share': 0.01,
            'commission_min_fee': 2.5,
            'commission_max_pct': 0.01,
            'initial_cash': 100000
        }
    }
    
    # Create scheduler
    scheduler = RetrainingScheduler(
        config=config,
        frequency='Weekly',
        performance_threshold=0.2,  # Require +0.2 Sharpe improvement
        min_sharpe=1.5  # Minimum Sharpe ratio to deploy
    )
    
    # Test single retraining run
    print("🧪 Testing single retraining run...\n")
    result = scheduler.retrain()
    
    print("\n📊 Retraining Result:")
    import json
    print(json.dumps(result, indent=2))
    
    # To run on schedule, uncomment:
    # scheduler.schedule()
