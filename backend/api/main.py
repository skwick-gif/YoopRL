"""
Backend API for YoopRL Trading System
Provides REST API for database operations

This Flask API handles:
- Equity data persistence (save and retrieve)
- Performance metrics
- Agent actions logging
- Risk events tracking

Runs on port 8000 to avoid conflict with IBKR Bridge (port 5080)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
import uuid
import json
import pandas as pd
from datetime import datetime, timedelta
import threading

# Load environment variables from .env file
load_dotenv()

# Add backend directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from training.train import train_agent, load_data
from models.model_manager import ModelManager
from evaluation.backtester import run_backtest
from utils.state_normalizer import StateNormalizer
from execution import agent_manager as live_agent_manager
from monitoring.routes import register_monitoring_routes
from api.live_routes import register_live_routes
from data_download.intraday_loader import ALLOWED_INTRADAY_SYMBOLS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize database
db = DatabaseManager()
live_agent_manager.attach_database(db)
register_monitoring_routes(app, db, logger)

# Initialize model manager
# Use absolute path or relative from project root
import os
from pathlib import Path

# Get project root (YoopRL directory)
project_root = Path(__file__).parent.parent.parent  # main.py -> api -> backend -> YoopRL
models_dir = project_root / 'backend' / 'models'

print(f"[DEBUG] Models directory: {models_dir}")
print(f"[DEBUG] Models directory exists: {models_dir.exists()}")

model_manager = ModelManager(base_dir=str(models_dir))
register_live_routes(app, db, live_agent_manager, model_manager, logger)

# Global dictionary to track training sessions
training_sessions = {}
training_threads = {}


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Returns: System status and database stats
    """
    try:
        stats = db.get_database_stats()
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'stats': stats
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/api/equity/save', methods=['POST'])
def save_equity():
    """
    Save equity data point
    
    Request body:
    {
        "net_liquidation": 100000.0,
        "buying_power": 400000.0,
        "cash": 50000.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "gross_position_value": 50000.0
    }
    """
    try:
        data = request.json
        
        # Extract required field
        net_liquidation = data.get('net_liquidation')
        if net_liquidation is None:
            return jsonify({'error': 'net_liquidation is required'}), 400
        
        # Extract optional fields
        buying_power = data.get('buying_power')
        cash = data.get('cash')
        unrealized_pnl = data.get('unrealized_pnl')
        realized_pnl = data.get('realized_pnl')
        gross_position_value = data.get('gross_position_value')
        
        # Save to database
        db.save_equity_point(
            net_liquidation=net_liquidation,
            buying_power=buying_power,
            cash=cash,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            gross_position_value=gross_position_value
        )
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        logger.error(f"Error saving equity point: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/date-range', methods=['GET'])
def get_training_date_range():
    """Return cached date bounds for requested training symbol."""

    symbol = (request.args.get('symbol') or '').strip().upper()
    frequency = (request.args.get('frequency') or 'daily').strip().lower()
    interval = (request.args.get('interval') or '1d').strip().lower()

    if not symbol:
        return jsonify({
            'status': 'error',
            'error': 'Symbol is required'
        }), 400

    try:
        bounds = None
        source = 'market_data'

        if frequency == 'intraday':
            interval = interval or '15m'
            bounds = db.get_intraday_session_bounds(symbol, interval)
            source = 'intraday_market_data'
        else:
            bounds = db.get_market_date_bounds(symbol)
            interval = '1d'

        if not bounds:
            return jsonify({
                'status': 'not_found',
                'symbol': symbol,
                'frequency': frequency,
                'interval': interval
            }), 404

        start_date = bounds.get('min_date') or bounds.get('start_date')
        end_date = bounds.get('max_date') or bounds.get('end_date')

        start = pd.to_datetime(start_date).date().isoformat() if start_date else None
        end = pd.to_datetime(end_date).date().isoformat() if end_date else None

        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'frequency': 'intraday' if frequency == 'intraday' else 'daily',
            'interval': interval,
            'start_date': start,
            'end_date': end,
            'source': source
        }), 200
    except Exception as exc:
        logger.error('Failed to resolve training date range for %s: %s', symbol, exc)
        return jsonify({
            'status': 'error',
            'error': str(exc)
        }), 500


@app.route('/api/equity/history', methods=['GET'])
def get_equity_history():
    """
    Get equity history for charting
    
    Query parameters:
    - hours: Get last N hours (optional)
    - start_date: Start date ISO format (optional)
    - end_date: End date ISO format (optional)
    
    Returns:
    [
        {
            "timestamp": 1699459200.0,
            "datetime": "2024-11-08T14:00:00",
            "net_liquidation": 100000.0,
            "buying_power": 400000.0,
            ...
        },
        ...
    ]
    """
    try:
        hours = request.args.get('hours', type=int)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        history = db.get_equity_history(
            hours=hours,
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify(history), 200
        
    except Exception as e:
        logger.error(f"Error retrieving equity history: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/stats', methods=['GET'])
def get_database_stats():
    """
    Get database statistics
    
    Returns:
    {
        "equity_history_count": 12345,
        "agent_actions_count": 567,
        "oldest_equity_date": "2024-01-01T00:00:00",
        "newest_equity_date": "2024-11-08T14:00:00",
        "db_size_mb": 15.5
    }
    """
    try:
        stats = db.get_database_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/cleanup', methods=['POST'])
def cleanup_database():
    """
    Manually trigger database cleanup
    Removes data older than retention period
    
    Returns:
    {
        "equity_deleted": 1000,
        "actions_deleted": 500,
        "logs_deleted": 2000,
        "cutoff_date": "2023-11-08T14:00:00"
    }
    """
    try:
        result = db.cleanup_old_data()
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/training/download', methods=['POST'])
def download_training_data():
    """
    Download and prepare training data with date-based incremental updates
    
    Request body:
    {
        "symbol": "IWM",
        "start_date": "2020-01-01",
        "end_date": "2024-11-08",
        "enable_sentiment": false,
        "enable_social_media": false,
        "enable_news": false,
        "enable_market_events": false,
        "enable_fundamental": false,
        "enable_multi_asset": false,
        "enable_macro": false,
        "force_redownload": false
    }
    
    Smart caching behavior:
    - Downloads data from start_date to end_date
    - Checks SQL for existing data
    - Only downloads missing date ranges (incremental update)
    - Merges with existing SQL data
    
    Returns:
    {
        "status": "success",
        "symbol": "IWM",
        "rows": 1258,
        "features": 15,
        "train_size": 1006,
        "test_size": 252,
        "date_range": "2020-01-01 to 2024-11-08",
        "cache_status": "incremental_update",
        "files": {...}
    }
    """
    try:
        from data_download.loader import prepare_training_data, save_prepared_data
        
        # Get parameters from request
        data = request.json or {}
        symbol = data.get('symbol', 'IWM')
        start_date = data.get('start_date', '2020-01-01')
        end_date = data.get('end_date', pd.Timestamp.now().strftime('%Y-%m-%d'))
        enable_sentiment = data.get('enable_sentiment', False)
        enable_social_media = data.get('enable_social_media', False)
        enable_news = data.get('enable_news', False)
        enable_market_events = data.get('enable_market_events', False)
        enable_fundamental = data.get('enable_fundamental', False)
        enable_multi_asset = data.get('enable_multi_asset', False)
        enable_macro = data.get('enable_macro', False)
        force_redownload = data.get('force_redownload', False)
        
        logger.info(f"Starting data download for {symbol} ({start_date} to {end_date})...")
        
        # Calculate period from dates (for backward compatibility)
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
        
        # Download and prepare data
        prepared, split = prepare_training_data(
            symbol=symbol,
            period=period,
            enable_sentiment=enable_sentiment,
            enable_social_media=enable_social_media,
            enable_news=enable_news,
            enable_market_events=enable_market_events,
            enable_fundamental=enable_fundamental,
            enable_multi_asset=enable_multi_asset,
            enable_macro=enable_macro,
            force_redownload=force_redownload,
        )
        
        # Save to disk
        output_dir = Path("d:/YoopRL/data/training")
        files = save_prepared_data(prepared, split, output_dir, symbol)
        
        logger.info(f"Training data prepared successfully for {symbol}")
        
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'rows': len(prepared.raw),
            'features': len(prepared.feature_names),
            'train_size': len(split.train),
            'test_size': len(split.test),
            'date_range': f"{start_date} to {end_date}",
            'files': files
        }), 200
        
    except Exception as e:
        logger.error(f"Error downloading training data: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/training/train', methods=['POST'])
def start_training():
    """
    Start a new training session
    
    Request body:
    {
        "agent_type": "PPO" or "SAC",
        "symbol": "AAPL",
        "hyperparameters": {
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "batch_size": 256,
            "n_steps": 2048,
            "episodes": 50000
        },
        "features": {
            "price": true,
            "volume": true,
            "rsi": true,
            ...
        },
        "training_settings": {
            "start_date": "2023-01-01",
            "end_date": "2024-11-01",
            "commission": 1.0,
            "initial_cash": 100000
        }
    }
    
    Returns:
    {
        "status": "success",
        "training_id": "uuid",
        "message": "Training started"
    }
    """
    try:
        data = request.json
        
        # Determine whether request targets intraday pipeline and validate symbol gating
        training_settings = data.get('training_settings') or {}
        agent_type = str(data.get('agent_type', '')).upper()
        interval = str(training_settings.get('interval', '')).lower()
        frequency = str(training_settings.get('data_frequency', '')).lower()
        intraday_flag = bool(training_settings.get('intraday_enabled', False))
        wants_intraday = any([
            agent_type == 'SAC_INTRADAY_DSR',
            intraday_flag,
            interval in {'15m', '15min'},
            frequency in {'intraday', '15m', '15min'},
        ])

        if wants_intraday:
            symbol = str(data.get('symbol') or training_settings.get('symbol', '')).upper()
            if symbol not in ALLOWED_INTRADAY_SYMBOLS:
                allowed_list = ', '.join(sorted(ALLOWED_INTRADAY_SYMBOLS))
                return jsonify({
                    'status': 'error',
                    'message': (
                        f"Intraday training currently supports only the following symbols: {allowed_list}"
                    ),
                    'error': 'unsupported_intraday_symbol',
                }), 400

        # Generate unique training ID
        training_id = str(uuid.uuid4())
        
        # Initialize session tracking
        training_sessions[training_id] = {
            'status': 'starting',
            'progress': 0,
            'current_timestep': 0,
            'current_reward': 0.0,
            'started_at': datetime.now().isoformat(),
            'config': data
        }
        
        # Define background task
        def run_training():
            try:
                training_sessions[training_id]['status'] = 'running'
                
                # Run training
                result = train_agent(data)
                
                # Update session with result
                training_sessions[training_id]['status'] = 'completed' if result['status'] == 'success' else 'failed'
                training_sessions[training_id]['result'] = result
                training_sessions[training_id]['completed_at'] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"Training failed: {e}", exc_info=True)
                training_sessions[training_id]['status'] = 'failed'
                training_sessions[training_id]['error'] = str(e)
                training_sessions[training_id]['completed_at'] = datetime.now().isoformat()
        
        # Start training in background thread
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        training_threads[training_id] = thread
        
        logger.info(f"Training session {training_id} started for {data.get('symbol')} ({data.get('agent_type')})")
        
        return jsonify({
            'status': 'success',
            'training_id': training_id,
            'message': 'Training started'
        }), 200
        
    except Exception as e:
        logger.error(f"Error starting training: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error': str(e)
        }), 500


@app.route('/api/training/progress/<training_id>', methods=['GET'])
def get_training_progress(training_id):
    """
    Get training progress for a session
    
    Returns:
    {
        "status": "running",
        "progress": 45.5,
        "current_timestep": 22750,
        "current_reward": 125.6,
        "started_at": "2024-11-08T14:00:00",
        "config": {...}
    }
    """
    try:
        if training_id not in training_sessions:
            return jsonify({
                'status': 'error',
                'error': 'Training session not found'
            }), 404
        
        session = training_sessions[training_id]
        
        # Read progress from file (updated by TrainingProgressCallback)
        progress_file = Path('training_progress.json')
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                session['progress'] = progress_data.get('progress_pct', 0)
                session['current_timestep'] = progress_data.get('timestep', 0)
                session['current_reward'] = progress_data.get('episode_reward', 0.0)
                session['episode_count'] = progress_data.get('episode_count', 0)
            except:
                pass
        
        return jsonify(session), 200
        
    except Exception as e:
        logger.error(f"Error getting training progress: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """
    Stop a training session
    
    Request body:
    {
        "training_id": "uuid"
    }
    
    Returns:
    {
        "status": "success",
        "message": "Training stopped"
    }
    """
    try:
        data = request.json
        training_id = data.get('training_id')
        
        if not training_id:
            return jsonify({
                'status': 'error',
                'error': 'training_id is required'
            }), 400
        
        if training_id not in training_sessions:
            return jsonify({
                'status': 'error',
                'error': 'Training session not found'
            }), 404
        
        # Update session status
        training_sessions[training_id]['status'] = 'stopped'
        training_sessions[training_id]['stopped_at'] = datetime.now().isoformat()
        
        # Note: Actually stopping the training thread is complex
        # For now we just mark it as stopped
        # TODO: Implement graceful stop with checkpoint saving
        
        logger.info(f"Training session {training_id} stopped")
        
        return jsonify({
            'status': 'success',
            'message': 'Training stopped'
        }), 200
        
    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/training/models', methods=['GET'])
def list_models():
    """
    List all available trained models
    
    Query parameters:
    - agent_type: Filter by agent type (optional)
    - symbol: Filter by symbol (optional)
    
    Returns:
    {
        "status": "success",
        "models": [
            {
                "model_id": "ppo_AAPL_v20241108_140000",
                "agent_type": "PPO",
                "symbol": "AAPL",
                "version": "v20241108_140000",
                "created": "2024-11-08T14:00:00",
                "metrics": {
                    "sharpe_ratio": 1.85,
                    "total_return": 25.6
                }
            },
            ...
        ]
    }
    """
    try:
        agent_type = request.args.get('agent_type')
        symbol = request.args.get('symbol')
        
        # Get models from model manager
        models = model_manager.list_models(
            agent_type=agent_type.lower() if agent_type else None,
            symbol=symbol
        )
        
        return jsonify({
            'status': 'success',
            'models': models,
            'count': len(models)
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/training/load_model', methods=['POST'])
def load_model():
    """
    Load a specific model
    
    Request body:
    {
        "model_id": "ppo_AAPL_v20241108_140000"
    }
    
    Returns:
    {
        "status": "success",
        "model": {
            "model_id": "ppo_AAPL_v20241108_140000",
            "agent_type": "PPO",
            "metadata": {...}
        }
    }
    """
    try:
        data = request.json
        model_id = data.get('model_id')
        
        if not model_id:
            return jsonify({
                'status': 'error',
                'error': 'model_id is required'
            }), 400
        
        # Load model metadata
        model_info = model_manager.get_model_info(model_id)
        
        if not model_info:
            return jsonify({
                'status': 'error',
                'error': 'Model not found'
            }), 404
        
        return jsonify({
            'status': 'success',
            'model': model_info
        }), 200
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/training/save_config', methods=['POST'])
def save_config():
    """
    Save training configuration
    
    Request body:
    {
        "name": "my_config",
        "agent_type": "PPO",
        "config": {
            "hyperparameters": {...},
            "features": {...},
            "training_settings": {...}
        }
    }
    
    Returns:
    {
        "status": "success",
        "config_file": "configs/my_config_PPO.json"
    }
    """
    try:
        data = request.json
        name = data.get('name', 'config')
        agent_type = data.get('agent_type', 'PPO')
        config = data.get('config', {})
        
        # Create configs directory
        config_dir = Path('backend/configs')
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_file = config_dir / f"{name}_{agent_type}.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved: {config_file}")
        
        return jsonify({
            'status': 'success',
            'config_file': str(config_file)
        }), 200
        
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/training/load_config/<config_name>', methods=['GET'])
def load_config(config_name):
    """
    Load training configuration
    
    Returns:
    {
        "status": "success",
        "config": {
            "hyperparameters": {...},
            "features": {...},
            "training_settings": {...}
        }
    }
    """
    try:
        config_file = Path(f'backend/configs/{config_name}.json')
        
        if not config_file.exists():
            return jsonify({
                'status': 'error',
                'error': 'Configuration not found'
            }), 404
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return jsonify({
            'status': 'success',
            'config': config
        }), 200
        
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/training/backtest', methods=['POST'])
def run_backtest_endpoint():
    """
    Run backtest on a trained model
    
    Request body:
    {
        "model_path": "backend/models/ppo/ppo_AAPL_v20241108_140000.zip",
        "agent_type": "PPO",
        "symbol": "AAPL",
        "start_date": "2024-01-01",
        "end_date": "2024-11-01"
    }
    
    Returns:
    {
        "status": "success",
        "metrics": {
            "sharpe_ratio": 1.85,
            "sortino_ratio": 2.10,
            "max_drawdown": -12.5,
            "win_rate": 65.5,
            "profit_factor": 1.85,
            "total_return": 25.6,
            "buy_and_hold_return": 18.2,
            "alpha": 7.4,
            ...
        },
        "results_file": "backend/results/backtest_PPO_20241108_140000.json"
    }
    """
    try:
        data = request.json
        
        model_path = data.get('model_path')
        agent_type = data.get('agent_type', 'PPO')
        symbol = data.get('symbol')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not model_path:
            return jsonify({
                'status': 'error',
                'error': 'model_path is required'
            }), 400

        # Resolve model path to absolute location on disk
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = (project_root / model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
        model_metadata = None
        if metadata_path.exists():
            try:
                with metadata_path.open('r', encoding='utf-8') as handle:
                    model_metadata = json.load(handle)
            except Exception as meta_exc:  # pragma: no cover - logging only
                logger.warning(f"Failed to load metadata for %s: %s", model_path.name, meta_exc)

        def _metadata_metrics(payload: dict | None) -> dict:
            if not payload:
                return {}
            if isinstance(payload.get('metrics'), dict) and payload['metrics']:
                return payload['metrics']

            metric_map = {
                'sharpe_ratio': ['sharpe_ratio'],
                'sortino_ratio': ['sortino_ratio'],
                'max_drawdown': ['max_drawdown'],
                'win_rate': ['win_rate'],
                'total_return': ['total_return'],
                'final_portfolio_value': ['final_balance', 'final_portfolio_value'],
                'num_trades': ['total_trades', 'num_trades'],
            }

            metrics = {}
            for target, candidates in metric_map.items():
                for key in candidates:
                    if key in payload and payload[key] is not None:
                        metrics[target] = payload[key]
                        break

            if 'avg_trade_return' not in metrics:
                trade_history = payload.get('trade_history') or []
                if isinstance(trade_history, list) and trade_history:
                    try:
                        metrics['avg_trade_return'] = sum(trade_history) / len(trade_history)
                    except TypeError:
                        pass

            return metrics

        def _is_intraday_request() -> bool:
            agent_tag = str(agent_type or '').upper()
            if 'INTRADAY' in agent_tag:
                return True

            data_freq = str(data.get('data_frequency', '')).lower()
            interval_hint = str(data.get('interval', '')).lower()
            if data_freq == 'intraday' or interval_hint in {'15m', '15min'}:
                return True

            if not model_metadata:
                return False

            settings = model_metadata.get('training_settings') or {}
            if str(settings.get('data_frequency', '')).lower() == 'intraday':
                return True
            if str(settings.get('interval', '')).lower() in {'15m', '15min'}:
                return True
            if bool(settings.get('intraday_enabled')):
                return True

            reward_mode = str(settings.get('reward_mode', '')).lower()
            return reward_mode == 'dsr'

        if _is_intraday_request():
            metrics_payload = _metadata_metrics(model_metadata)
            if not metrics_payload:
                return jsonify({
                    'status': 'error',
                    'error': 'Intraday model metadata missing metrics. Cannot provide backtest summary.'
                }), 400

            total_trades = metrics_payload.get('num_trades')
            if total_trades is None and model_metadata is not None:
                total_trades = model_metadata.get('total_trades')
                if total_trades is None and isinstance(model_metadata.get('trade_history'), list):
                    total_trades = len(model_metadata['trade_history'])

            response_payload = {
                'status': 'success',
                'source': 'metadata',
                'intraday': True,
                'metrics': metrics_payload,
                'total_trades': total_trades,
                'results_file': None,
                'message': 'Returned stored metrics for intraday model (backtest replay unavailable)'
            }

            if model_metadata is not None:
                final_value = metrics_payload.get('final_portfolio_value')
                if final_value is None:
                    final_value = model_metadata.get('final_balance') or model_metadata.get('final_portfolio_value')
                if final_value is not None:
                    response_payload['final_portfolio_value'] = final_value

            return jsonify(response_payload), 200

        # Load test data
        # TODO: Replace with actual data loading from SQL
        from training.train import load_data

        logger.info(f"Loading test data for {symbol} ({start_date} to {end_date})...")
        _, test_data = load_data(symbol, start_date, end_date)
        
        logger.info(f"Running backtest on {model_path}...")

        commission_payload = None
        if model_metadata is not None:
            commission_payload = (
                model_metadata.get('commission_config')
                or model_metadata.get('training_settings')
            )
        commission_payload = commission_payload or data.get('training_settings')
        
        # Run backtest
        results = run_backtest(
            model_path=str(model_path),
            test_data=test_data,
            agent_type=agent_type,
            save_path=None,  # Don't save for now
            commission_settings=commission_payload,
        )
        
        # Optionally save results
        results_file = None
        if data.get('save_results', False):
            results_file = f"backend/results/backtest_{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            from evaluation.backtester import Backtester
            backtester = Backtester(
                model_path,
                test_data,
                agent_type,
                commission_settings=commission_payload,
            )
            backtester.save_results(results, results_file)
        
        logger.info(f"Backtest complete. Sharpe: {results['metrics']['sharpe_ratio']}, Return: {results['metrics']['total_return']}%")
        
        return jsonify({
            'status': 'success',
            'metrics': results['metrics'],
            'results_file': results_file,
            'equity_curve': results['equity_curve'][:100],  # First 100 points only
            'total_trades': len(results['trades'])
        }), 200
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
@app.route('/api/training/drift_status', methods=['GET'])
def check_drift_status():
    """
    Check for data drift in recent market data
    
    Query parameters:
    - symbol: Stock/ETF symbol (required)
    - agent_type: PPO or SAC (required)
    - days: Number of recent days to check (default 30)
    
    Returns:
    {
        "status": "success",
        "drift_detected": true/false,
        "severity": "medium/high/critical",
        "affected_features": ["rsi", "macd"],
        "drift_scores": {...},
        "needs_retraining": true/false,
        "recommendation": "Retrain model with recent data",
        "last_check": "2024-11-08T14:00:00"
    }
    """
    try:
        symbol = request.args.get('symbol')
        agent_type = request.args.get('agent_type', 'PPO')
        days = request.args.get('days', 30, type=int)
        
        if not symbol:
            return jsonify({
                'status': 'error',
                'error': 'symbol is required'
            }), 400
        
        logger.info(f"Checking drift for {symbol} ({agent_type}) - last {days} days")
        
        # Load normalizer parameters (from training)
        normalizer_path = f"backend/models/normalizer_{symbol}_{agent_type}.json"
        
        if not Path(normalizer_path).exists():
            return jsonify({
                'status': 'success',
                'drift_detected': False,
                'severity': 'unknown',
                'affected_features': [],
                'drift_scores': {},
                'needs_retraining': False,
                'recommendation': 'No trained model found. Train a model first to enable drift detection.',
                'last_check': datetime.now().isoformat()
            }), 200
        
        # Load normalizer
        normalizer = StateNormalizer(method='zscore')
        normalizer.load_params(normalizer_path)
        
        # Download recent market data
        try:
            import yfinance as yf
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(symbol)
            recent_df = ticker.history(start=start_date, end=end_date)
            
            if recent_df.empty:
                return jsonify({
                    'status': 'success',
                    'drift_detected': False,
                    'severity': 'unknown',
                    'affected_features': [],
                    'drift_scores': {},
                    'needs_retraining': False,
                    'recommendation': f'No recent data available for {symbol}',
                    'last_check': datetime.now().isoformat()
                }), 200
            
            # Prepare features (simple version - just use available columns)
            # Map yfinance columns to our feature names
            feature_mapping = {
                'Close': 'price',
                'Volume': 'volume'
            }
            
            recent_data = pd.DataFrame()
            for yf_col, feature_name in feature_mapping.items():
                if yf_col in recent_df.columns:
                    recent_data[feature_name] = recent_df[yf_col]
            
            # Add dummy features for now (TODO: calculate real technical indicators)
            if 'price' in recent_data.columns:
                recent_data['vix'] = 20.0  # Placeholder
            
            # Only use features that exist in both training and current data
            available_features = [col for col in recent_data.columns if col in ['price', 'volume', 'vix']]
            
            if not available_features:
                return jsonify({
                    'status': 'success',
                    'drift_detected': False,
                    'severity': 'unknown',
                    'affected_features': [],
                    'drift_scores': {},
                    'needs_retraining': False,
                    'recommendation': 'Insufficient features for drift detection',
                    'last_check': datetime.now().isoformat()
                }), 200
            
            recent_features = recent_data[available_features].values
            
        except Exception as data_error:
            logger.error(f"Error loading recent data: {data_error}")
            return jsonify({
                'status': 'error',
                'error': f'Failed to load recent market data: {str(data_error)}',
                'drift_detected': False,
                'needs_retraining': False
            }), 500
        
        # Check drift
        drift_results = normalizer.check_drift_status(
            recent_data=recent_features,
            feature_names=available_features,
            threshold=0.4
        )
        
        logger.info(f"Drift check complete: drift_detected={drift_results.get('drift_detected')}, "
                   f"severity={drift_results.get('severity')}")
        
        return jsonify(drift_results), 200
        
    except Exception as e:
        logger.error(f"Error checking drift: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'needs_retraining': False
        }), 500
if __name__ == '__main__':
    logger.info("Starting YoopRL Backend API on port 8000...")
    logger.info("Database initialized with auto-cleanup enabled")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,  # Set to True for development
        threaded=True
    )
