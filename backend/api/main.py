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

# Load environment variables from .env file
load_dotenv()

# Add backend directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

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
    Download and prepare training data
    
    Request body:
    {
        "symbol": "IWM",
        "period": "5y",
        "enable_sentiment": false,
        "enable_multi_asset": false,
        "force_redownload": false
    }
    
    Smart caching behavior:
    - If force_redownload=false (default):
      * Checks cache, if fresh (< 1 day old) uses it
      * If cache exists but old, downloads only new data since last update
      * Merges old + new data
    - If force_redownload=true:
      * Always downloads full dataset from scratch
    
    Returns:
    {
        "status": "success",
        "symbol": "IWM",
        "rows": 1258,
        "features": 15,
        "train_size": 1006,
        "test_size": 252,
        "cache_status": "incremental_update",  // or "full_download" or "cache_hit"
        "files": {
            "raw": "d:/YoopRL/data/training/IWM_raw.csv",
            "train": "d:/YoopRL/data/training/IWM_train.csv",
            "test": "d:/YoopRL/data/training/IWM_test.csv"
        }
    }
    """
    try:
        from data_download.loader import prepare_training_data, save_prepared_data
        
        # Get parameters from request
        data = request.json or {}
        symbol = data.get('symbol', 'IWM')
        period = data.get('period', '5y')
        enable_sentiment = data.get('enable_sentiment', False)
        enable_multi_asset = data.get('enable_multi_asset', False)
        force_redownload = data.get('force_redownload', False)
        
        logger.info(f"Starting data download for {symbol} ({period}), force_redownload={force_redownload}...")
        
        # Download and prepare data
        prepared, split = prepare_training_data(
            symbol=symbol,
            period=period,
            enable_sentiment=enable_sentiment,
            enable_multi_asset=enable_multi_asset,
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
            'files': files
        }), 200
        
    except Exception as e:
        logger.error(f"Error downloading training data: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
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
