"""
Data Download Package for YoopRL Trading System

This module handles:
- Market data download from Yahoo Finance
- Feature engineering (technical indicators, sentiment, etc.)
- Data preparation for RL agent training
- Multi-asset correlation features
- Live market data snapshots
"""

from .loader import download_history, prepare_training_data
from .config import DownloadConfig
# Lazy import for sentiment to avoid NLTK blocking
# from .sentiment_features import SentimentFeatureAggregator
# from .sentiment_service import EnterpriseSentimentAnalyzer
from .multi_asset_loader import download_histories, prepare_multi_asset_features
from .live_data_helpers import (
    snapshot_fetch,
    save_snapshot,
    load_snapshots,
    bar_interval_seconds,
    next_fetch_time
)

__all__ = [
    # Data loading
    'download_history', 
    'prepare_training_data',
    'download_histories',
    'prepare_multi_asset_features',
    
    # Configuration
    'DownloadConfig',
    
    # Sentiment
    'SentimentFeatureAggregator',
    'EnterpriseSentimentAnalyzer',
    
    # Live data
    'snapshot_fetch',
    'save_snapshot',
    'load_snapshots',
    'bar_interval_seconds',
    'next_fetch_time',
]
