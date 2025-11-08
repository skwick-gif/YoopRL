"""
Market Data Loading and Preprocessing

This module handles:
- Downloading historical OHLCV data from Yahoo Finance with SQL-backed incremental updates
- Preparing features using FeatureEngineering
- Splitting data into train/test sets
"""

import sys
from pathlib import Path

# Add backend directory to path for absolute imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from dataclasses import dataclass
from typing import Tuple, Optional
import pandas as pd
import yfinance as yf
import numpy as np
import logging

from data_download.config import TrainingDataConfig
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Global database manager instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get or create database manager singleton"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


@dataclass
class PreparedData:
    """
    Container for prepared training data
    
    Attributes:
        raw: Original OHLCV data from Yahoo Finance
        processed: Data with technical indicators and features
        normalized: Normalized features ready for training
        feature_names: List of feature column names
    """
    raw: pd.DataFrame
    processed: pd.DataFrame
    normalized: pd.DataFrame
    feature_names: list


@dataclass
class DataSplit:
    """
    Train/test split container
    
    Attributes:
        train: Training dataset (usually 80% of data)
        test: Testing dataset (usually 20% of data)
    """
    train: pd.DataFrame
    test: pd.DataFrame


def download_history(
    symbol: str, 
    period: str = "5y", 
    progress: bool = False,
    cache_dir: Optional[Path] = None,
    force_redownload: bool = False
) -> pd.DataFrame:
    """
    Download historical OHLCV data from Yahoo Finance with SQL-backed incremental updates
    
    Smart incremental update strategy:
    1. Check SQL database for latest date
    2. If data exists and is current (today), return from SQL
    3. If data exists but outdated, download only missing days
    4. If no data exists, download full period
    5. Save all new data to SQL for future incremental updates
    
    Args:
        symbol: Stock symbol (e.g., "IWM", "SPY", "AAPL")
        period: Time period ("1y", "2y", "5y", "max") - used only for initial download
        progress: Show download progress bar
        cache_dir: Unused (kept for backward compatibility)
        force_redownload: Force full redownload and replace SQL data
                 
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        
    Raises:
        RuntimeError: If no data is returned (invalid symbol or period)
    """
    db = get_db_manager()
    
    # Check if we have data in SQL database
    latest_date = db.get_latest_market_date(symbol)
    
    if latest_date and not force_redownload:
        # Check if data is current (updated today)
        today = pd.Timestamp.now().date()
        days_old = (today - latest_date).days
        
        if days_old == 0:
            logger.info(f"Using SQL data for {symbol} (updated today)")
            return db.get_market_data(symbol)
        
        logger.info(f"Found SQL data for {symbol} (last update: {latest_date}, {days_old} days old)")
        logger.info(f"Downloading only new data since {latest_date}...")
        
        try:
            # Download only new data since last SQL date
            ticker = yf.Ticker(symbol)
            new_df = ticker.history(
                start=latest_date + pd.Timedelta(days=1),  # Start from day after last SQL date
                end=pd.Timestamp.now(),
                auto_adjust=False,
                actions=False,
            )
            
            if new_df.empty:
                logger.info(f"No new data available for {symbol}, using SQL data")
                return db.get_market_data(symbol)
            
            # Handle column names (yfinance sometimes changes format)
            if isinstance(new_df.columns, pd.MultiIndex):
                new_df.columns = new_df.columns.get_level_values(0)
            
            # Save new data to SQL
            db.save_market_data(symbol, new_df)
            logger.info(f"Added {len(new_df)} new rows to {symbol} SQL data")
            
            # Return complete dataset from SQL (old + new merged)
            merged_df = db.get_market_data(symbol)
            logger.info(f"Total {symbol} data: {len(merged_df)} rows")
            
            return merged_df
            
        except Exception as e:
            logger.warning(f"Failed to download incremental data: {e}, using SQL data")
            return db.get_market_data(symbol)
    
    # No SQL data or force redownload - download full dataset
    logger.info(f"Downloading full {symbol} dataset for period {period}...")
    
    try:
        df = yf.download(
            symbol,
            period=period,
            progress=progress,
            auto_adjust=False,  # Keep both Close and Adj Close
            actions=False,  # Don't include dividends/splits
            group_by="column",  # Group by column names (not tickers)
        )
        
        # Handle MultiIndex columns (sometimes yfinance returns this)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            raise RuntimeError(
                f"No data returned for {symbol} (period: {period}). "
                f"Check if symbol exists and period is valid."
            )
        
        # Clean up: Remove rows with NaT index (corrupted data)
        df = df[df.index.notna()]
        
        logger.info(f"Successfully downloaded {len(df)} bars for {symbol}")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Save to SQL database
        db.save_market_data(symbol, df)
        logger.info(f"Saved {len(df)} rows to SQL database")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {e}")
        raise


def prepare_training_data(
    symbol: str,
    period: str = "5y",
    train_test_split: float = 0.8,
    cache_dir: Optional[Path] = None,
    enable_sentiment: bool = False,
    enable_social_media: bool = False,
    enable_news: bool = False,
    enable_market_events: bool = False,
    enable_fundamental: bool = False,
    enable_multi_asset: bool = False,
    enable_macro: bool = False,
    multi_asset_symbols: Optional[list] = None,
    force_redownload: bool = False,
    feature_config: Optional[dict] = None,  # NEW: Feature selection from UI
) -> Tuple[PreparedData, DataSplit]:
    """
    Complete pipeline: download → features → normalize → split
    
    Args:
        symbol: Stock symbol to download
        period: Time period for historical data
        train_test_split: Fraction for training (0.8 = 80% train, 20% test)
        cache_dir: Directory for caching multi-asset/sentiment data
        enable_sentiment: LEGACY - Include sentiment features (use enable_social_media/enable_news instead)
        enable_social_media: Include social media sentiment (Reddit, StockTwits, Google Trends)
        enable_news: Include news sentiment (Alpha Vantage, Finnhub, NewsAPI)
        enable_market_events: Include market events (earnings dates, dividends, splits) - IMPLEMENTED via yfinance
        enable_fundamental: Include fundamental data (P/E, EPS, margins, debt/equity) - IMPLEMENTED via yfinance
        enable_multi_asset: Include cross-asset correlation features
        enable_macro: Include macro indicators (VIX, yields, DXY, oil, gold) - IMPLEMENTED via yfinance + FRED
        multi_asset_symbols: List of symbols for cross-asset features
        feature_config: Feature selection config from UI (includes macro, technical indicators)
                            
    Returns:
        Tuple of (PreparedData, DataSplit)
        - PreparedData: Contains raw, processed, normalized data
        - DataSplit: Contains train/test split
        
    Example:
        >>> data, split = prepare_training_data("IWM", period="5y", enable_news=True)
        >>> print(f"Train shape: {split.train.shape}")
        >>> print(f"Test shape: {split.test.shape}")
        >>> print(f"Features: {data.feature_names[:10]}")
    """
    # Combine sentiment flags: Social Media + News both use sentiment_data table
    enable_sentiment_combined = enable_sentiment or enable_social_media or enable_news
    
    # Update feature_config to include fundamentals, events, macro
    if feature_config is None:
        feature_config = {}
    
    # Add flags to feature_config for selective computation
    if enable_fundamental:
        feature_config['fundamental'] = {'enabled': True}
    if enable_market_events:
        feature_config['market_events'] = {'enabled': True}
    if enable_macro:
        feature_config['macro'] = {'enabled': True}
    
    # Step 1: Download raw data from Yahoo Finance (with smart caching)
    logger.info(f"Step 1/4: Downloading {symbol} data...")
    raw_df = download_history(
        symbol, 
        period=period, 
        cache_dir=cache_dir,
        force_redownload=force_redownload
    )
    
    # Step 2: Feature engineering (technical indicators)
    logger.info(f"Step 2/4: Computing technical indicators...")
    
    # For now, we'll use basic technical indicators
    # We'll implement full FeatureEngineering in the next file
    from data_download.feature_engineering import FeatureEngineering
    
    if cache_dir is None:
        cache_dir = Path("d:/YoopRL/data/cache")
    
    fe = FeatureEngineering(
        symbol=symbol,
        enable_multi_asset=enable_multi_asset,
        multi_asset_symbols=multi_asset_symbols or ["SPY", "QQQ", "TLT", "GLD"],
        enable_sentiment=enable_sentiment_combined,  # ← Use combined flag
        cache_root=cache_dir,
        feature_config=feature_config,  # ← NEW: Pass feature selection from UI
    )
    
    # Process: Add technical indicators
    processed_df = fe.process(raw_df.copy())
    
    # Step 3: Normalize features for neural network
    logger.info(f"Step 3/4: Normalizing features...")
    normalized_df = fe.normalize(processed_df, fit=True)
    
    # Step 4: Split into train/test
    logger.info(f"Step 4/4: Splitting train/test...")
    split_idx = int(len(normalized_df) * train_test_split)
    
    if split_idx <= 0 or split_idx >= len(normalized_df):
        raise ValueError(
            f"Invalid train/test split index: {split_idx} "
            f"(dataset length: {len(normalized_df)})"
        )
    
    train_df = normalized_df.iloc[:split_idx].copy()
    test_df = normalized_df.iloc[split_idx:].copy()
    
    logger.info(f"✓ Training data prepared successfully!")
    logger.info(f"  - Raw data: {raw_df.shape}")
    logger.info(f"  - Processed data: {processed_df.shape}")
    logger.info(f"  - Train shape: {train_df.shape}")
    logger.info(f"  - Test shape: {test_df.shape}")
    logger.info(f"  - Features: {len(fe.feature_names)}")
    
    prepared = PreparedData(
        raw=raw_df,
        processed=processed_df,
        normalized=normalized_df,
        feature_names=fe.feature_names,
    )
    
    split = DataSplit(train=train_df, test=test_df)
    
    return prepared, split


def save_prepared_data(
    prepared: PreparedData,
    split: DataSplit,
    output_dir: Path,
    symbol: str,
) -> dict:
    """
    Save prepared data to timestamped folder structure
    
    Creates folder: data/training/{SYMBOL}_{TIMESTAMP}/
    Maintains only 10 most recent training datasets
    
    Args:
        prepared: PreparedData object
        split: DataSplit object
        output_dir: Base directory for training data (e.g., d:/YoopRL/data/training)
        symbol: Stock symbol (for file naming)
               
    Returns:
        Dictionary with file paths and metadata
    """
    from datetime import datetime
    import shutil
    
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_folder = output_dir / f"{symbol}_{timestamp}"
    training_folder.mkdir(parents=True, exist_ok=True)
    
    files = {}
    
    # Save raw data
    raw_path = training_folder / f"{symbol}_raw.csv"
    prepared.raw.to_csv(raw_path)
    files['raw'] = str(raw_path)
    logger.info(f"Saved raw data: {raw_path}")
    
    # Save processed data (with indicators)
    processed_path = training_folder / f"{symbol}_processed.csv"
    prepared.processed.to_csv(processed_path)
    files['processed'] = str(processed_path)
    logger.info(f"Saved processed data: {processed_path}")
    
    # Save normalized data
    normalized_path = training_folder / f"{symbol}_normalized.csv"
    prepared.normalized.to_csv(normalized_path)
    files['normalized'] = str(normalized_path)
    logger.info(f"Saved normalized data: {normalized_path}")
    
    # Save train split
    train_path = training_folder / f"{symbol}_train.csv"
    split.train.to_csv(train_path)
    files['train'] = str(train_path)
    logger.info(f"Saved train data: {train_path}")
    
    # Save test split
    test_path = training_folder / f"{symbol}_test.csv"
    split.test.to_csv(test_path)
    files['test'] = str(test_path)
    logger.info(f"Saved test data: {test_path}")
    
    # Save feature names
    features_path = training_folder / f"{symbol}_features.txt"
    with open(features_path, 'w') as f:
        f.write('\n'.join(prepared.feature_names))
    files['features'] = str(features_path)
    logger.info(f"Saved feature names: {features_path}")
    
    # Save metadata
    metadata = {
        'symbol': symbol,
        'timestamp': timestamp,
        'date_range': f"{prepared.raw.index[0]} to {prepared.raw.index[-1]}",
        'total_rows': len(prepared.raw),
        'train_rows': len(split.train),
        'test_rows': len(split.test),
        'feature_count': len(prepared.feature_names),
        'features': prepared.feature_names
    }
    
    import json
    metadata_path = training_folder / f"{symbol}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    files['metadata'] = str(metadata_path)
    logger.info(f"Saved metadata: {metadata_path}")
    
    # Cleanup old training folders (keep only 10 most recent)
    cleanup_old_training_folders(output_dir, symbol, keep_count=10)
    
    files['training_folder'] = str(training_folder)
    return files


def cleanup_old_training_folders(base_dir: Path, symbol: str, keep_count: int = 10):
    """
    Remove old training folders, keeping only the most recent ones
    
    Args:
        base_dir: Base training directory
        symbol: Stock symbol to filter folders
        keep_count: Number of recent folders to keep
    """
    import shutil
    
    # Find all folders for this symbol
    pattern = f"{symbol}_*"
    folders = sorted(base_dir.glob(pattern), key=lambda p: p.name, reverse=True)
    
    # Remove old folders beyond keep_count
    for old_folder in folders[keep_count:]:
        try:
            shutil.rmtree(old_folder)
            logger.info(f"Removed old training folder: {old_folder.name}")
        except Exception as e:
            logger.warning(f"Failed to remove {old_folder}: {e}")
