"""
Multi-asset data loading and feature preparation for portfolio environments.
Downloads multiple asset histories using SQL-backed incremental updates.
"""

from typing import Dict, List, Tuple
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_histories(symbols: List[str], period: str = "5y", force_redownload: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Download historical OHLCV data for multiple symbols with SQL-backed incremental updates
    
    Uses the main loader.download_history() which implements:
    - SQL database storage
    - Incremental updates (download only missing days)
    - Automatic caching and deduplication
    
    Args:
        symbols: List of stock symbols (e.g., ["SPY", "QQQ", "TLT", "GLD"])
        period: Time period for initial download (e.g., "5y", "max")
        force_redownload: Force full redownload instead of incremental
        
    Returns:
        Dictionary mapping symbol to DataFrame with OHLCV data
        
    Raises:
        ValueError: If no symbols provided
        RuntimeError: If no data returned for any symbol
    """
    from .loader import download_history
    
    data_by_symbol: Dict[str, pd.DataFrame] = {}
    
    if not symbols:
        raise ValueError("No symbols provided for multi-asset download")
    
    for sym in symbols:
        logger.info(f"Downloading {sym} history (SQL-backed incremental)...")
        
        try:
            df = download_history(
                symbol=sym,
                period=period,
                progress=False,
                force_redownload=force_redownload
            )
            
            if df.empty:
                raise RuntimeError(f"No data returned for {sym}")
            
            data_by_symbol[sym] = df
            logger.info(f"✓ {sym}: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
            
        except Exception as e:
            logger.error(f"Failed to download {sym}: {e}")
            raise
    
    logger.info(f"Multi-asset download complete: {len(data_by_symbol)} symbols")
    return data_by_symbol


def prepare_multi_asset_features(
    histories: Dict[str, pd.DataFrame],
    enable_sentiment: bool = False,
    cache_root: Path = None,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Prepare consistent feature set across symbols using a single fitted engineer.
    
    Strategy:
    - Fit a FeatureEngineering instance on the first symbol's history
    - Apply the same engineer (transform) to each symbol for identical feature columns
    - Disable multi-asset cross-features here to keep per-asset features clean
    
    Args:
        histories: Dictionary mapping symbol to raw OHLCV DataFrame
        enable_sentiment: Whether to include sentiment features
        cache_root: Root directory for caching
        
    Returns:
        Tuple of (prepared_data, feature_names)
        - prepared_data: Dict mapping symbol to normalized DataFrame
        - feature_names: List of feature column names
        
    Raises:
        ValueError: If histories is empty or missing expected columns
    """
    from .feature_engineering import FeatureEngineering
    
    if not histories:
        raise ValueError("Empty histories for multi-asset preparation")
    
    # Sort symbols for reproducibility
    symbols = sorted(histories.keys())
    base_symbol = symbols[0]
    
    logger.info(f"Preparing multi-asset features for {len(symbols)} symbols...")
    logger.info(f"Base symbol for fitting: {base_symbol}")
    
    # Create feature engineer with cross-asset disabled
    if cache_root is None:
        cache_root = Path("data/cache")
    
    fe = FeatureEngineering(
        symbol=base_symbol,
        enable_multi_asset=False,  # Disable to avoid data leakage across assets
        enable_sentiment=enable_sentiment,
        cache_root=cache_root,
    )
    
    # Fit on the base symbol
    processed_base = fe.process(histories[base_symbol].copy())
    normalized_base = fe.normalize(processed_base, fit=True)
    feature_names = fe.feature_names.copy()
    
    logger.info(f"Fitted on {base_symbol}: {len(feature_names)} features")
    
    prepared: Dict[str, pd.DataFrame] = {base_symbol: normalized_base}
    
    # Transform remaining symbols
    for sym in symbols[1:]:
        logger.info(f"Transforming {sym}...")
        
        proc = fe.process(histories[sym].copy())
        norm = fe.normalize(proc, fit=False)
        
        # Ensure required columns exist
        missing = [c for c in feature_names if c not in norm.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns for {sym}: {missing[:5]}...")
        
        prepared[sym] = norm
        logger.info(f"Completed {sym}: {len(norm)} rows")
    
    logger.info(f"Multi-asset preparation complete: {len(prepared)} symbols")
    
    return prepared, feature_names
