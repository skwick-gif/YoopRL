"""
Feature Engineering - Technical indicators and sentiment features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from typing import List, Optional
import logging

from .sentiment_features import SentimentFeatureAggregator

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Feature engineering with technical indicators and optional sentiment features
    """
    
    def __init__(
        self,
        symbol: str = "IWM",
        enable_multi_asset: bool = False,
        multi_asset_symbols: Optional[List[str]] = None,
        enable_sentiment: bool = False,
        cache_root: Optional[Path] = None,
    ):
        self.symbol = symbol
        self.enable_multi_asset = enable_multi_asset
        self.multi_asset_symbols = multi_asset_symbols or []
        self.enable_sentiment = enable_sentiment
        self.cache_root = cache_root or Path("data/cache")
        self.scaler = RobustScaler()
        self.feature_names = []
        self._sentiment_aggregator: Optional[SentimentFeatureAggregator] = None
        
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators and optional sentiment features to dataframe
        
        Args:
            df: Raw OHLCV dataframe
            
        Returns:
            Dataframe with added features
        """
        logger.info("Adding technical indicators...")
        
        # Simple returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Volatility
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['atr'] = self._calculate_atr(df)
        
        # Volume features
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # Price position
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Add sentiment features if enabled
        if self.enable_sentiment:
            df = self._add_sentiment_features(df)
        
        # Drop rows with NaN (warmup period)
        df = df.dropna()
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        logger.info(f"Added {len(self.feature_names)} features")
        return df
        
    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment features to dataframe with SQL database caching
        
        Strategy:
        1. Try to load sentiment from SQL database (fast)
        2. If not in SQL, compute using EnterpriseSentimentAnalyzer:
           - Professional APIs: Alpha Vantage NEWS_SENTIMENT, Finnhub news-sentiment, NewsAPI.org
           - High-quality institutional data
           - Requires paid API keys but provides reliable sentiment
        3. Save computed sentiment to SQL for future reuse
        4. Broadcast sentiment values to all rows in dataframe
        
        BACKUP OPTION:
        If professional sentiment fails, SentimentFeatureAggregator (sentiment_features.py) 
        is available as fallback:
        - Free social sentiment from Yahoo Finance, Reddit r/wallstreetbets, StockTwits
        - To use: Replace EnterpriseSentimentAnalyzer with SentimentFeatureAggregator
        - Note: More noisy but free, good for testing or when API limits reached
        
        Args:
            df: Dataframe with OHLCV data (index = datetime)
            
        Returns:
            Dataframe with sentiment features added as columns
        """
        logger.info("Adding sentiment features (Professional APIs: Alpha Vantage, Finnhub, NewsAPI)...")
        
        try:
            from ..database.db_manager import DatabaseManager
            from .sentiment_service import EnterpriseSentimentAnalyzer
            from datetime import datetime
            import asyncio
            
            db = DatabaseManager()
            
            # Get the most recent date in the dataframe
            latest_date = df.index[-1].strftime('%Y-%m-%d')
            
            # Try to load sentiment from SQL first
            sentiment_row = db.get_sentiment_data(self.symbol, latest_date)
            
            if sentiment_row is None:
                logger.info(f"No SQL sentiment data for {self.symbol} on {latest_date}, computing using professional APIs...")
                
                # Initialize Enterprise Sentiment Analyzer with API keys from .env
                analyzer = EnterpriseSentimentAnalyzer()
                
                # Run async sentiment aggregation
                sentiment_data = asyncio.run(analyzer.aggregate(self.symbol))
                
                logger.info(f"Professional sentiment computed for {self.symbol}: {sentiment_data}")
                
                # Save to SQL for future reuse
                db.save_sentiment_data(
                    symbol=self.symbol,
                    date=latest_date,
                    sentiment_data=sentiment_data
                )
                logger.info(f"Saved professional sentiment to SQL for {self.symbol} on {latest_date}")
            else:
                # Load sentiment from SQL (cached)
                logger.info(f"Loading sentiment from SQL for {self.symbol}...")
                
                # Parse sentiment data from SQL row
                # Enterprise sentiment has 8 metrics
                sentiment_data = sentiment_row  # db.get_sentiment_data already returns dict
                logger.info(f"Loaded cached professional sentiment for {self.symbol}")
            
            # Broadcast sentiment features to all rows
            for key, value in sentiment_data.items():
                df[key] = value
                
            logger.info(f"Added {len(sentiment_data)} professional sentiment features")
            
        except Exception as e:
            logger.warning(f"Failed to add professional sentiment features: {e}")
            logger.info("Continuing without sentiment features")
            logger.info("FALLBACK: Consider using SentimentFeatureAggregator (sentiment_features.py) for free social sentiment")
            import traceback
            traceback.print_exc()  # DEBUG: Print full stack trace
        
        return df
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
        
    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize features using RobustScaler
        
        Args:
            df: Dataframe with features
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized dataframe
        """
        logger.info("Normalizing features...")
        
        # Get only feature columns
        feature_cols = self.feature_names
        
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
            
        return df
