"""
Configuration for data download and feature engineering
Contains all settings for downloading market data and preparing features
"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# טוען משתני סביבה מקובץ .env
load_dotenv()


@dataclass
class DownloadConfig:
    """
    Configuration for market data download
    """
    # Symbol to download (e.g., "IWM", "SPY", "QQQ")
    symbol: str = "IWM"
    
    # Time period for historical data (e.g., "5y", "2y", "1y", "6mo")
    data_period: str = "5y"
    
    # Train/test split ratio (0.8 = 80% train, 20% test)
    train_test_split: float = 0.8
    
    # Cache directory for downloaded data
    cache_dir: Path = field(default_factory=lambda: Path("d:/YoopRL/data/cache"))


@dataclass
class FeatureConfig:
    """
    Configuration for feature engineering
    """
    
    # === Multi-Asset Features ===
    # Enable cross-asset features (correlation with SPY, QQQ, etc.)
    enable_multi_asset: bool = True
    
    # List of symbols for cross-asset features (SPY=S&P500, QQQ=Nasdaq, TLT=Bonds, GLD=Gold)
    multi_asset_symbols: List[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "TLT", "GLD"]
    )
    
    # === Sentiment Features ===
    # Enable sentiment analysis from news/social media
    enable_sentiment: bool = True
    
    # === Technical Indicator Features ===
    # Enable regime detection (bull/bear market)
    enable_regime_features: bool = True
    
    # Enable fractal features (multi-timeframe patterns)
    enable_fractal_features: bool = True
    
    # Enable turbulence index (market stress indicator)
    enable_turbulence: bool = True
    turbulence_window: int = 50  # Rolling window for calculation
    
    # === Benchmark Comparison ===
    # Compare to benchmark (usually SPY)
    enable_benchmark_context: bool = True
    benchmark_symbol: str = "SPY"
    benchmark_windows: List[int] = field(default_factory=lambda: [20, 60])
    
    # === Feature Selection ===
    # Remove highly correlated features (reduces overfitting)
    enable_correlation_filter: bool = True
    correlation_threshold: float = 0.9  # Remove if correlation > 0.9
    
    # === Dimensionality Reduction ===
    # Use PCA to reduce feature count (optional)
    enable_pca: bool = False
    pca_components: int = 32  # Number of components to keep
    
    
@dataclass
class APIKeysConfig:
    """
    API keys for sentiment data sources
    
    Keys should be set in .env file:
    - ALPHA_VANTAGE_KEY
    - FINNHUB_KEY
    - NEWS_API_KEY (optional)
    """
    # Alpha Vantage - News sentiment
    # Get your key: https://www.alphavantage.co/support/#api-key
    # 500 requests/day, 5 requests/minute (free tier)
    alpha_vantage_key: str = field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY", "")
    )
    
    # Finnhub - Company news & sentiment
    # Get your key: https://finnhub.io/register
    # 60 requests/minute (free tier)
    finnhub_key: str = field(
        default_factory=lambda: os.getenv("FINNHUB_KEY", "")
    )
    
    # NewsAPI - General news (optional, not implemented yet)
    # 100 requests/day (free tier)
    news_api_key: str = field(
        default_factory=lambda: os.getenv("NEWS_API_KEY", "")
    )


@dataclass
class TrainingDataConfig:
    """
    Complete configuration for training data preparation
    """
    download: DownloadConfig = field(default_factory=DownloadConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)
    
    # Output directory for prepared data
    output_dir: Path = field(default_factory=lambda: Path("d:/YoopRL/data/training"))
    
    def __post_init__(self):
        """Create necessary directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.download.cache_dir.mkdir(parents=True, exist_ok=True)


def create_default_config(symbol: str = "IWM") -> TrainingDataConfig:
    """
    Create default configuration for a given symbol
    
    Args:
        symbol: Stock symbol (e.g., "IWM", "SPY")
        
    Returns:
        TrainingDataConfig with default settings
    """
    config = TrainingDataConfig()
    config.download.symbol = symbol
    return config
