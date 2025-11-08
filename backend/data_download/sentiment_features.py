"""
Sentiment Feature Aggregator for market data enhancement.
Downloads and analyzes sentiment from multiple sources: Yahoo Finance news, Reddit, StockTwits.
Uses caching with TTL to minimize redundant API calls.

⚠️ BACKUP OPTION ⚠️
This module provides FREE social/retail sentiment data and is kept as a fallback option.

CURRENT SYSTEM:
- feature_engineering.py uses EnterpriseSentimentAnalyzer (sentiment_service.py)
- Professional APIs: Alpha Vantage, Finnhub, NewsAPI
- Higher quality, institutional-grade sentiment data

TO SWITCH TO THIS MODULE (Social Sentiment):
1. Open backend/data_download/feature_engineering.py
2. In _add_sentiment_features() method:
   - Replace: from .sentiment_service import EnterpriseSentimentAnalyzer
   - With: from .sentiment_features import SentimentFeatureAggregator
3. Replace: analyzer = EnterpriseSentimentAnalyzer()
   - With: aggregator = SentimentFeatureAggregator(symbol=self.symbol)
4. Replace: sentiment_data = asyncio.run(analyzer.aggregate(self.symbol))
   - With: sentiment_data = aggregator.aggregate()

TRADE-OFFS:
- ✅ Free (no API keys required)
- ✅ Social sentiment from retail investors
- ❌ Noisier data
- ❌ Web scraping can be fragile
- ❌ No professional news sources
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from textblob import TextBlob

logger = logging.getLogger(__name__)


class SentimentFeatureAggregator:
    """
    Aggregates sentiment data from multiple sources for a given stock symbol.
    
    ⚠️ BACKUP OPTION - Currently not used in feature_engineering.py
    
    Features:
    - Yahoo Finance news sentiment
    - Reddit r/wallstreetbets sentiment
    - StockTwits sentiment
    - Google Trends interest (basic)
    
    Implements caching with 6-hour TTL to avoid redundant API calls.
    """
    
    CACHE_TTL_HOURS = 6
    
    def __init__(self, symbol: str, cache_dir: Optional[str] = None):
        """
        Initialize sentiment aggregator for a stock symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            cache_dir: Directory for sentiment cache (default: data/cache/sentiment)
        """
        self.symbol = symbol.upper()
        
        # Set cache directory
        if cache_dir is None:
            cache_dir = Path("data/cache/sentiment")
        else:
            cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_path = cache_dir / f"{self.symbol}.json"
        self._cache: Optional[Dict[str, float]] = None
        self._session = requests.Session()
        
    def _load_cache(self) -> Optional[Dict[str, float]]:
        """
        Load cached sentiment data if it exists and is not expired.
        
        Returns:
            Cached sentiment dict or None if cache is missing/expired
        """
        if not self.cache_path.exists():
            return None
            
        try:
            with open(self.cache_path, "r") as f:
                data = json.load(f)
                
            # Check cache age
            cached_time = datetime.fromisoformat(data.get("timestamp", ""))
            age = datetime.now() - cached_time
            
            if age < timedelta(hours=self.CACHE_TTL_HOURS):
                logger.info(f"Using cached sentiment for {self.symbol} (age: {age})")
                return data.get("features")
            else:
                logger.info(f"Sentiment cache expired for {self.symbol} (age: {age})")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load sentiment cache: {e}")
            return None
            
    def _store_cache(self, features: Dict[str, float]) -> None:
        """
        Store sentiment features to cache file.
        
        Args:
            features: Sentiment feature dictionary
        """
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "features": features
            }
            
            with open(self.cache_path, "w") as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Cached sentiment for {self.symbol}")
            
        except Exception as e:
            logger.warning(f"Failed to store sentiment cache: {e}")
            
    def _analyze_text(self, text: str) -> Optional[float]:
        """
        Analyze text sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment polarity [-1.0, 1.0] or None if analysis fails
        """
        if not text or not isinstance(text, str):
            return None
            
        try:
            blob = TextBlob(text)
            return float(blob.sentiment.polarity)
        except Exception as e:
            logger.debug(f"Failed to analyze text: {e}")
            return None
            
    def _fetch_yahoo_finance(self) -> List[float]:
        """
        Fetch sentiment from Yahoo Finance news articles.
        
        Returns:
            List of sentiment scores
        """
        url = f"https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": self.symbol, "quotesCount": 0, "newsCount": 10}
        
        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            news = data.get("news", [])
            
            sentiments = []
            for article in news:
                title = article.get("title", "")
                score = self._analyze_text(title)
                if score is not None:
                    sentiments.append(score)
                    
            logger.info(f"Yahoo Finance: {len(sentiments)} sentiment scores for {self.symbol}")
            return sentiments
            
        except Exception as e:
            logger.warning(f"Failed to fetch Yahoo Finance sentiment: {e}")
            return []
            
    def _fetch_reddit(self) -> List[float]:
        """
        Fetch sentiment from Reddit r/wallstreetbets.
        
        Returns:
            List of sentiment scores
        """
        url = "https://www.reddit.com/r/wallstreetbets/search.json"
        params = {
            "q": self.symbol,
            "restrict_sr": "on",
            "sort": "new",
            "limit": 25
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        
        try:
            response = self._session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            posts = data.get("data", {}).get("children", [])
            
            sentiments = []
            for post in posts:
                post_data = post.get("data", {})
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")
                text = f"{title} {selftext}"
                
                score = self._analyze_text(text)
                if score is not None:
                    sentiments.append(score)
                    
            logger.info(f"Reddit: {len(sentiments)} sentiment scores for {self.symbol}")
            return sentiments
            
        except Exception as e:
            logger.warning(f"Failed to fetch Reddit sentiment: {e}")
            return []
            
    def _fetch_stocktwits(self) -> List[float]:
        """
        Fetch sentiment from StockTwits API.
        
        Returns:
            List of sentiment scores
        """
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{self.symbol}.json"
        
        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            messages = data.get("messages", [])
            
            sentiments = []
            for message in messages:
                text = message.get("body", "")
                score = self._analyze_text(text)
                if score is not None:
                    sentiments.append(score)
                    
            logger.info(f"StockTwits: {len(sentiments)} sentiment scores for {self.symbol}")
            return sentiments
            
        except Exception as e:
            logger.warning(f"Failed to fetch StockTwits sentiment: {e}")
            return []
            
    def _fetch_google_interest(self) -> float:
        """
        Fetch basic Google Trends interest score (simplified).
        
        Returns:
            Interest score (0.0-0.1)
        """
        url = f"https://www.google.com/search?q={self.symbol}+stock"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        try:
            response = self._session.get(url, headers=headers, timeout=10)
            return 0.1 if response.status_code == 200 else 0.0
        except Exception:
            return 0.0
            
    def aggregate(self) -> Dict[str, float]:
        """
        Aggregate sentiment from all sources.
        
        Returns:
            Dictionary with sentiment features:
            - sentiment_mean: Average sentiment [-1.0, 1.0]
            - sentiment_std: Sentiment standard deviation
            - sentiment_positive_ratio: Ratio of positive sentiments
            - sentiment_negative_ratio: Ratio of negative sentiments
            - sentiment_neutral_ratio: Ratio of neutral sentiments
            - sentiment_source_count: Number of sentiment sources
            - google_trends: Google Trends interest score
        """
        # Return cached result if available
        if self._cache is not None:
            return self._cache
            
        # Try loading from cache file
        cached = self._load_cache()
        if cached is not None:
            self._cache = cached
            return cached
            
        # Fetch sentiment from all sources
        sources = [
            self._fetch_yahoo_finance,
            self._fetch_reddit,
            self._fetch_stocktwits,
        ]
        
        all_scores: List[float] = []
        for source in sources:
            all_scores.extend(source())
            
        google_trends = self._fetch_google_interest()
        
        # Handle case with no sentiment data
        if not all_scores:
            self._cache = {
                "sentiment_mean": 0.0,
                "sentiment_std": 0.0,
                "sentiment_positive_ratio": 0.5,
                "sentiment_negative_ratio": 0.5,
                "sentiment_neutral_ratio": 0.0,
                "sentiment_source_count": 0.0,
                "google_trends": float(google_trends),
            }
            return self._cache
            
        # Calculate sentiment statistics
        scores = np.array(all_scores, dtype=float)
        count = float(len(scores))
        
        self._cache = {
            "sentiment_mean": float(np.mean(scores)),
            "sentiment_std": float(np.std(scores)),
            "sentiment_positive_ratio": float(np.sum(scores > 0) / count),
            "sentiment_negative_ratio": float(np.sum(scores < 0) / count),
            "sentiment_neutral_ratio": float(np.sum(scores == 0) / count),
            "sentiment_source_count": count,
            "google_trends": float(google_trends),
        }
        
        # Store to cache
        self._store_cache(self._cache)
        
        return self._cache
