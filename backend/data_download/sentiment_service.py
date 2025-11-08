"""
Enterprise Sentiment Service for professional-grade market sentiment analysis.
Aggregates sentiment from paid API sources: Alpha Vantage NEWS_SENTIMENT, Finnhub news-sentiment, NewsAPI.org

✅ ACTIVE MODULE - Currently used by feature_engineering.py

FEATURES:
- Professional news sources (institutional quality)
- Real-time sentiment scores
- Relevance scoring and buzz metrics
- NewsAPI.org integration with TextBlob analysis

API REQUIREMENTS (in .env):
- ALPHA_VANTAGE_KEY: Alpha Vantage API key (500 requests/day free tier)
- FINNHUB_KEY: Finnhub API key (60 requests/minute free tier)
- NEWS_API_KEY: NewsAPI key (100 requests/day free tier)

BACKUP OPTION:
If API limits are reached or keys unavailable, switch to SentimentFeatureAggregator:
- See sentiment_features.py for free social sentiment (Yahoo, Reddit, StockTwits)
- Instructions in sentiment_features.py header
"""

import asyncio
from typing import Dict, Optional
import aiohttp
import logging
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EnterpriseSentimentAnalyzer:
    """
    Professional sentiment analyzer using paid API sources.
    
    ✅ ACTIVE - Used by feature_engineering.py for training data
    
    Supported APIs:
    - Alpha Vantage NEWS_SENTIMENT: News sentiment with relevance scores
    - Finnhub news-sentiment: Company news sentiment with buzz metrics
    - NewsAPI.org: Article search with TextBlob sentiment analysis
    
    Returns 8 sentiment features:
    1. sentiment: Average sentiment score
    2. relevance_score: Alpha Vantage news relevance
    3. article_count: Finnhub article count
    4. buzz: Finnhub buzz score
    5. articles_in_last_week: Finnhub weekly article count
    6. newsapi_sentiment: NewsAPI TextBlob sentiment mean
    7. newsapi_sentiment_std: NewsAPI sentiment standard deviation
    8. newsapi_articles: NewsAPI article count
    """
    
    def __init__(
        self, 
        alpha_vantage_key: Optional[str] = None, 
        finnhub_key: Optional[str] = None, 
        news_api_key: Optional[str] = None
    ):
        """
        Initialize sentiment analyzer with API keys.
        
        Keys are loaded from environment variables if not provided:
        - ALPHA_VANTAGE_KEY
        - FINNHUB_KEY
        - NEWS_API_KEY
        
        Args:
            alpha_vantage_key: Alpha Vantage API key (default: from env)
            finnhub_key: Finnhub API key (default: from env)
            news_api_key: NewsAPI key (default: rom env)
        """
        import os
        self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_KEY", "")
        self.finnhub_key = finnhub_key or os.getenv("FINNHUB_KEY", "")
        self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY", "")
    
    async def get_alpha_vantage_sentiment(self, symbol: str) -> Dict[str, float]:
        """
        Fetch sentiment from Alpha Vantage NEWS_SENTIMENT API.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with sentiment, relevance, and article_count
        """
        if not self.alpha_vantage_key:
            return {"sentiment": 0.0, "relevance": 0.0}
        
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self.alpha_vantage_key,
            "limit": 50,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    data = await response.json()
            
            feed = data.get("feed", [])
            sentiments: list[float] = []
            relevances: list[float] = []
            
            for article in feed:
                for record in article.get("ticker_sentiment", []):
                    if record.get("ticker") == symbol:
                        sentiments.append(float(record.get("ticker_sentiment_score", 0.0)))
                        relevances.append(float(record.get("relevance_score", 0.0)))
            
            return {
                "sentiment": float(sum(sentiments) / len(sentiments)) if sentiments else 0.0,
                "relevance": float(sum(relevances) / len(relevances)) if relevances else 0.0,
                "article_count": len(sentiments),
            }
        
        except Exception as exc:
            logger.warning(f"Alpha Vantage sentiment failed: {exc}")
            return {"sentiment": 0.0, "relevance": 0.0}
    
    async def get_finnhub_sentiment(self, symbol: str) -> Dict[str, float]:
        """
        Fetch sentiment from Finnhub news-sentiment API.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with sentiment, buzz, and articles_in_last_week
        """
        if not self.finnhub_key:
            return {"sentiment": 0.0, "buzz": 0.0}
        
        url = "https://finnhub.io/api/v1/news-sentiment"
        params = {"symbol": symbol, "token": self.finnhub_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    data = await response.json()
            
            buzz = data.get("buzz", {})
            sentiment = data.get("sentiment", {})
            
            return {
                "sentiment": float(sentiment.get("score", 0.0)),
                "buzz": float(buzz.get("buzz", 0.0)),
                "articles_in_last_week": float(buzz.get("articlesInLastWeek", 0)),
            }
        
        except Exception as exc:
            logger.warning(f"Finnhub sentiment failed: {exc}")
            return {"sentiment": 0.0, "buzz": 0.0}
    
    async def get_newsapi_sentiment(self, symbol: str) -> Dict[str, float]:
        """
        Fetch news articles from NewsAPI and compute sentiment using TextBlob.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with sentiment_mean, sentiment_std, and article_count
        """
        if not self.news_api_key:
            logger.debug("NewsAPI key not configured, skipping NewsAPI sentiment")
            return {"newsapi_sentiment": 0.0, "newsapi_articles": 0}
        
        from datetime import datetime, timedelta
        from textblob import TextBlob
        
        # Get news from last 7 days
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "from": from_date,
            "to": to_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 50,
            "apiKey": self.news_api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    data = await response.json()
            
            if data.get("status") != "ok":
                logger.warning(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return {"newsapi_sentiment": 0.0, "newsapi_articles": 0}
            
            articles = data.get("articles", [])
            sentiments = []
            
            for article in articles:
                # Analyze sentiment of title + description
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    blob = TextBlob(text)
                    sentiments.append(blob.sentiment.polarity)
            
            if sentiments:
                import numpy as np
                return {
                    "newsapi_sentiment": float(np.mean(sentiments)),
                    "newsapi_sentiment_std": float(np.std(sentiments)),
                    "newsapi_articles": len(sentiments)
                }
            else:
                return {"newsapi_sentiment": 0.0, "newsapi_articles": 0}
        
        except Exception as exc:
            logger.warning(f"NewsAPI sentiment failed: {exc}")
            return {"newsapi_sentiment": 0.0, "newsapi_articles": 0}
    
    async def aggregate(self, symbol: str) -> Dict[str, float]:
        """
        Aggregate sentiment from all available sources.
        
        Includes:
        - Alpha Vantage NEWS_SENTIMENT (requires ALPHA_VANTAGE_KEY)
        - Finnhub news-sentiment (requires FINNHUB_KEY)
        - NewsAPI with TextBlob analysis (requires NEWS_API_KEY)
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with combined sentiment metrics from all sources
        """
        tasks = [
            self.get_alpha_vantage_sentiment(symbol), 
            self.get_finnhub_sentiment(symbol),
            self.get_newsapi_sentiment(symbol)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Merge all results
        merged: Dict[str, float] = {}
        for result in results:
            merged.update(result)
        
        logger.info(f"Aggregated sentiment for {symbol}: {len(merged)} metrics from {len(results)} sources")
        return merged
