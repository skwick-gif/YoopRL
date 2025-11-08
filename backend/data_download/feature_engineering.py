"""
Feature Engineering - Technical indicators and sentiment features
"""

import sys
from pathlib import Path

# Add backend directory to path for absolute imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import List, Optional
import logging

# Lazy import to avoid NLTK blocking
# from data_download.sentiment_features import SentimentFeatureAggregator

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Feature engineering with technical indicators and optional sentiment features
    
    NEW: Selective feature computation - only calculates what was requested
    """
    
    def __init__(
        self,
        symbol: str = "IWM",
        enable_multi_asset: bool = False,
        multi_asset_symbols: Optional[List[str]] = None,
        enable_sentiment: bool = False,
        cache_root: Optional[Path] = None,
        feature_config: Optional[dict] = None,  # NEW: Feature selection from UI
    ):
        self.symbol = symbol
        self.enable_multi_asset = enable_multi_asset
        self.multi_asset_symbols = multi_asset_symbols or []
        self.enable_sentiment = enable_sentiment
        self.cache_root = cache_root or Path("data/cache")
        self.scaler = RobustScaler()
        self.feature_names = []
        self._sentiment_aggregator = None  # Lazy init when needed
        
        # Feature configuration from UI (what to calculate)
        self.feature_config = feature_config or {}
        
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators and optional sentiment features to dataframe
        
        SELECTIVE COMPUTATION: Only calculates features that are enabled in feature_config
        
        Args:
            df: Raw OHLCV dataframe
            
        Returns:
            Dataframe with added features (only requested ones)
        """
        logger.info("Adding technical indicators (selective based on UI configuration)...")
        
        # ===== ALWAYS ADD: Returns (mandatory for RL environment) =====
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # ===== CONDITIONAL: Technical Indicators (only if requested) =====
        
        # RSI
        if self._is_enabled('rsi'):
            period = self.feature_config.get('rsi', {}).get('period', 14)
            logger.info(f"  - Adding RSI (period={period})")
            df['rsi'] = self._calculate_rsi(df['Close'], period)
        
        # MACD
        if self._is_enabled('macd'):
            params = self.feature_config.get('macd', {}).get('params', '12,26,9')
            fast, slow, signal = [int(x) for x in params.split(',')]
            logger.info(f"  - Adding MACD ({fast},{slow},{signal})")
            macd_line, signal_line, histogram = self._calculate_macd(df['Close'], fast, slow, signal)
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_histogram'] = histogram
        
        # EMA
        if self._is_enabled('ema'):
            periods = self.feature_config.get('ema', {}).get('periods', '10,50')
            period_list = [int(x) for x in periods.split(',')]
            logger.info(f"  - Adding EMA (periods={period_list})")
            for period in period_list:
                df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        
        # SMA (always add basic ones for reference, they're cheap)
        # These are used by other indicators too
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        
        # Bollinger Bands
        if self._is_enabled('bollinger'):
            params = self.feature_config.get('bollinger', {}).get('params', '20,2')
            period, std_dev = [int(x) for x in params.split(',')]
            logger.info(f"  - Adding Bollinger Bands (period={period}, std={std_dev})")
            sma = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            df['bb_upper'] = sma + (std_dev * std)
            df['bb_lower'] = sma - (std_dev * std)
            df['bb_middle'] = sma
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Stochastic
        if self._is_enabled('stochastic'):
            params = self.feature_config.get('stochastic', {}).get('params', '14,3')
            k_period, d_period = [int(x) for x in params.split(',')]
            logger.info(f"  - Adding Stochastic (K={k_period}, D={d_period})")
            df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df, k_period, d_period)
        
        # VIX (if requested and available)
        if self._is_enabled('vix'):
            logger.info("  - VIX requested (not implemented yet - would need separate data source)")
            # TODO: Download VIX data from Yahoo Finance (^VIX) and merge
        
        # ===== VOLATILITY & VOLUME (always useful, cheap to compute) =====
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['atr'] = self._calculate_atr(df)
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        # ===== PRICE POSITION (always useful) =====
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # ===== ALTERNATIVE DATA =====
        
        # Add multi-asset correlation features if enabled
        if self.enable_multi_asset:
            logger.info(f"Adding multi-asset features for {self.multi_asset_symbols}...")
            df = self._add_multi_asset_features(df)
        
        # Add sentiment features if enabled
        if self.enable_sentiment:
            df = self._add_sentiment_features(df)
        
        # ===== FUNDAMENTAL FEATURES (if enabled) =====
        if self._is_enabled('fundamental'):
            logger.info("Adding fundamental features...")
            df = self._add_fundamental_features(df)
        
        # ===== MARKET EVENTS FEATURES (if enabled) =====
        if self._is_enabled('market_events'):
            logger.info("Adding market events features...")
            df = self._add_market_events_features(df)
        
        # ===== MACRO INDICATORS (if enabled) =====
        if self._is_enabled('macro'):
            logger.info("Adding macro indicators...")
            df = self._add_macro_features(df)
        
        # Drop rows with NaN ONLY for critical columns (price, volume, returns)
        # Allow NaN in macro/fundamental columns (they get filled with forward/backward fill)
        critical_cols = ['Close', 'returns']
        df = df.dropna(subset=critical_cols)
        
        # Fill remaining NaN in other columns with 0 (better than dropping all data)
        df = df.fillna(0)
        
        # Store feature names (exclude OHLCV columns)
        self.feature_names = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        logger.info(f"✅ Added {len(self.feature_names)} features: {', '.join(self.feature_names[:10])}...")
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
            # Use absolute imports instead of relative
            from database.db_manager import DatabaseManager
            from data_download.sentiment_service import EnterpriseSentimentAnalyzer
            from datetime import datetime
            import asyncio
            
            db = DatabaseManager()
            
            # Get the most recent VALID date in the dataframe (skip NaT)
            valid_dates = df.index.dropna()
            if len(valid_dates) == 0:
                logger.warning("No valid dates in dataframe, skipping sentiment")
                return df
            
            latest_date = valid_dates[-1].strftime('%Y-%m-%d')
            
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
        
    def _add_multi_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add correlation features from other assets (SPY, QQQ, TLT, GLD)
        
        Args:
            df: Dataframe with primary asset OHLCV data
            
        Returns:
            Dataframe with multi-asset correlation features
        """
        from data_download.multi_asset_loader import download_histories
        
        try:
            # Download histories for correlation assets
            histories = download_histories(
                symbols=self.multi_asset_symbols,
                period="5y",
                force_redownload=False
            )
            
            # For each asset, add returns and correlation
            for symbol in self.multi_asset_symbols:
                if symbol in histories:
                    asset_df = histories[symbol][['Close']].copy()
                    asset_df.columns = [f'{symbol.lower()}_close']
                    
                    # Join with main df (inner join to keep only overlapping dates)
                    joined = df.join(asset_df, how='left')
                    
                    # Forward fill missing values using ffill()
                    joined[f'{symbol.lower()}_close'] = joined[f'{symbol.lower()}_close'].ffill()
                    
                    # Add close price ratio
                    df[f'{symbol.lower()}_close_ratio'] = joined[f'{symbol.lower()}_close'] / joined[f'{symbol.lower()}_close'].shift(1)
                    
                    # Add returns
                    asset_returns = joined[f'{symbol.lower()}_close'].pct_change()
                    df[f'{symbol.lower()}_returns'] = asset_returns
                    
                    # Add rolling correlation with main asset (only if we have 'returns' column)
                    if 'returns' in df.columns:
                        df[f'{symbol.lower()}_corr_20'] = df['returns'].rolling(20).corr(asset_returns)
            
            logger.info(f"Added multi-asset features for {len(self.multi_asset_symbols)} symbols")
            
        except Exception as e:
            logger.warning(f"Failed to add multi-asset features: {e}")
            logger.info("Continuing without multi-asset features")
            import traceback
            traceback.print_exc()
        
        return df
        
    def _is_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled in the configuration
        
        Args:
            feature_name: Name of feature (e.g., 'rsi', 'macd', 'ema')
            
        Returns:
            True if feature should be calculated
        """
        if not self.feature_config:
            # No config provided - default to False (selective mode)
            return False
        
        feature = self.feature_config.get(feature_name, {})
        
        # Handle both dict format and boolean format
        if isinstance(feature, dict):
            return feature.get('enabled', False)
        else:
            return bool(feature)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            prices: Price series (typically Close)
            period: RSI period (default 14)
            
        Returns:
            RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series (typically Close)
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        """
        Calculate Stochastic Oscillator
        
        Args:
            df: Dataframe with High, Low, Close
            k_period: %K period (default 14)
            d_period: %D period (default 3)
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()
        
        stoch_k = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k, stoch_d
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def _add_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add company fundamental features to dataframe
        
        Strategy:
        - Download fundamentals from yfinance (cached in SQL)
        - Broadcast single value to all rows (fundamentals are constant over short periods)
        - Create derived features (quality score, value score, growth score)
        
        Features added (if data available):
        - forward_pe, price_to_book, market_cap_log (log scale)
        - profit_margins, operating_margins, gross_margins
        - roe, roa, revenue_growth
        - debt_to_equity, current_ratio, quick_ratio
        - free_cashflow_log, operating_cashflow_log
        - trailing_eps, forward_eps, book_value
        - beta, short_ratio
        - Quality Score: Composite of margins + ROE + cash flow
        - Value Score: Composite of PE + PB + target vs current price
        - Growth Score: Revenue growth + EPS growth potential
        
        Args:
            df: Dataframe with OHLCV data
            
        Returns:
            Dataframe with fundamental features added
        """
        try:
            from data_download.fundamentals_loader import get_fundamentals
            
            # Download fundamentals (cached in SQL)
            fundamentals = get_fundamentals(self.symbol)
            
            if not fundamentals:
                logger.warning(f"No fundamentals available for {self.symbol}")
                return df
            
            # Add raw fundamental features
            df['forward_pe'] = fundamentals.get('forward_pe')
            df['price_to_book'] = fundamentals.get('price_to_book')
            df['market_cap_log'] = np.log(fundamentals['market_cap']) if fundamentals.get('market_cap') else None
            
            df['profit_margins'] = fundamentals.get('profit_margins')
            df['operating_margins'] = fundamentals.get('operating_margins')
            df['gross_margins'] = fundamentals.get('gross_margins')
            df['roe'] = fundamentals.get('return_on_equity')
            df['roa'] = fundamentals.get('return_on_assets')
            df['revenue_growth'] = fundamentals.get('revenue_growth')
            
            df['debt_to_equity'] = fundamentals.get('debt_to_equity')
            df['current_ratio'] = fundamentals.get('current_ratio')
            df['quick_ratio'] = fundamentals.get('quick_ratio')
            
            df['free_cashflow_log'] = np.log(abs(fundamentals['free_cashflow'])) if fundamentals.get('free_cashflow') else None
            df['operating_cashflow_log'] = np.log(abs(fundamentals['operating_cashflow'])) if fundamentals.get('operating_cashflow') else None
            
            df['trailing_eps'] = fundamentals.get('trailing_eps')
            df['forward_eps'] = fundamentals.get('forward_eps')
            df['book_value'] = fundamentals.get('book_value')
            df['beta'] = fundamentals.get('beta')
            df['short_ratio'] = fundamentals.get('short_ratio')
            
            # Derived features: Quality, Value, Growth scores
            # Quality Score: Higher margins + ROE + positive cash flow = quality company
            quality_components = []
            if fundamentals.get('profit_margins'): quality_components.append(fundamentals['profit_margins'] * 100)
            if fundamentals.get('return_on_equity'): quality_components.append(fundamentals['return_on_equity'] * 100)
            if fundamentals.get('current_ratio'): quality_components.append(min(fundamentals['current_ratio'] / 2, 1) * 20)  # Cap at 2.0
            df['quality_score'] = np.mean(quality_components) if quality_components else None
            
            # Value Score: Lower PE + PB + higher target/current = undervalued
            value_components = []
            if fundamentals.get('forward_pe') and fundamentals['forward_pe'] > 0:
                value_components.append(max(0, 100 - fundamentals['forward_pe']))  # Lower is better
            if fundamentals.get('price_to_book') and fundamentals['price_to_book'] > 0:
                value_components.append(max(0, 50 - fundamentals['price_to_book'] * 10))  # Lower is better
            df['value_score'] = np.mean(value_components) if value_components else None
            
            # Growth Score: Revenue growth + EPS growth potential
            growth_components = []
            if fundamentals.get('revenue_growth'): growth_components.append(fundamentals['revenue_growth'] * 100)
            if fundamentals.get('forward_eps') and fundamentals.get('trailing_eps') and fundamentals['trailing_eps'] != 0:
                eps_growth = (fundamentals['forward_eps'] - fundamentals['trailing_eps']) / abs(fundamentals['trailing_eps']) * 100
                growth_components.append(eps_growth)
            df['growth_score'] = np.mean(growth_components) if growth_components else None
            
            feature_count = len([c for c in df.columns if c.startswith(('forward_', 'price_to_', 'market_cap_', 'profit_', 'operating_', 'gross_', 'roe', 'roa', 'revenue_', 'debt_', 'current_', 'quick_', 'free_', 'trailing_', 'book_', 'beta', 'short_', 'quality_', 'value_', 'growth_'))])
            logger.info(f"  ✓ Added {feature_count} fundamental features")
            
        except Exception as e:
            logger.error(f"Error adding fundamental features: {e}")
        
        return df
    
    def _add_market_events_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market events features to dataframe
        
        Strategy:
        - Download events from yfinance (cached in SQL)
        - Compute forward-looking features (days to earnings)
        - Compute backward-looking features (days since dividend/split)
        - Create risk signals (high volatility expected before earnings)
        
        Features added:
        - days_to_earnings: Days until next earnings (0 if unknown)
        - earnings_in_week: 1 if earnings within 7 days, else 0
        - earnings_surprise_potential: Spread between high/low EPS estimates (uncertainty)
        - has_dividend: 1 if stock pays dividends, else 0
        - days_since_dividend: Days since last dividend
        - dividend_yield_approx: Approximate annual yield
        - has_split: 1 if stock had splits, else 0
        - days_since_split: Days since last split
        
        Agent can use this to:
        - Avoid/embrace earnings volatility
        - Factor in dividend income for position sizing
        - Understand stock split history (growth vs value)
        
        Args:
            df: Dataframe with OHLCV data
            
        Returns:
            Dataframe with event features added
        """
        try:
            from data_download.fundamentals_loader import get_market_events
            from datetime import datetime
            
            # Download events (cached in SQL)
            events = get_market_events(self.symbol)
            
            if not events:
                logger.warning(f"No market events available for {self.symbol}")
                return df
            
            today = datetime.now().date()
            
            # Earnings features
            if events.get('earnings') and events['earnings'].get('earnings_date'):
                try:
                    earnings_date = pd.to_datetime(events['earnings']['earnings_date']).date()
                    days_to_earnings = (earnings_date - today).days
                    
                    df['days_to_earnings'] = max(0, days_to_earnings)  # 0 if past
                    df['earnings_in_week'] = 1 if 0 <= days_to_earnings <= 7 else 0
                    
                    # Earnings surprise potential (uncertainty)
                    est_low = events['earnings'].get('earnings_estimate_low', 0) or 0
                    est_high = events['earnings'].get('earnings_estimate_high', 0) or 0
                    df['earnings_surprise_potential'] = abs(est_high - est_low) if est_high and est_low else 0
                    
                except Exception as e:
                    logger.debug(f"Error parsing earnings date: {e}")
                    df['days_to_earnings'] = 0
                    df['earnings_in_week'] = 0
                    df['earnings_surprise_potential'] = 0
            else:
                df['days_to_earnings'] = 0
                df['earnings_in_week'] = 0
                df['earnings_surprise_potential'] = 0
            
            # Dividend features
            if events.get('dividend') and events['dividend'].get('last_dividend_date'):
                try:
                    dividend_date = pd.to_datetime(events['dividend']['last_dividend_date']).date()
                    days_since_dividend = (today - dividend_date).days
                    
                    df['has_dividend'] = 1
                    df['days_since_dividend'] = days_since_dividend
                    
                    # Approximate annual dividend yield
                    dividend_amount = events['dividend'].get('last_dividend_amount', 0) or 0
                    current_price = df['Close'].iloc[-1]
                    # Assume quarterly dividends
                    df['dividend_yield_approx'] = (dividend_amount * 4 / current_price * 100) if current_price > 0 else 0
                    
                except Exception as e:
                    logger.debug(f"Error parsing dividend date: {e}")
                    df['has_dividend'] = 0
                    df['days_since_dividend'] = 9999
                    df['dividend_yield_approx'] = 0
            else:
                df['has_dividend'] = 0
                df['days_since_dividend'] = 9999
                df['dividend_yield_approx'] = 0
            
            # Split features
            if events.get('split') and events['split'].get('last_split_date'):
                try:
                    split_date = pd.to_datetime(events['split']['last_split_date']).date()
                    days_since_split = (today - split_date).days
                    
                    df['has_split'] = 1
                    df['days_since_split'] = days_since_split
                    
                except Exception as e:
                    logger.debug(f"Error parsing split date: {e}")
                    df['has_split'] = 0
                    df['days_since_split'] = 9999
            else:
                df['has_split'] = 0
                df['days_since_split'] = 9999
            
            logger.info(f"  ✓ Added 8 market event features")
            
        except Exception as e:
            logger.error(f"Error adding market event features: {e}")
        
        return df
    
    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add macro economic indicators to dataframe
        
        Strategy:
        - Download macro data for entire dataframe date range (cached in SQL)
        - Merge on date (macro is same for all stocks on same day)
        - Create derived features (risk regime, interest rate sensitivity)
        
        Features added (if data available):
        - vix, vix_change: Volatility index (fear gauge)
        - treasury_10y, treasury_2y, yield_curve_spread: Interest rates
        - dxy, dxy_change: Dollar strength
        - oil_price, gold_price: Commodities
        - gdp_growth, unemployment_rate, cpi, fed_funds_rate: Economic data (FRED)
        
        Derived features:
        - risk_regime: 0=Low VIX (<15), 1=Normal (15-25), 2=High VIX (>25)
        - yield_curve_inverted: 1 if 2Y > 10Y (recession warning), else 0
        - dollar_strength_regime: 0=Weak (<95), 1=Normal (95-105), 2=Strong (>105)
        
        Agent benefits:
        - Adjust position sizing: Reduce in high VIX
        - Sector rotation: Tech vs value based on yields
        - Safe haven: Gold when dollar weak
        - Recession signals: Inverted yield curve
        
        Args:
            df: Dataframe with OHLCV data
            
        Returns:
            Dataframe with macro features added
        """
        try:
            from data_download.macro_indicators import get_macro_indicators
            
            # Filter out NaT values from index FIRST
            valid_idx = df.index.notna()
            if not valid_idx.all():
                logger.warning(f"Filtering out {(~valid_idx).sum()} NaT values from index")
                df = df[valid_idx].copy()  # Create explicit copy
            
            if df.empty:
                logger.warning("DataFrame is empty after filtering NaT")
                return df
            
            # Get date range from dataframe
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
            
            # Download macro data (cached in SQL)
            macro_df = get_macro_indicators(start_date, end_date)
            
            if macro_df is None or macro_df.empty:
                logger.warning("No macro indicators available")
                return df
            
            # Prepare for merge: convert both dataframes to have numeric merge_key
            df_temp = df.copy()
            df_temp['merge_key'] = pd.to_datetime(df_temp.index).astype('int64') // 10**9  # Unix timestamp
            
            # Prepare macro_df for merge
            macro_df_reset = macro_df.reset_index()
            if 'date' in macro_df_reset.columns:
                macro_df_reset['merge_key'] = pd.to_datetime(macro_df_reset['date']).astype('int64') // 10**9
            else:
                macro_df_reset['merge_key'] = pd.to_datetime(macro_df_reset.index).astype('int64') // 10**9
            
            # Merge on numeric timestamp (forward fill macro data for missing dates)
            merged = pd.merge_asof(
                df_temp.sort_values('merge_key'),
                macro_df_reset.sort_values('merge_key'),
                on='merge_key',
                direction='backward'  # Use most recent macro data
            )
            
            # Set index back
            merged = merged.set_index(df.index)
            
            # Add macro columns to original df
            macro_cols = ['vix', 'vix_change', 'treasury_10y', 'treasury_2y', 'yield_curve_spread',
                         'dxy', 'dxy_change', 'oil_price', 'gold_price',
                         'gdp_growth', 'unemployment_rate', 'cpi', 'fed_funds_rate']
            
            for col in macro_cols:
                if col in merged.columns:
                    df[col] = merged[col].values
                    # Fill NaN values with forward fill, then backward fill (using modern syntax)
                    df[col] = df[col].ffill().bfill()
            
            # Derived features (handle NaN values)
            if 'vix' in df.columns and df['vix'].notna().any():
                # Only create risk_regime if we have VIX data
                df['risk_regime'] = pd.cut(df['vix'].fillna(20), bins=[0, 15, 25, 100], labels=[0, 1, 2]).astype(float)
            
            if 'yield_curve_spread' in df.columns and df['yield_curve_spread'].notna().any():
                df['yield_curve_inverted'] = (df['yield_curve_spread'].fillna(0) < 0).astype(int)
            
            if 'dxy' in df.columns and df['dxy'].notna().any():
                df['dollar_strength_regime'] = pd.cut(df['dxy'].fillna(100), bins=[0, 95, 105, 200], labels=[0, 1, 2]).astype(float)
            
            macro_feature_count = len([c for c in macro_cols if c in df.columns]) + 3  # +3 for derived
            logger.info(f"  ✓ Added {macro_feature_count} macro features")
            
        except Exception as e:
            logger.error(f"Error adding macro features: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return df
        
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
