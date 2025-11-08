"""
Fundamentals and Market Events Loader

This module downloads and caches:
1. Company fundamental metrics (valuation, profitability, growth, health)
2. Market events (earnings dates, dividends, stock splits)

Data Sources:
- yfinance: Free, reliable, comprehensive fundamental data
- SQL database: Caching to minimize API calls

Key Features:
- 24 fundamental metrics covering all aspects of company analysis
- Event tracking with forward-looking earnings calendar
- Smart caching: Only download if data is stale (>1 day old)
- Robust error handling with fallback to cached data
- Broadcast fundamentals to all rows in training dataframe
"""

import sys
from pathlib import Path

# Add backend directory to path for absolute imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class FundamentalsLoader:
    """
    Downloads and caches company fundamental data and market events
    
    Fundamentals Strategy:
    - Fundamentals change slowly (quarterly earnings)
    - Cache in SQL with 24-hour TTL
    - Broadcast single value to all dataframe rows (fundamentals are constant over short periods)
    
    Events Strategy:
    - Track upcoming earnings, past dividends, past splits
    - Generate forward-looking features (days_to_earnings, has_upcoming_dividend)
    - Help agent avoid/embrace event volatility
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize fundamentals loader
        
        Args:
            db_manager: Database manager instance (creates new if None)
        """
        self.db = db_manager or DatabaseManager()
        logger.info("FundamentalsLoader initialized")
    
    def download_fundamentals(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Download company fundamental metrics from yfinance with SQL caching
        
        Metrics downloaded (24 total):
        - Valuation: PE ratio, P/B, Market Cap, Enterprise Value
        - Profitability: Margins (profit, operating, gross), ROE, ROA
        - Growth: Revenue growth, Total revenue
        - Health: Debt/Equity, Current ratio, Quick ratio
        - Cash Flow: Free cash flow, Operating cash flow
        - Per Share: EPS (trailing, forward), Book value
        - Risk: Beta
        - Ownership: Shares outstanding, Institutional %, Short ratio
        - Targets: Analyst mean price target
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            force_refresh: Force download even if cached data exists
            
        Returns:
            Dictionary with fundamental metrics, or None if download fails
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Try to load from SQL cache first (unless force_refresh)
        if not force_refresh:
            cached = self._load_fundamentals_from_sql(symbol, today)
            if cached:
                logger.info(f"✓ Loaded fundamentals for {symbol} from SQL cache")
                return cached
        
        # Download fresh data from yfinance
        logger.info(f"Downloading fundamentals for {symbol} from yfinance...")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'symbol' not in info:
                logger.warning(f"No fundamental data available for {symbol}")
                return None
            
            # Extract fundamentals with proper handling of None values
            fundamentals = {
                'symbol': symbol,
                'date': today,
                
                # Valuation
                'forward_pe': info.get('forwardPE'),
                'price_to_book': info.get('priceToBook'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                
                # Profitability
                'profit_margins': info.get('profitMargins'),
                'operating_margins': info.get('operatingMargins'),
                'gross_margins': info.get('grossMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                
                # Growth
                'revenue_growth': info.get('revenueGrowth'),
                'total_revenue': info.get('totalRevenue'),
                
                # Financial Health
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                
                # Cash Flow
                'free_cashflow': info.get('freeCashflow'),
                'operating_cashflow': info.get('operatingCashflow'),
                
                # Per Share Metrics
                'trailing_eps': info.get('trailingEps'),
                'forward_eps': info.get('forwardEps'),
                'book_value': info.get('bookValue'),
                
                # Risk
                'beta': info.get('beta'),
                
                # Ownership
                'shares_outstanding': info.get('sharesOutstanding'),
                'held_percent_institutions': info.get('heldPercentInstitutions'),
                'short_ratio': info.get('shortRatio'),
                
                # Analyst Targets
                'target_mean_price': info.get('targetMeanPrice'),
            }
            
            # Count how many metrics we got
            non_null_count = sum(1 for v in fundamentals.values() if v is not None and v != symbol and v != today)
            logger.info(f"✓ Downloaded {non_null_count}/24 fundamental metrics for {symbol}")
            
            # Save to SQL
            self._save_fundamentals_to_sql(fundamentals)
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Failed to download fundamentals for {symbol}: {e}")
            # Try to return cached data as fallback
            cached = self._load_fundamentals_from_sql(symbol)
            if cached:
                logger.info(f"⚠️ Using stale cached fundamentals for {symbol}")
                return cached
            return None
    
    def download_market_events(self, symbol: str, force_refresh: bool = False) -> Dict[str, any]:
        """
        Download market events (earnings, dividends, splits) from yfinance
        
        Events tracked:
        1. Earnings Calendar: Upcoming earnings dates with estimates
        2. Dividends History: Past dividend payments
        3. Stock Splits History: Past split events
        
        Forward-looking features:
        - days_to_next_earnings: How many days until next earnings (critical for volatility)
        - upcoming_earnings_surprise: Potential EPS surprise based on estimates
        - last_dividend_days_ago: Days since last dividend (for dividend stocks)
        - last_split_days_ago: Days since last split (for high-growth stocks)
        
        Args:
            symbol: Stock symbol
            force_refresh: Force download even if cached
            
        Returns:
            Dictionary with events data and computed features
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Try SQL cache first
        if not force_refresh:
            cached = self._load_events_from_sql(symbol, today)
            if cached:
                logger.info(f"✓ Loaded market events for {symbol} from SQL cache")
                return cached
        
        logger.info(f"Downloading market events for {symbol} from yfinance...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # 1. Earnings Calendar (forward-looking)
            earnings_data = None
            try:
                calendar = ticker.calendar
                if calendar is not None and isinstance(calendar, dict):
                    earnings_dates = calendar.get('Earnings Date', [])
                    if earnings_dates:
                        # Get next earnings date
                        next_earnings = earnings_dates[0] if isinstance(earnings_dates, list) else earnings_dates
                        earnings_data = {
                            'earnings_date': str(next_earnings),
                            'earnings_estimate_low': calendar.get('Earnings Low'),
                            'earnings_estimate_high': calendar.get('Earnings High'),
                            'earnings_estimate_avg': calendar.get('Earnings Average'),
                            'revenue_estimate_low': calendar.get('Revenue Low'),
                            'revenue_estimate_high': calendar.get('Revenue High'),
                            'revenue_estimate_avg': calendar.get('Revenue Average'),
                        }
                        logger.info(f"  Next earnings: {next_earnings}")
            except Exception as e:
                logger.debug(f"No earnings calendar for {symbol}: {e}")
            
            # 2. Dividends History
            dividend_data = None
            try:
                actions = ticker.actions
                if actions is not None and not actions.empty and 'Dividends' in actions.columns:
                    dividends = actions[actions['Dividends'] > 0]
                    if not dividends.empty:
                        last_dividend = dividends.iloc[-1]
                        dividend_data = {
                            'last_dividend_date': str(last_dividend.name.date()),
                            'last_dividend_amount': float(last_dividend['Dividends']),
                        }
                        logger.info(f"  Last dividend: ${last_dividend['Dividends']:.2f} on {last_dividend.name.date()}")
            except Exception as e:
                logger.debug(f"No dividend history for {symbol}: {e}")
            
            # 3. Stock Splits History
            split_data = None
            try:
                actions = ticker.actions
                if actions is not None and not actions.empty and 'Stock Splits' in actions.columns:
                    splits = actions[actions['Stock Splits'] > 0]
                    if not splits.empty:
                        last_split = splits.iloc[-1]
                        split_data = {
                            'last_split_date': str(last_split.name.date()),
                            'last_split_ratio': f"{int(last_split['Stock Splits'])}:1",
                        }
                        logger.info(f"  Last split: {int(last_split['Stock Splits'])}:1 on {last_split.name.date()}")
            except Exception as e:
                logger.debug(f"No split history for {symbol}: {e}")
            
            # Compile events
            events = {
                'symbol': symbol,
                'date': today,
                'earnings': earnings_data,
                'dividend': dividend_data,
                'split': split_data,
            }
            
            # Save to SQL
            self._save_events_to_sql(events)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to download market events for {symbol}: {e}")
            # Fallback to cached data
            cached = self._load_events_from_sql(symbol)
            if cached:
                logger.info(f"⚠️ Using cached market events for {symbol}")
                return cached
            return {'symbol': symbol, 'date': today, 'earnings': None, 'dividend': None, 'split': None}
    
    def _load_fundamentals_from_sql(self, symbol: str, date: Optional[str] = None) -> Optional[Dict]:
        """Load fundamentals from SQL database"""
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            
            if date:
                # Try exact date match
                cursor.execute("""
                    SELECT * FROM fundamental_data 
                    WHERE symbol = ? AND date = ?
                """, (symbol, date))
            else:
                # Get most recent
                cursor.execute("""
                    SELECT * FROM fundamental_data 
                    WHERE symbol = ? 
                    ORDER BY date DESC 
                    LIMIT 1
                """, (symbol,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Error loading fundamentals from SQL: {e}")
            return None
    
    def _save_fundamentals_to_sql(self, fundamentals: Dict):
        """Save fundamentals to SQL database"""
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO fundamental_data (
                    symbol, date, forward_pe, price_to_book, market_cap, enterprise_value,
                    profit_margins, operating_margins, gross_margins, return_on_equity, return_on_assets,
                    revenue_growth, total_revenue, debt_to_equity, current_ratio, quick_ratio,
                    free_cashflow, operating_cashflow, trailing_eps, forward_eps, book_value,
                    beta, shares_outstanding, held_percent_institutions, short_ratio, target_mean_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fundamentals['symbol'], fundamentals['date'],
                fundamentals.get('forward_pe'), fundamentals.get('price_to_book'),
                fundamentals.get('market_cap'), fundamentals.get('enterprise_value'),
                fundamentals.get('profit_margins'), fundamentals.get('operating_margins'),
                fundamentals.get('gross_margins'), fundamentals.get('return_on_equity'),
                fundamentals.get('return_on_assets'), fundamentals.get('revenue_growth'),
                fundamentals.get('total_revenue'), fundamentals.get('debt_to_equity'),
                fundamentals.get('current_ratio'), fundamentals.get('quick_ratio'),
                fundamentals.get('free_cashflow'), fundamentals.get('operating_cashflow'),
                fundamentals.get('trailing_eps'), fundamentals.get('forward_eps'),
                fundamentals.get('book_value'), fundamentals.get('beta'),
                fundamentals.get('shares_outstanding'), fundamentals.get('held_percent_institutions'),
                fundamentals.get('short_ratio'), fundamentals.get('target_mean_price')
            ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Saved fundamentals to SQL for {fundamentals['symbol']}")
            
        except Exception as e:
            logger.error(f"Error saving fundamentals to SQL: {e}")
    
    def _load_events_from_sql(self, symbol: str, date: Optional[str] = None) -> Optional[Dict]:
        """Load market events from SQL database"""
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            
            # Load all recent events for this symbol
            cursor.execute("""
                SELECT * FROM market_events 
                WHERE symbol = ? 
                ORDER BY event_date DESC
            """, (symbol,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return None
            
            # Organize events by type
            events = {
                'symbol': symbol,
                'date': date or datetime.now().strftime('%Y-%m-%d'),
                'earnings': None,
                'dividend': None,
                'split': None,
            }
            
            for row in rows:
                row_dict = dict(row)
                event_type = row_dict['event_type']
                
                if event_type == 'earnings' and not events['earnings']:
                    events['earnings'] = {
                        'earnings_date': row_dict['earnings_date'],
                        'earnings_estimate_low': row_dict['earnings_estimate_low'],
                        'earnings_estimate_high': row_dict['earnings_estimate_high'],
                        'earnings_estimate_avg': row_dict['earnings_estimate_avg'],
                        'revenue_estimate_low': row_dict['revenue_estimate_low'],
                        'revenue_estimate_high': row_dict['revenue_estimate_high'],
                        'revenue_estimate_avg': row_dict['revenue_estimate_avg'],
                    }
                elif event_type == 'dividend' and not events['dividend']:
                    events['dividend'] = {
                        'last_dividend_date': row_dict['event_date'],
                        'last_dividend_amount': row_dict['dividend_amount'],
                    }
                elif event_type == 'split' and not events['split']:
                    events['split'] = {
                        'last_split_date': row_dict['event_date'],
                        'last_split_ratio': row_dict['split_ratio'],
                    }
            
            return events
            
        except Exception as e:
            logger.error(f"Error loading events from SQL: {e}")
            return None
    
    def _save_events_to_sql(self, events: Dict):
        """Save market events to SQL database"""
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save earnings event
            if events.get('earnings'):
                earnings = events['earnings']
                cursor.execute("""
                    INSERT OR REPLACE INTO market_events (
                        symbol, event_type, event_date, earnings_date,
                        earnings_estimate_low, earnings_estimate_high, earnings_estimate_avg,
                        revenue_estimate_low, revenue_estimate_high, revenue_estimate_avg,
                        dividend_amount, split_ratio, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    events['symbol'], 'earnings', earnings.get('earnings_date'),
                    earnings.get('earnings_date'), earnings.get('earnings_estimate_low'),
                    earnings.get('earnings_estimate_high'), earnings.get('earnings_estimate_avg'),
                    earnings.get('revenue_estimate_low'), earnings.get('revenue_estimate_high'),
                    earnings.get('revenue_estimate_avg'), None, None, created_at
                ))
            
            # Save dividend event
            if events.get('dividend'):
                dividend = events['dividend']
                cursor.execute("""
                    INSERT OR REPLACE INTO market_events (
                        symbol, event_type, event_date, earnings_date,
                        earnings_estimate_low, earnings_estimate_high, earnings_estimate_avg,
                        revenue_estimate_low, revenue_estimate_high, revenue_estimate_avg,
                        dividend_amount, split_ratio, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    events['symbol'], 'dividend', dividend.get('last_dividend_date'),
                    None, None, None, None, None, None, None,
                    dividend.get('last_dividend_amount'), None, created_at
                ))
            
            # Save split event
            if events.get('split'):
                split = events['split']
                cursor.execute("""
                    INSERT OR REPLACE INTO market_events (
                        symbol, event_type, event_date, earnings_date,
                        earnings_estimate_low, earnings_estimate_high, earnings_estimate_avg,
                        revenue_estimate_low, revenue_estimate_high, revenue_estimate_avg,
                        dividend_amount, split_ratio, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    events['symbol'], 'split', split.get('last_split_date'),
                    None, None, None, None, None, None, None, None,
                    split.get('last_split_ratio'), created_at
                ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Saved market events to SQL for {events['symbol']}")
            
        except Exception as e:
            logger.error(f"Error saving events to SQL: {e}")


# Convenience functions for feature_engineering.py integration
def get_fundamentals(symbol: str, db_manager: Optional[DatabaseManager] = None) -> Optional[Dict]:
    """
    Convenience function to get fundamentals with caching
    
    Args:
        symbol: Stock symbol
        db_manager: Optional database manager instance
        
    Returns:
        Dictionary with fundamental metrics or None
    """
    loader = FundamentalsLoader(db_manager)
    return loader.download_fundamentals(symbol)


def get_market_events(symbol: str, db_manager: Optional[DatabaseManager] = None) -> Dict:
    """
    Convenience function to get market events with caching
    
    Args:
        symbol: Stock symbol
        db_manager: Optional database manager instance
        
    Returns:
        Dictionary with market events data
    """
    loader = FundamentalsLoader(db_manager)
    return loader.download_market_events(symbol)
