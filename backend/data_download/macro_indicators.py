"""
Macro Economic Indicators Loader

This module downloads and caches market-wide economic indicators to enhance
RL agent's understanding of the broader economic environment.

Data Sources:
1. yfinance (Free, real-time):
   - VIX (^VIX): Volatility Index (fear gauge)
   - 10Y Treasury (^TNX): 10-Year Treasury Yield
   - 2Y Treasury (^IRX): 2-Year Treasury Yield  
   - DXY (DX-Y.NYB): US Dollar Index
   - Oil (CL=F): Crude Oil WTI Futures
   - Gold (GC=F): Gold Futures

2. FRED API (Federal Reserve Economic Data):
   - GDP Growth Rate
   - Unemployment Rate  
   - CPI (Consumer Price Index)
   - Fed Funds Rate

Why Macro Indicators Matter:
- Risk-On/Risk-Off detection: VIX spike → sell equities, buy safe havens
- Interest Rate sensitivity: Rising yields → tech stocks underperform
- Dollar strength: Strong dollar → commodities/emerging markets weak
- Yield curve: Inversion (2Y > 10Y) → recession warning
- Commodity trends: Oil/Gold as inflation hedges

Agent Benefits:
- Context-aware trading: Adjust position sizing based on VIX
- Regime detection: Bull market vs Bear market vs Sideways
- Correlation awareness: Tech stocks vs yields, gold vs dollar
- Risk management: Reduce exposure before earnings in high-VIX environment
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
from typing import Dict, Optional
import os
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class MacroIndicatorsLoader:
    """
    Downloads and caches macro economic indicators
    
    Strategy:
    - Download daily macro data with 24-hour TTL
    - Compute derived features (VIX change, yield curve spread)
    - Broadcast to all dataframe rows (macro is constant across all stocks on same day)
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None, fred_api_key: Optional[str] = None):
        """
        Initialize macro indicators loader
        
        Args:
            db_manager: Database manager instance
            fred_api_key: FRED API key for economic data (optional)
        """
        self.db = db_manager or DatabaseManager()
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        
        if not self.fred_api_key:
            logger.warning("FRED_API_KEY not found in environment - FRED data will be skipped")
        
        logger.info("MacroIndicatorsLoader initialized")
    
    def download_macro_indicators(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Download macro indicators for date range with SQL caching
        
        Returns DataFrame with daily macro indicators:
        - vix: CBOE Volatility Index
        - vix_change: Daily VIX change  
        - treasury_10y: 10-Year Treasury Yield (%)
        - treasury_2y: 2-Year Treasury Yield (%)
        - yield_curve_spread: 10Y - 2Y (recession indicator if negative)
        - dxy: US Dollar Index
        - dxy_change: Daily DXY change
        - oil_price: Crude Oil WTI price
        - gold_price: Gold spot price
        - gdp_growth: GDP growth rate (%) - from FRED
        - unemployment_rate: Unemployment rate (%) - from FRED
        - cpi: Consumer Price Index - from FRED
        - fed_funds_rate: Federal Funds Rate (%) - from FRED
        
        Args:
            start_date: Start date (YYYY-MM-DD), defaults to 5 years ago
            end_date: End date (YYYY-MM-DD), defaults to today
            force_refresh: Force download even if cached
            
        Returns:
            DataFrame with macro indicators indexed by date
        """
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        # Try SQL cache first
        if not force_refresh:
            cached = self._load_macro_from_sql(start_date, end_date)
            if cached is not None and not cached.empty:
                logger.info(f"✓ Loaded {len(cached)} days of macro data from SQL cache")
                return cached
        
        logger.info(f"Downloading macro indicators from {start_date} to {end_date}...")
        
        # Download yfinance data
        yf_data = self._download_yfinance_macro(start_date, end_date)
        
        # Download FRED data (if API key available)
        fred_data = self._download_fred_macro(start_date, end_date)
        
        # Merge datasets
        if fred_data is not None:
            macro_df = pd.merge(yf_data, fred_data, left_index=True, right_index=True, how='outer')
        else:
            macro_df = yf_data
        
        # Forward fill FRED data (it's released monthly/quarterly)
        fred_cols = ['gdp_growth', 'unemployment_rate', 'cpi', 'fed_funds_rate']
        for col in fred_cols:
            if col in macro_df.columns:
                macro_df[col] = macro_df[col].ffill()
        
        # Save to SQL
        self._save_macro_to_sql(macro_df)
        
        logger.info(f"✓ Downloaded {len(macro_df)} days of macro indicators")
        return macro_df
    
    def _download_yfinance_macro(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download market indicators from yfinance
        
        Symbols:
        - ^VIX: CBOE Volatility Index (fear gauge)
        - ^TNX: 10-Year Treasury Yield (interest rate benchmark)
        - ^IRX: 2-Year Treasury Yield (short-term rates)
        - DX-Y.NYB: US Dollar Index (dollar strength)
        - CL=F: Crude Oil WTI Futures (energy/inflation)
        - GC=F: Gold Futures (safe haven)
        """
        logger.info("  Downloading from yfinance (VIX, Yields, DXY, Oil, Gold)...")
        
        symbols = {
            '^VIX': 'vix',
            '^TNX': 'treasury_10y',
            '^IRX': 'treasury_2y',
            'DX-Y.NYB': 'dxy',
            'CL=F': 'oil_price',
            'GC=F': 'gold_price',
        }
        
        dfs = {}
        
        for symbol, col_name in symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                
                if not hist.empty:
                    dfs[col_name] = hist['Close']
                    logger.debug(f"    ✓ {symbol}: {len(hist)} days")
                else:
                    logger.warning(f"    ✗ {symbol}: No data")
                    
            except Exception as e:
                logger.warning(f"    ✗ {symbol}: {e}")
        
        # Combine into single DataFrame
        macro_df = pd.DataFrame(dfs)
        macro_df.index = pd.to_datetime(macro_df.index).date
        macro_df.index.name = 'date'
        
        # Compute derived features
        if 'vix' in macro_df.columns:
            macro_df['vix_change'] = macro_df['vix'].diff()
        
        if 'dxy' in macro_df.columns:
            macro_df['dxy_change'] = macro_df['dxy'].diff()
        
        if 'treasury_10y' in macro_df.columns and 'treasury_2y' in macro_df.columns:
            macro_df['yield_curve_spread'] = macro_df['treasury_10y'] - macro_df['treasury_2y']
        
        return macro_df
    
    def _download_fred_macro(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Download economic indicators from FRED API
        
        Series:
        - GDP: Gross Domestic Product growth rate
        - UNRATE: Unemployment Rate
        - CPIAUCSL: Consumer Price Index (All Urban Consumers)
        - FEDFUNDS: Federal Funds Effective Rate
        """
        if not self.fred_api_key:
            logger.debug("  Skipping FRED download (no API key)")
            return None
        
        logger.info("  Downloading from FRED (GDP, Unemployment, CPI, Fed Funds)...")
        
        try:
            import fredapi
            fred = fredapi.Fred(api_key=self.fred_api_key)
            
            series = {
                'GDP': 'gdp_growth',          # Quarterly
                'UNRATE': 'unemployment_rate', # Monthly
                'CPIAUCSL': 'cpi',            # Monthly
                'FEDFUNDS': 'fed_funds_rate', # Monthly
            }
            
            fred_data = {}
            
            for series_id, col_name in series.items():
                try:
                    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                    
                    # Convert GDP to growth rate if needed
                    if series_id == 'GDP':
                        data = data.pct_change(periods=4) * 100  # YoY growth rate
                    
                    fred_data[col_name] = data
                    logger.debug(f"    ✓ {series_id}: {len(data)} observations")
                    
                except Exception as e:
                    logger.warning(f"    ✗ {series_id}: {e}")
            
            if not fred_data:
                return None
            
            fred_df = pd.DataFrame(fred_data)
            fred_df.index = pd.to_datetime(fred_df.index).date
            fred_df.index.name = 'date'
            
            return fred_df
            
        except ImportError:
            logger.warning("  ✗ fredapi not installed - run: pip install fredapi")
            return None
        except Exception as e:
            logger.error(f"  ✗ FRED download failed: {e}")
            return None
    
    def _load_macro_from_sql(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load macro indicators from SQL database"""
        try:
            conn = self.db._get_connection()
            
            query = """
                SELECT * FROM macro_indicators 
                WHERE date >= ? AND date <= ?
                ORDER BY date
            """
            
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            conn.close()
            
            if df.empty:
                return None
            
            # Set index
            df['date'] = pd.to_datetime(df['date']).dt.date
            df = df.set_index('date')
            df = df.drop(columns=['id'], errors='ignore')
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading macro from SQL: {e}")
            return None
    
    def _save_macro_to_sql(self, macro_df: pd.DataFrame):
        """Save macro indicators to SQL database"""
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            
            for date, row in macro_df.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO macro_indicators (
                        date, vix, vix_change, treasury_10y, treasury_2y, yield_curve_spread,
                        dxy, dxy_change, oil_price, gold_price,
                        gdp_growth, unemployment_rate, cpi, fed_funds_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(date),
                    row.get('vix'), row.get('vix_change'),
                    row.get('treasury_10y'), row.get('treasury_2y'), row.get('yield_curve_spread'),
                    row.get('dxy'), row.get('dxy_change'),
                    row.get('oil_price'), row.get('gold_price'),
                    row.get('gdp_growth'), row.get('unemployment_rate'),
                    row.get('cpi'), row.get('fed_funds_rate')
                ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Saved {len(macro_df)} days of macro data to SQL")
            
        except Exception as e:
            logger.error(f"Error saving macro to SQL: {e}")


# Convenience function for feature_engineering.py integration
def get_macro_indicators(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_manager: Optional[DatabaseManager] = None
) -> pd.DataFrame:
    """
    Convenience function to get macro indicators with caching
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_manager: Optional database manager instance
        
    Returns:
        DataFrame with macro indicators indexed by date
    """
    loader = MacroIndicatorsLoader(db_manager)
    return loader.download_macro_indicators(start_date, end_date)
