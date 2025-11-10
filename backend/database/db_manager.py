"""
Database Manager for YoopRL Trading System
Handles all database operations using SQLite

This module manages:
- Equity history (portfolio value over time)
- Agent actions and decisions
- Performance metrics
- Risk events and alerts
- System logs

SQLite is used for:
- Simplicity: Single file, no server setup required
- Performance: Fast reads/writes for single application
- Sufficient capacity: Can handle years of trading data
- Easy backup: Just copy the .db file
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database for trading system data persistence
    
    Auto-cleanup: Removes data older than retention period to manage disk space
    Thread-safe: Uses connection pooling for concurrent reads
    """
    
    def __init__(self, db_path: str = "d:/YoopRL/data/trading.db", retention_days: int = 365):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
            retention_days: How many days of data to keep (default: 365 = 1 year)
        """
        self.db_path = db_path
        self.retention_days = retention_days
        
        # Create data directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._init_schema()
        
        logger.info(f"Database initialized at {db_path} with {retention_days} days retention")
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get database connection
        
        Returns connection with:
        - Row factory for dict-like access
        - WAL mode for better concurrency
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
        return conn
    
    def _init_schema(self):
        """
        Initialize database schema
        Creates all required tables if they don't exist
        
        Tables:
        - equity_history: Portfolio value over time (for equity curve)
        - agent_actions: All agent decisions and trades
        - performance_metrics: Daily/weekly/monthly performance stats
        - risk_events: Stop-loss, drawdown breaches, etc.
        - system_logs: General system events
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Equity History Table
            # Stores portfolio Net Liquidation value at 5-second intervals
            # Used for: Equity Curve chart, drawdown calculation, performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,              -- Unix timestamp (seconds since epoch)
                    datetime TEXT NOT NULL,                -- Human-readable datetime (ISO format)
                    net_liquidation REAL NOT NULL,         -- Total portfolio value (cash + positions)
                    buying_power REAL,                     -- Available buying power
                    cash REAL,                             -- Cash balance
                    unrealized_pnl REAL,                   -- Unrealized profit/loss
                    realized_pnl REAL,                     -- Realized profit/loss (closed trades)
                    gross_position_value REAL              -- Total value of open positions
                )
            """)
            
            # Create index for faster time-based queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_equity_timestamp 
                ON equity_history(timestamp)
            """)
            
            # Agent Actions Table
            # Stores every decision made by RL agents (PPO/SAC)
            # Used for: Performance analysis, backtesting, model improvement
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    agent_name TEXT NOT NULL,              -- 'PPO' or 'SAC'
                    symbol TEXT NOT NULL,                  -- Asset traded (e.g., 'AAPL', 'TNA')
                    action TEXT NOT NULL,                  -- 'BUY', 'SELL', 'HOLD'
                    quantity REAL NOT NULL,                -- Number of shares
                    price REAL NOT NULL,                   -- Execution price
                    reward REAL,                           -- Reward received for this action
                    rationale TEXT,                        -- Agent's reasoning (optional)
                    confidence REAL,                       -- Confidence score (0-1)
                    state_json TEXT                        -- Full state at time of decision (JSON)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_actions_timestamp 
                ON agent_actions(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_actions_symbol 
                ON agent_actions(symbol)
            """)
            
            # Performance Metrics Table
            # Aggregated daily/weekly/monthly statistics
            # Used for: Performance dashboard, historical analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,                    -- Date (YYYY-MM-DD)
                    period_type TEXT NOT NULL,             -- 'daily', 'weekly', 'monthly'
                    total_pnl REAL,                        -- Total profit/loss
                    win_rate REAL,                         -- Percentage of winning trades
                    sharpe_ratio REAL,                     -- Risk-adjusted return
                    sortino_ratio REAL,                    -- Downside risk-adjusted return
                    max_drawdown REAL,                     -- Maximum peak-to-trough decline
                    total_trades INTEGER,                  -- Number of trades executed
                    winning_trades INTEGER,                -- Number of profitable trades
                    losing_trades INTEGER,                 -- Number of losing trades
                    avg_win REAL,                          -- Average winning trade size
                    avg_loss REAL                          -- Average losing trade size
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_date 
                ON performance_metrics(date)
            """)
            
            # Risk Events Table
            # Critical events that require attention or triggered risk management
            # Used for: Alerts, risk monitoring, compliance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    event_type TEXT NOT NULL,              -- 'STOP_LOSS', 'DRAWDOWN_BREACH', 'POSITION_LIMIT', etc.
                    severity TEXT NOT NULL,                -- 'INFO', 'WARNING', 'CRITICAL'
                    agent_name TEXT,                       -- Affected agent (if applicable)
                    symbol TEXT,                           -- Affected symbol (if applicable)
                    description TEXT NOT NULL,             -- Event details
                    value REAL,                            -- Relevant value (e.g., drawdown %, position size)
                    threshold REAL,                        -- Threshold that was breached
                    action_taken TEXT                      -- What action was taken (e.g., 'POSITION_CLOSED')
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_risk_timestamp 
                ON risk_events(timestamp)
            """)
            
            # System Logs Table
            # General system events and status changes
            # Used for: Debugging, monitoring, audit trail
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    component TEXT NOT NULL,               -- 'IBKR_BRIDGE', 'PPO_AGENT', 'SAC_AGENT', 'WATCHDOG', etc.
                    level TEXT NOT NULL,                   -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
                    message TEXT NOT NULL,                 -- Log message
                    details_json TEXT                      -- Additional details (JSON)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp 
                ON system_logs(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_component 
                ON system_logs(component)
            """)
            
            # Market Data Table
            # Historical OHLCV data for training
            # Used for: RL agent training, backtesting, technical analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,                  -- Stock symbol (e.g., 'IWM', 'SPY')
                    timestamp REAL NOT NULL,               -- Unix timestamp (seconds since epoch)
                    datetime TEXT NOT NULL,                -- Human-readable datetime (ISO format)
                    open REAL,                             -- Opening price
                    high REAL,                             -- High price
                    low REAL,                              -- Low price
                    close REAL,                            -- Closing price
                    volume REAL,                           -- Trading volume
                    adj_close REAL,                        -- Adjusted close price
                    UNIQUE(symbol, timestamp)              -- Prevent duplicate entries
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_symbol_timestamp 
                ON market_data(symbol, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_datetime 
                ON market_data(datetime)
            """)
            
            # Intraday Market Data Table
            # Stores high-frequency bars (e.g., 15m) with session metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intraday_market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    datetime TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    session_date TEXT NOT NULL,
                    minutes_from_open REAL,
                    bar_index INTEGER,
                    time_fraction REAL,
                    is_session_end INTEGER,
                    UNIQUE(symbol, interval, timestamp)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_intraday_symbol_interval_datetime
                ON intraday_market_data(symbol, interval, datetime)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_intraday_session
                ON intraday_market_data(symbol, interval, session_date)
            """)

            # Sentiment Data Table
            # Daily sentiment features from news and social media
            # Used for: Enhancing RL agent features with market sentiment
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,                  -- Stock symbol
                    date TEXT NOT NULL,                    -- Date (YYYY-MM-DD)
                    sentiment_mean REAL,                   -- Average sentiment score [-1, 1]
                    sentiment_std REAL,                    -- Sentiment standard deviation
                    sentiment_positive_ratio REAL,         -- Ratio of positive sentiments
                    sentiment_negative_ratio REAL,         -- Ratio of negative sentiments
                    sentiment_neutral_ratio REAL,          -- Ratio of neutral sentiments
                    sentiment_source_count REAL,           -- Number of sentiment sources
                    google_trends REAL,                    -- Google Trends interest score (LEGACY - social sentiment)
                    
                    -- Professional API sentiment columns (Enterprise Sentiment Analyzer)
                    pro_sentiment REAL,                    -- Average professional sentiment score
                    pro_relevance_score REAL,              -- Alpha Vantage relevance score
                    pro_article_count INTEGER,             -- Finnhub article count
                    pro_buzz REAL,                         -- Finnhub buzz score
                    pro_articles_in_last_week INTEGER,     -- Finnhub weekly articles
                    pro_newsapi_sentiment REAL,            -- NewsAPI TextBlob sentiment mean
                    pro_newsapi_sentiment_std REAL,        -- NewsAPI sentiment std deviation
                    pro_newsapi_articles INTEGER,          -- NewsAPI article count
                    
                    UNIQUE(symbol, date)                   -- One entry per symbol per day
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_date 
                ON sentiment_data(symbol, date)
            """)
            
            # Migration: Add professional sentiment columns if they don't exist
            # This allows upgrading existing databases without losing data
            try:
                cursor.execute("SELECT pro_sentiment FROM sentiment_data LIMIT 1")
            except sqlite3.OperationalError:
                # Columns don't exist - add them
                logger.info("Migrating sentiment_data table to add professional columns...")
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN pro_sentiment REAL")
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN pro_relevance_score REAL")
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN pro_article_count INTEGER")
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN pro_buzz REAL")
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN pro_articles_in_last_week INTEGER")
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN pro_newsapi_sentiment REAL")
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN pro_newsapi_sentiment_std REAL")
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN pro_newsapi_articles INTEGER")
                logger.info("✓ Professional sentiment columns added successfully")
            
            # Fundamental Data Table
            # Company fundamental metrics from yfinance
            # Used for: Value investing signals, quality scoring, financial health
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fundamental_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,                  -- Stock symbol
                    date TEXT NOT NULL,                    -- Date of snapshot (YYYY-MM-DD)
                    
                    -- Valuation Metrics
                    forward_pe REAL,                       -- Forward P/E ratio
                    price_to_book REAL,                    -- Price to Book ratio
                    market_cap REAL,                       -- Market capitalization
                    enterprise_value REAL,                 -- Enterprise value
                    
                    -- Profitability Metrics
                    profit_margins REAL,                   -- Profit margin (%)
                    operating_margins REAL,                -- Operating margin (%)
                    gross_margins REAL,                    -- Gross margin (%)
                    return_on_equity REAL,                 -- ROE (%)
                    return_on_assets REAL,                 -- ROA (%)
                    
                    -- Growth Metrics
                    revenue_growth REAL,                   -- Revenue growth YoY (%)
                    total_revenue REAL,                    -- Total revenue
                    
                    -- Financial Health
                    debt_to_equity REAL,                   -- Debt to Equity ratio
                    current_ratio REAL,                    -- Current ratio
                    quick_ratio REAL,                      -- Quick ratio (acid test)
                    
                    -- Cash Flow
                    free_cashflow REAL,                    -- Free cash flow
                    operating_cashflow REAL,               -- Operating cash flow
                    
                    -- Per Share Metrics
                    trailing_eps REAL,                     -- Trailing EPS
                    forward_eps REAL,                      -- Forward EPS
                    book_value REAL,                       -- Book value per share
                    
                    -- Risk Metrics
                    beta REAL,                             -- Beta (volatility vs market)
                    
                    -- Ownership
                    shares_outstanding REAL,               -- Total shares outstanding
                    held_percent_institutions REAL,        -- Institutional ownership (%)
                    short_ratio REAL,                      -- Short ratio (days to cover)
                    
                    -- Analyst Targets
                    target_mean_price REAL,                -- Mean analyst target price
                    
                    UNIQUE(symbol, date)                   -- One snapshot per symbol per day
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_fundamental_symbol_date 
                ON fundamental_data(symbol, date)
            """)
            
            # Market Events Table
            # Corporate actions and scheduled events from yfinance
            # Used for: Event-driven trading signals, risk management
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,                  -- Stock symbol
                    event_type TEXT NOT NULL,              -- 'earnings', 'dividend', 'split'
                    event_date TEXT NOT NULL,              -- Date of event (YYYY-MM-DD)
                    
                    -- Earnings Data
                    earnings_date TEXT,                    -- Scheduled earnings announcement
                    earnings_estimate_low REAL,            -- EPS estimate low
                    earnings_estimate_high REAL,           -- EPS estimate high
                    earnings_estimate_avg REAL,            -- EPS estimate average
                    revenue_estimate_low REAL,             -- Revenue estimate low
                    revenue_estimate_high REAL,            -- Revenue estimate high
                    revenue_estimate_avg REAL,             -- Revenue estimate average
                    
                    -- Dividend Data
                    dividend_amount REAL,                  -- Dividend per share
                    
                    -- Stock Split Data
                    split_ratio TEXT,                      -- Split ratio (e.g., "2:1")
                    
                    created_at TEXT NOT NULL,              -- When this record was created
                    
                    UNIQUE(symbol, event_type, event_date) -- Prevent duplicate events
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_symbol_date 
                ON market_events(symbol, event_date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type 
                ON market_events(event_type)
            """)
            
            # Macro Indicators Table
            # Economic indicators and market-wide metrics
            # Used for: Regime detection, risk-on/risk-off signals, correlation analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS macro_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,                    -- Date (YYYY-MM-DD)
                    
                    -- Market Sentiment
                    vix REAL,                              -- CBOE Volatility Index (fear gauge)
                    vix_change REAL,                       -- Daily VIX change
                    
                    -- Interest Rates
                    treasury_10y REAL,                     -- 10-Year Treasury Yield (%)
                    treasury_2y REAL,                      -- 2-Year Treasury Yield (%)
                    yield_curve_spread REAL,               -- 10Y - 2Y spread (inversion signal)
                    
                    -- Currency
                    dxy REAL,                              -- US Dollar Index
                    dxy_change REAL,                       -- Daily DXY change
                    
                    -- Commodities
                    oil_price REAL,                        -- Crude Oil WTI price
                    gold_price REAL,                       -- Gold spot price
                    
                    -- FRED Economic Data
                    gdp_growth REAL,                       -- GDP growth rate (%)
                    unemployment_rate REAL,                -- Unemployment rate (%)
                    cpi REAL,                              -- Consumer Price Index
                    fed_funds_rate REAL,                   -- Federal Funds Rate (%)
                    
                    UNIQUE(date)                           -- One entry per day
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_macro_date 
                ON macro_indicators(date)
            """)
            
            conn.commit()
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def save_equity_point(self, net_liquidation: float, buying_power: float = None, 
                         cash: float = None, unrealized_pnl: float = None, 
                         realized_pnl: float = None, gross_position_value: float = None):
        """
        Save a single equity data point
        
        This should be called every 5 seconds by the IBKR service
        to build the equity curve over time
        
        Args:
            net_liquidation: Total portfolio value (required)
            buying_power: Available buying power (optional)
            cash: Cash balance (optional)
            unrealized_pnl: Unrealized P&L from open positions (optional)
            realized_pnl: Realized P&L from closed trades (optional)
            gross_position_value: Total value of positions (optional)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            now = datetime.now()
            timestamp = now.timestamp()
            datetime_str = now.isoformat()
            
            cursor.execute("""
                INSERT INTO equity_history 
                (timestamp, datetime, net_liquidation, buying_power, cash, 
                 unrealized_pnl, realized_pnl, gross_position_value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, datetime_str, net_liquidation, buying_power, cash,
                  unrealized_pnl, realized_pnl, gross_position_value))
            
            conn.commit()
            logger.debug(f"Saved equity point: ${net_liquidation:,.2f}")
            
        except Exception as e:
            logger.error(f"Error saving equity point: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_equity_history(self, hours: int = None, start_date: str = None, 
                          end_date: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve equity history for charting
        
        Args:
            hours: Get last N hours (default: all data)
            start_date: Start date (ISO format, optional)
            end_date: End date (ISO format, optional)
            
        Returns:
            List of equity points with timestamp and value
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            query = "SELECT * FROM equity_history WHERE 1=1"
            params = []
            
            if hours:
                cutoff = (datetime.now() - timedelta(hours=hours)).timestamp()
                query += " AND timestamp >= ?"
                params.append(cutoff)
            
            if start_date:
                start_ts = datetime.fromisoformat(start_date).timestamp()
                query += " AND timestamp >= ?"
                params.append(start_ts)
            
            if end_date:
                end_ts = datetime.fromisoformat(end_date).timestamp()
                query += " AND timestamp <= ?"
                params.append(end_ts)
            
            query += " ORDER BY timestamp ASC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error retrieving equity history: {e}")
            raise
        finally:
            conn.close()
    
    def cleanup_old_data(self):
        """
        Remove data older than retention period
        
        This is called automatically to manage disk space
        Keeps only the most recent data based on retention_days setting
        
        Data removed:
        - Equity history older than retention_days
        - Agent actions older than retention_days
        - Logs older than retention_days
        
        Data preserved:
        - Performance metrics (aggregated, small size)
        - Risk events (important for compliance)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cutoff = (datetime.now() - timedelta(days=self.retention_days)).timestamp()
            cutoff_date = datetime.fromtimestamp(cutoff).isoformat()
            
            # Clean equity history
            cursor.execute("DELETE FROM equity_history WHERE timestamp < ?", (cutoff,))
            equity_deleted = cursor.rowcount
            
            # Clean agent actions
            cursor.execute("DELETE FROM agent_actions WHERE timestamp < ?", (cutoff,))
            actions_deleted = cursor.rowcount
            
            # Clean system logs
            cursor.execute("DELETE FROM system_logs WHERE timestamp < ?", (cutoff,))
            logs_deleted = cursor.rowcount
            
            # Vacuum to reclaim disk space
            cursor.execute("VACUUM")
            
            conn.commit()
            
            logger.info(f"Cleanup completed: Removed {equity_deleted} equity points, "
                       f"{actions_deleted} actions, {logs_deleted} logs older than {cutoff_date}")
            
            return {
                'equity_deleted': equity_deleted,
                'actions_deleted': actions_deleted,
                'logs_deleted': logs_deleted,
                'cutoff_date': cutoff_date
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def log_system_event(
        self,
        component: str,
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a system log entry for operational auditing."""

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            now = datetime.now()
            timestamp = now.timestamp()
            cursor.execute(
                """
                INSERT INTO system_logs (timestamp, datetime, component, level, message, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    now.isoformat(),
                    component,
                    level.upper(),
                    message,
                    json.dumps(details) if details is not None else None,
                ),
            )
            conn.commit()
        except Exception as exc:  # pragma: no cover - logging shouldn't break main flow
            logger.error("Failed to log system event: %s", exc)
            conn.rollback()
        finally:
            conn.close()
    
    def save_market_data(self, symbol: str, df: 'pd.DataFrame'):
        """
        Save market data (OHLCV) to database
        
        Uses INSERT OR REPLACE to handle updates gracefully
        Prevents duplicate entries with UNIQUE constraint on (symbol, timestamp)
        
        Args:
            symbol: Stock ticker symbol (e.g., 'IWM', 'SPY')
            df: DataFrame with OHLCV data (index = datetime, columns = Open, High, Low, Close, Volume, Adj Close)
        """
        import pandas as pd
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            rows_added = 0
            
            for idx, row in df.iterrows():
                # Convert pandas Timestamp to datetime
                dt = pd.to_datetime(idx)
                timestamp = dt.timestamp()
                datetime_str = dt.isoformat()
                
                # INSERT OR REPLACE to handle duplicates
                cursor.execute("""
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, datetime, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    timestamp,
                    datetime_str,
                    float(row.get('Open', 0)) if pd.notna(row.get('Open')) else None,
                    float(row.get('High', 0)) if pd.notna(row.get('High')) else None,
                    float(row.get('Low', 0)) if pd.notna(row.get('Low')) else None,
                    float(row.get('Close', 0)) if pd.notna(row.get('Close')) else None,
                    float(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None,
                    float(row.get('Adj Close', row.get('Close', 0))) if pd.notna(row.get('Adj Close', row.get('Close'))) else None
                ))
                rows_added += 1
            
            conn.commit()
            logger.info(f"Saved {rows_added} market data rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving market data for {symbol}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_market_data(self, symbol: str, start_date: str = None, end_date: str = None) -> 'pd.DataFrame':
        """
        Retrieve market data from database
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (ISO format YYYY-MM-DD, optional)
            end_date: End date (ISO format YYYY-MM-DD, optional)
            
        Returns:
            DataFrame with OHLCV data (index = datetime)
        """
        import pandas as pd
        
        conn = self._get_connection()
        
        try:
            query = """
                SELECT datetime, open, high, low, close, volume, adj_close
                FROM market_data 
                WHERE symbol = ?
            """
            params = [symbol]
            
            if start_date:
                query += " AND datetime >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND datetime <= ?"
                params.append(end_date)
            
            query += " ORDER BY datetime ASC"
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['datetime'], index_col='datetime')
            
            # Rename columns to match yfinance format (capitalized)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            
            logger.info(f"Retrieved {len(df)} market data rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving market data for {symbol}: {e}")
            raise
        finally:
            conn.close()

    def get_market_date_bounds(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return earliest and latest dates available for daily market data."""

        query = (
            "SELECT MIN(datetime) AS min_date, MAX(datetime) AS max_date "
            "FROM market_data WHERE symbol = ?"
        )

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(query, (symbol.upper(),))
            row = cursor.fetchone()
            if not row or not row["min_date"] or not row["max_date"]:
                return None

            return {
                "min_date": row["min_date"],
                "max_date": row["max_date"],
            }
        except Exception as exc:
            logger.error("Error retrieving market data bounds for %s: %s", symbol, exc)
            return None
        finally:
            conn.close()

    def save_intraday_data(self, symbol: str, interval: str, df: 'pd.DataFrame') -> int:
        """Persist intraday OHLCV bars with session metadata."""

        import pandas as pd
        import pytz

        if df.empty:
            return 0

        required_columns = {
            'Open', 'High', 'Low', 'Close', 'Volume',
            'session_date', 'minutes_from_open', 'bar_index',
            'time_fraction', 'is_session_end'
        }
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Intraday DataFrame missing columns: {sorted(missing)}")

        ny_tz = pytz.timezone("America/New_York")
        symbol = symbol.upper()
        interval = interval.lower()

        def _safe_float(value):
            return float(value) if pd.notna(value) else None

        def _safe_int(value):
            return int(value) if pd.notna(value) else None

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            rows = []
            for idx, row in df.iterrows():
                ts = pd.to_datetime(idx)
                if ts.tzinfo is None:
                    ts = ts.tz_localize(ny_tz)
                else:
                    ts = ts.tz_convert(ny_tz)
                ts_utc = ts.tz_convert(pytz.UTC)

                rows.append((
                    symbol,
                    interval,
                    float(ts_utc.timestamp()),
                    ts.isoformat(),
                    _safe_float(row.get('Open')),
                    _safe_float(row.get('High')),
                    _safe_float(row.get('Low')),
                    _safe_float(row.get('Close')),
                    _safe_float(row.get('Volume')),
                    str(row.get('session_date')),
                    _safe_float(row.get('minutes_from_open')),
                    _safe_int(row.get('bar_index')),
                    _safe_float(row.get('time_fraction')),
                    1 if bool(row.get('is_session_end')) else 0
                ))

            cursor.executemany(
                """
                INSERT OR REPLACE INTO intraday_market_data (
                    symbol, interval, timestamp, datetime,
                    open, high, low, close, volume,
                    session_date, minutes_from_open, bar_index,
                    time_fraction, is_session_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows
            )

            conn.commit()
            logger.info(
                "Saved %s intraday rows for %s (%s)",
                len(rows), symbol, interval
            )
            return len(rows)

        except Exception as exc:
            logger.error("Error saving intraday data for %s (%s): %s", symbol, interval, exc)
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_intraday_session_bounds(
        self,
        symbol: str,
        interval: str = "15m"
    ) -> Optional[Dict[str, Any]]:
        """Return earliest and latest session dates for intraday data."""

        query = (
            "SELECT MIN(session_date) AS min_date, MAX(session_date) AS max_date "
            "FROM intraday_market_data WHERE symbol = ? AND interval = ?"
        )

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, (symbol.upper(), interval))
            row = cursor.fetchone()
            if not row or not row["min_date"] or not row["max_date"]:
                return None

            return {
                "min_date": row["min_date"],
                "max_date": row["max_date"],
            }
        except Exception as exc:
            logger.error(
                "Error retrieving intraday session bounds for %s (%s): %s",
                symbol,
                interval,
                exc,
            )
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def get_intraday_data(
        self,
        symbol: str,
        interval: str = "15m",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> 'pd.DataFrame':
        """Retrieve intraday bars with session metadata from SQLite."""

        import pandas as pd
        import pytz

        conn = self._get_connection()

        try:
            query = (
                "SELECT datetime, open, high, low, close, volume, session_date, "
                "minutes_from_open, bar_index, time_fraction, is_session_end "
                "FROM intraday_market_data WHERE symbol = ? AND interval = ?"
            )
            params: list[Any] = [symbol.upper(), interval.lower()]

            if start:
                query += " AND datetime >= ?"
                params.append(start)

            if end:
                query += " AND datetime <= ?"
                params.append(end)

            query += " ORDER BY datetime ASC"

            df = pd.read_sql_query(query, conn, params=params)
            if df.empty:
                return df

            # Normalize datetime index to a consistent timezone to avoid mixed tz errors
            utc_index = pd.DatetimeIndex(pd.to_datetime(df['datetime'], utc=True))
            ny_tz = pytz.timezone("America/New_York")
            dt_index = utc_index.tz_convert(ny_tz)

            df = df.drop(columns=['datetime'])
            df.index = dt_index
            df['is_session_end'] = df['is_session_end'].astype(bool)

            return df

        except Exception as exc:
            logger.error("Error retrieving intraday data for %s (%s): %s", symbol, interval, exc)
            raise
        finally:
            conn.close()
    
    def get_latest_market_date(self, symbol: str) -> Optional['datetime']:
        """
        Get the most recent date for which we have market data
        
        Used to determine how many days of data are missing
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Latest date as datetime object, or None if no data exists
        """
        from datetime import datetime
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT MAX(datetime) as latest_date
                FROM market_data
                WHERE symbol = ?
            """, (symbol,))
            
            result = cursor.fetchone()
            
            if result and result['latest_date']:
                # Parse ISO format datetime string to datetime object
                latest_date_str = result['latest_date']
                latest_dt = datetime.fromisoformat(latest_date_str)
                logger.debug(f"Latest market data for {symbol}: {latest_dt.date()}")
                return latest_dt.date()
            
            logger.debug(f"No market data found for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest market date for {symbol}: {e}")
            raise
        finally:
            conn.close()
    
    def save_sentiment_data(self, symbol: str, date: str, sentiment_data: Dict[str, float]):
        """
        Save sentiment features for a specific symbol and date
        
        Supports both legacy (social) and professional (enterprise) sentiment formats
        
        Args:
            symbol: Stock ticker symbol
            date: Date (YYYY-MM-DD format)
            sentiment_data: Dictionary with sentiment metrics
                Professional format (8 keys):
                    - sentiment: Average sentiment score
                    - relevance_score: Alpha Vantage relevance
                    - article_count: Finnhub article count
                    - buzz: Finnhub buzz score
                    - articles_in_last_week: Finnhub weekly articles
                    - newsapi_sentiment: NewsAPI sentiment mean
                    - newsapi_sentiment_std: NewsAPI sentiment std
                    - newsapi_articles: NewsAPI article count
                
                Legacy format (7 keys) - for backup/fallback:
                    - sentiment_mean, sentiment_std, positive_ratio, etc.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Detect format (professional vs legacy)
            is_professional = 'sentiment' in sentiment_data and 'relevance_score' in sentiment_data
            
            if is_professional:
                # Professional sentiment (EnterpriseSentimentAnalyzer)
                cursor.execute("""
                    INSERT OR REPLACE INTO sentiment_data 
                    (symbol, date, 
                     pro_sentiment, pro_relevance_score, pro_article_count, pro_buzz,
                     pro_articles_in_last_week, pro_newsapi_sentiment, 
                     pro_newsapi_sentiment_std, pro_newsapi_articles)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    date,
                    sentiment_data.get('sentiment', 0.0),
                    sentiment_data.get('relevance_score', 0.0),
                    sentiment_data.get('article_count', 0),
                    sentiment_data.get('buzz', 0.0),
                    sentiment_data.get('articles_in_last_week', 0),
                    sentiment_data.get('newsapi_sentiment', 0.0),
                    sentiment_data.get('newsapi_sentiment_std', 0.0),
                    sentiment_data.get('newsapi_articles', 0)
                ))
                logger.debug(f"Saved professional sentiment for {symbol} on {date}")
            else:
                # Legacy social sentiment (SentimentFeatureAggregator)
                cursor.execute("""
                    INSERT OR REPLACE INTO sentiment_data 
                    (symbol, date, sentiment_mean, sentiment_std, 
                     sentiment_positive_ratio, sentiment_negative_ratio, 
                     sentiment_neutral_ratio, sentiment_source_count, google_trends)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    date,
                    sentiment_data.get('sentiment_mean', 0.0),
                    sentiment_data.get('sentiment_std', 0.0),
                    sentiment_data.get('positive_ratio', 0.5),
                    sentiment_data.get('negative_ratio', 0.5),
                    sentiment_data.get('neutral_ratio', 0.0),
                    sentiment_data.get('source_count', 0.0),
                    sentiment_data.get('google_trends', 0.0)
                ))
                logger.debug(f"Saved legacy social sentiment for {symbol} on {date}")
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving sentiment data for {symbol}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
            conn.close()
    
    def get_sentiment_data(self, symbol: str, date_str: str = None) -> Optional[Dict[str, float]]:
        """
        Retrieve sentiment data for a specific symbol and date
        
        Automatically detects format (professional vs legacy) and returns appropriate dict
        
        ⚠️ IMPORTANT - SENTIMENT FORMAT CONFIGURATION:
        
        OPTION 2 (CURRENT - Professional APIs):
        - Returns professional sentiment with 8 features: sentiment, relevance_score, article_count, 
          buzz, articles_in_last_week, newsapi_sentiment, newsapi_sentiment_std, newsapi_articles
        - If only legacy data exists, returns None to trigger fresh professional download
        - Used when feature_engineering.py uses EnterpriseSentimentAnalyzer
        
        OPTION 1 (BACKUP - Social Sentiment):
        - If switching back to SentimentFeatureAggregator in feature_engineering.py:
          1. Update the else block below to return legacy format instead of None:
             return {
                 'sentiment_mean': result['sentiment_mean'],
                 'sentiment_std': result['sentiment_std'],
                 'positive_ratio': result['sentiment_positive_ratio'],
                 'negative_ratio': result['sentiment_negative_ratio'],
                 'neutral_ratio': result['sentiment_neutral_ratio'],
                 'source_count': result['sentiment_source_count'],
                 'google_trends': result['google_trends']
             }
          2. This allows using cached social sentiment without re-downloading
        
        Args:
            symbol: Stock ticker symbol
            date_str: Date (YYYY-MM-DD format, optional - if None, gets latest)
            
        Returns:
            Dictionary with sentiment features (professional or legacy format), or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            if date_str:
                cursor.execute("""
                    SELECT * FROM sentiment_data
                    WHERE symbol = ? AND date = ?
                """, (symbol, date_str))
            else:
                cursor.execute("""
                    SELECT * FROM sentiment_data
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT 1
                """, (symbol,))
            
            result = cursor.fetchone()
            
            if result:
                # OPTION 2: Check if professional sentiment exists (prioritize over legacy)
                if result['pro_sentiment'] is not None:
                    # Return professional sentiment format (8 features)
                    return {
                        'sentiment': result['pro_sentiment'],
                        'relevance_score': result['pro_relevance_score'] or 0.0,
                        'article_count': result['pro_article_count'] or 0,
                        'buzz': result['pro_buzz'] or 0.0,
                        'articles_in_last_week': result['pro_articles_in_last_week'] or 0,
                        'newsapi_sentiment': result['pro_newsapi_sentiment'] or 0.0,
                        'newsapi_sentiment_std': result['pro_newsapi_sentiment_std'] or 0.0,
                        'newsapi_articles': result['pro_newsapi_articles'] or 0
                    }
                else:
                    # OPTION 2 BEHAVIOR: Legacy social sentiment exists but professional doesn't
                    # Return None to trigger fresh professional sentiment download
                    # NOTE: If switching to OPTION 1 (SentimentFeatureAggregator), 
                    #       change this to return legacy format instead of None (see docstring)
                    logger.debug(f"Legacy sentiment found for {symbol}, will fetch professional sentiment instead")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving sentiment data for {symbol}: {e}")
            raise
        finally:
            conn.close()
    
    def log_agent_action(
        self,
        agent_name: str,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        reward: Optional[float] = None,
        rationale: Optional[str] = None,
        confidence: Optional[float] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a single agent decision for monitoring and audit trails."""

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            now = datetime.now()
            cursor.execute(
                """
                INSERT INTO agent_actions (
                    timestamp, datetime, agent_name, symbol, action,
                    quantity, price, reward, rationale, confidence, state_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now.timestamp(),
                    now.isoformat(),
                    agent_name,
                    symbol,
                    action,
                    float(quantity),
                    float(price),
                    reward if reward is not None else None,
                    rationale,
                    confidence if confidence is not None else None,
                    json.dumps(state) if state is not None else None,
                ),
            )
            conn.commit()
        except Exception as exc:  # pragma: no cover - logging should not interrupt flow
            logger.error("Failed to log agent action: %s", exc)
            conn.rollback()
        finally:
            conn.close()

    def log_risk_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        agent_name: Optional[str] = None,
        symbol: Optional[str] = None,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
        action_taken: Optional[str] = None,
    ) -> None:
        """Record a risk alert for alerting and later review."""

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            now = datetime.now()
            cursor.execute(
                """
                INSERT INTO risk_events (
                    timestamp, datetime, event_type, severity, agent_name, symbol,
                    description, value, threshold, action_taken
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now.timestamp(),
                    now.isoformat(),
                    event_type,
                    severity.upper(),
                    agent_name,
                    symbol,
                    description,
                    value,
                    threshold,
                    action_taken,
                ),
            )
            conn.commit()
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to log risk event: %s", exc)
            conn.rollback()
        finally:
            conn.close()

    def get_recent_agent_actions(
        self,
        limit: int = 50,
        agent_name: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most recent agent actions ordered by timestamp descending."""

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM agent_actions WHERE 1=1"
            params: List[Any] = []

            if agent_name:
                query += " AND agent_name = ?"
                params.append(agent_name)

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            actions: List[Dict[str, Any]] = []
            for row in rows:
                payload = dict(row)
                state_json = payload.get("state_json")
                if state_json:
                    try:
                        payload["state"] = json.loads(state_json)
                    except json.JSONDecodeError:
                        payload["state"] = None
                payload.pop("state_json", None)
                actions.append(payload)

            return actions
        except Exception as exc:
            logger.error("Error fetching agent actions: %s", exc)
            raise
        finally:
            conn.close()

    def get_recent_risk_events(
        self,
        limit: int = 50,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch latest risk events sorted by time descending."""

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM risk_events WHERE 1=1"
            params: List[Any] = []

            if severity:
                query += " AND severity = ?"
                params.append(severity.upper())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.error("Error fetching risk events: %s", exc)
            raise
        finally:
            conn.close()

    def get_recent_system_logs(
        self,
        limit: int = 100,
        component: Optional[str] = None,
        level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent system log entries."""

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM system_logs WHERE 1=1"
            params: List[Any] = []

            if component:
                query += " AND component = ?"
                params.append(component)

            if level:
                query += " AND level = ?"
                params.append(level.upper())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            logs: List[Dict[str, Any]] = []
            for row in rows:
                entry = dict(row)
                details = entry.get("details_json")
                if details:
                    try:
                        entry["details"] = json.loads(details)
                    except json.JSONDecodeError:
                        entry["details"] = None
                entry.pop("details_json", None)
                logs.append(entry)

            return logs
        except Exception as exc:
            logger.error("Error fetching system logs: %s", exc)
            raise
        finally:
            conn.close()

    def get_equity_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Compute aggregate equity metrics for the monitoring dashboard."""

        history = self.get_equity_history(hours=hours)
        if not history:
            return {
                'hours': hours,
                'has_data': False,
                'net_liquidation': 0.0,
                'change_pct': 0.0,
                'max_drawdown_pct': 0.0,
                'buying_power': 0.0,
                'cash': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'history': [],
            }

        latest = history[-1]
        first = history[0]

        initial_value = float(first.get('net_liquidation') or 0.0)
        latest_value = float(latest.get('net_liquidation') or 0.0)
        change_pct = 0.0
        if initial_value:
            change_pct = ((latest_value - initial_value) / initial_value) * 100.0

        max_value = None
        max_drawdown_pct = 0.0
        for point in history:
            value = float(point.get('net_liquidation') or 0.0)
            if value <= 0:
                continue
            if max_value is None or value > max_value:
                max_value = value
            drawdown = (value - max_value) / max_value if max_value else 0.0
            if drawdown < max_drawdown_pct:
                max_drawdown_pct = drawdown

        max_drawdown_pct = abs(max_drawdown_pct * 100.0)

        trimmed_history = history[-200:] if len(history) > 200 else history
        curve = [
            {
                'timestamp': point['timestamp'],
                'datetime': point['datetime'],
                'net_liquidation': float(point.get('net_liquidation') or 0.0),
            }
            for point in trimmed_history
        ]

        return {
            'hours': hours,
            'has_data': True,
            'timestamp': latest.get('timestamp'),
            'datetime': latest.get('datetime'),
            'net_liquidation': latest_value,
            'buying_power': float(latest.get('buying_power') or 0.0),
            'cash': float(latest.get('cash') or 0.0),
            'unrealized_pnl': float(latest.get('unrealized_pnl') or 0.0),
            'realized_pnl': float(latest.get('realized_pnl') or 0.0),
            'gross_position_value': float(latest.get('gross_position_value') or 0.0),
            'change_pct': change_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'history': curve,
        }

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns info about:
        - Number of records in each table
        - Database file size
        - Date range of data
        - Disk space usage
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # Count records in each table
            tables = ['equity_history', 'agent_actions', 'performance_metrics', 
                     'risk_events', 'system_logs']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[f'{table}_count'] = cursor.fetchone()['count']
            
            # Get date range for equity history
            cursor.execute("""
                SELECT 
                    MIN(datetime) as oldest,
                    MAX(datetime) as newest
                FROM equity_history
            """)
            date_range = cursor.fetchone()
            stats['oldest_equity_date'] = date_range['oldest']
            stats['newest_equity_date'] = date_range['newest']
            
            # Get database file size
            db_file = Path(self.db_path)
            if db_file.exists():
                stats['db_size_mb'] = db_file.stat().st_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            raise
        finally:
            conn.close()
