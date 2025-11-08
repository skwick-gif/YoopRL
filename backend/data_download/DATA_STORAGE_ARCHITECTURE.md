# Data Storage & Training Pipeline Architecture

**Document Version:** 1.0  
**Date:** November 8, 2025  
**Author:** YoopRL Development Team

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Storage Strategy](#storage-strategy)
3. [Data Types & Storage Locations](#data-types--storage-locations)
4. [Database Schema](#database-schema)
5. [File Structure](#file-structure)
6. [Download Flow](#download-flow)
7. [Training Pipeline](#training-pipeline)
8. [Incremental Update Logic](#incremental-update-logic)
9. [Caching Strategy](#caching-strategy)
10. [Cleanup & Retention](#cleanup--retention)

---

## 🎯 Overview

The YoopRL trading system uses a **hybrid storage approach** combining SQL and CSV files to optimize for both data management and training performance.

### Core Principles:

1. **SQL = Master Storage** (Source of Truth)
   - Centralized data repository
   - Incremental updates
   - Historical data preservation
   - Query flexibility

2. **CSV = Training Cache** (Performance Optimization)
   - Fast data loading for RL agents
   - Reproducible training datasets
   - No database overhead during training

3. **Incremental Downloads** (Efficiency)
   - Download only missing data
   - Check last available date in SQL
   - Merge new data with existing

---

## 🗄️ Storage Strategy

### Why Hybrid Approach?

#### Problem with SQL-Only:
- ❌ Slow during training (query overhead per batch)
- ❌ DB lock contention with multiple training runs
- ❌ Complex indexing for time-series queries

#### Problem with CSV-Only:
- ❌ No incremental updates (re-download everything)
- ❌ Data duplication across projects
- ❌ Difficult to query/filter historical data

#### Solution: Hybrid
- ✅ SQL: Master storage with incremental updates
- ✅ CSV: Export for training (once per download)
- ✅ Best of both worlds

---

## 📊 Data Types & Storage Locations

### 1. OHLCV (Market Data)

**What:** Open, High, Low, Close, Volume, Adj Close  
**Sources:** Yahoo Finance (via yfinance)  
**Update Frequency:** Daily (market close)

**Storage:**
- **SQL Table:** `market_data`
- **CSV Export:** `data/training/{SYMBOL}_{TIMESTAMP}/raw.csv`

**Why SQL:**
- Long historical data (years)
- Incremental updates essential
- Shared across multiple training runs

**Format (SQL):**
```sql
market_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp REAL NOT NULL,
    datetime TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    adj_close REAL,
    UNIQUE(symbol, timestamp)
)
```

**Format (CSV):**
```csv
Date,Open,High,Low,Close,Volume,Adj Close
2025-11-08,228.45,229.12,227.89,228.67,12500000,228.67
```

---

### 2. Technical Indicators

**What:** SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.  
**Computation:** Calculated on-the-fly from OHLCV  
**Update Frequency:** Same as OHLCV

**Storage:**
- **SQL:** ❌ Not stored (computed dynamically)
- **CSV Export:** ✅ `data/training/{SYMBOL}_{TIMESTAMP}/processed.csv`

**Why NOT SQL:**
- Derived from OHLCV (redundant storage)
- Easy to recompute if needed
- Can change indicator parameters without re-downloading

**Why CSV:**
- Pre-computed for training speed
- Consistent features across train/test split
- Reproducible results

**Format (CSV):**
```csv
Date,Close,returns,log_returns,sma_20,ema_12,rsi_14,macd,atr,...
2025-11-08,228.67,0.0023,0.0023,227.45,228.12,62.3,0.45,1.23,...
```

---

### 3. Sentiment Data

**What:** News sentiment, social media sentiment, buzz metrics  
**Sources:**
- Yahoo Finance News
- Reddit r/wallstreetbets
- StockTwits
- Alpha Vantage API
- Finnhub API

**Update Frequency:** Daily (6-hour cache TTL)

**Storage:**
- **SQL Table:** `sentiment_data` (recommended)
- **JSON Cache:** `data/cache/sentiment/{SYMBOL}.json` (temporary, 6h TTL)
- **CSV Export:** Merged into `processed.csv`

**Why SQL:**
- Sentiment is a **valuable feature** for training
- Historical sentiment trends are important
- Daily granularity (not too much data)

**Why JSON Cache:**
- Temporary cache to avoid API rate limits
- 6-hour TTL (sentiment changes frequently)
- Fast lookup during feature engineering

**Format (SQL):**
```sql
sentiment_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    sentiment_mean REAL,
    sentiment_std REAL,
    sentiment_positive_ratio REAL,
    sentiment_negative_ratio REAL,
    sentiment_neutral_ratio REAL,
    sentiment_source_count REAL,
    google_trends REAL,
    UNIQUE(symbol, date)
)
```

**Format (JSON Cache):**
```json
{
  "timestamp": "2025-11-08T14:30:00",
  "symbol": "IWM",
  "features": {
    "sentiment_mean": 0.15,
    "sentiment_std": 0.32,
    "sentiment_positive_ratio": 0.62,
    "sentiment_negative_ratio": 0.28,
    "sentiment_neutral_ratio": 0.10,
    "sentiment_source_count": 25.0,
    "google_trends": 0.1
  }
}
```

---

### 4. Fundamental Data (Future Implementation)

**What:** P/E ratio, EPS, Revenue, Market Cap, Balance Sheet  
**Sources:** Alpha Vantage, Finnhub, Yahoo Finance  
**Update Frequency:** Quarterly (earnings reports)

**Storage:**
- **SQL Table:** `fundamentals_data` (recommended)
- **CSV Export:** Merged into `processed.csv`

**Why SQL:**
- Quarterly updates (low data volume)
- Long historical importance
- Easy to query for analysis

**Format (SQL):**
```sql
fundamentals_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    pe_ratio REAL,
    eps REAL,
    revenue REAL,
    market_cap REAL,
    debt_to_equity REAL,
    roe REAL,
    roa REAL,
    UNIQUE(symbol, date)
)
```

---

### 5. Multi-Asset Data

**What:** Correlation features from SPY, QQQ, TLT, GLD  
**Sources:** Yahoo Finance  
**Update Frequency:** Daily

**Storage:**
- **SQL:** Same `market_data` table (different symbols)
- **CSV Export:** `data/training/multi_asset/{SYMBOL}_processed.csv`

**Symbols:**
- **SPY** - S&P 500 (market benchmark)
- **QQQ** - NASDAQ 100 (tech sector)
- **TLT** - 20+ Year Treasury (bonds)
- **GLD** - Gold (safe haven)

**Why Important:**
- Portfolio context
- Risk correlation
- Market regime detection

---

## 🗃️ Database Schema

### SQLite Database: `d:/YoopRL/data/trading.db`

#### Existing Tables:

```sql
-- Portfolio equity tracking (every 5 seconds)
equity_history (
    id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    datetime TEXT NOT NULL,
    net_liquidation REAL NOT NULL,
    buying_power REAL,
    cash REAL,
    unrealized_pnl REAL,
    realized_pnl REAL,
    gross_position_value REAL
)

-- Agent trading decisions
agent_actions (
    id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    datetime TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    reward REAL,
    rationale TEXT,
    confidence REAL,
    state_json TEXT
)

-- Performance statistics
performance_metrics (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    period_type TEXT NOT NULL,
    total_pnl REAL,
    win_rate REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_win REAL,
    avg_loss REAL
)

-- Risk management events
risk_events (
    id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    datetime TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    agent_name TEXT,
    symbol TEXT,
    description TEXT NOT NULL,
    value REAL,
    threshold REAL,
    action_taken TEXT
)

-- System logs
system_logs (
    id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    datetime TEXT NOT NULL,
    component TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT
)
```

#### New Tables (To Be Added):

```sql
-- Historical market data (OHLCV)
market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp REAL NOT NULL,
    datetime TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    adj_close REAL,
    UNIQUE(symbol, timestamp)
)
CREATE INDEX idx_market_symbol_timestamp ON market_data(symbol, timestamp);

-- Sentiment history
sentiment_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    sentiment_mean REAL,
    sentiment_std REAL,
    sentiment_positive_ratio REAL,
    sentiment_negative_ratio REAL,
    sentiment_neutral_ratio REAL,
    sentiment_source_count REAL,
    google_trends REAL,
    UNIQUE(symbol, date)
)
CREATE INDEX idx_sentiment_symbol_date ON sentiment_data(symbol, date);

-- Fundamental data (quarterly)
fundamentals_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    pe_ratio REAL,
    eps REAL,
    revenue REAL,
    market_cap REAL,
    debt_to_equity REAL,
    roe REAL,
    roa REAL,
    UNIQUE(symbol, date)
)
CREATE INDEX idx_fundamentals_symbol_date ON fundamentals_data(symbol, date);
```

---

## 📁 File Structure

```
d:/YoopRL/
├── data/
│   ├── trading.db                          ← Master SQL database
│   ├── cache/                              ← Temporary caches
│   │   └── sentiment/
│   │       ├── IWM.json                    ← 6h TTL cache
│   │       ├── SPY.json
│   │       └── ...
│   └── training/                           ← Training datasets (timestamped)
│       ├── IWM_2025-11-08_14-30-00/       ← Latest download
│       │   ├── raw.csv                     ← OHLCV from SQL
│       │   ├── processed.csv               ← + Technical indicators
│       │   ├── normalized.csv              ← + Normalization
│       │   ├── train.csv                   ← 80% split
│       │   ├── test.csv                    ← 20% split
│       │   └── metadata.json               ← Dataset info
│       ├── IWM_2025-11-07_10-15-00/       ← Previous download (kept for comparison)
│       │   └── ...
│       └── multi_asset/                    ← Multi-asset correlation features
│           ├── SPY_processed.csv
│           ├── QQQ_processed.csv
│           ├── TLT_processed.csv
│           └── GLD_processed.csv
│
├── backend/
│   ├── database/
│   │   ├── db_manager.py                   ← Database operations
│   │   └── ...
│   └── data_download/
│       ├── loader.py                       ← Main download logic
│       ├── multi_asset_loader.py           ← Multi-asset download
│       ├── sentiment_features.py           ← Sentiment aggregation
│       ├── feature_engineering.py          ← Technical indicators
│       └── ...
│
└── frontend/
    └── src/
        └── components/
            └── TabTraining.jsx             ← Download button UI
```

---

## 🔄 Download Flow

### User Action: Click "Download Training Data" Button

```
[User clicks Download Button in Training Tab]
         ↓
[Frontend: TabTraining.jsx]
    POST /api/training/download
    {
        "symbol": "IWM",
        "period": "5y",
        "enable_sentiment": true,
        "enable_multi_asset": true,
        "force_redownload": false
    }
         ↓
[Backend: api/main.py]
    /api/training/download endpoint
         ↓
[Data Download Module: loader.py]
    prepare_training_data()
         ↓
    ┌─────────────────────────────────────────┐
    │ Step 1: Check SQL for existing data     │
    │ - Query: SELECT MAX(datetime)           │
    │   FROM market_data WHERE symbol='IWM'   │
    │ - Result: Last date = 2025-11-07        │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Step 2: Download only missing data      │
    │ - Yahoo Finance: Download from          │
    │   2025-11-07 to today                   │
    │ - Result: 1 day of new bars             │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Step 3: Save to SQL                     │
    │ - INSERT INTO market_data               │
    │ - Merge with existing data              │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Step 4: Download Multi-Asset            │
    │ - SPY, QQQ, TLT, GLD (same logic)       │
    │ - Check SQL → Download new → Save       │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Step 5: Fetch Sentiment                 │
    │ - Check JSON cache (6h TTL)             │
    │ - If expired: Download from APIs        │
    │ - Save to sentiment_data table          │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Step 6: Export from SQL to CSV          │
    │ - Query all data for symbol             │
    │ - Save as raw.csv                       │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Step 7: Feature Engineering             │
    │ - Calculate technical indicators        │
    │ - Merge sentiment features              │
    │ - Save as processed.csv                 │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Step 8: Normalize                       │
    │ - RobustScaler on features              │
    │ - Save as normalized.csv                │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Step 9: Train/Test Split                │
    │ - 80% train, 20% test                   │
    │ - Temporal split (not random!)          │
    │ - Save as train.csv, test.csv           │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │ Step 10: Save Metadata                  │
    │ - Dataset stats                         │
    │ - Feature names                         │
    │ - Date ranges                           │
    │ - Save as metadata.json                 │
    └─────────────────────────────────────────┘
         ↓
[Backend Response]
    {
        "status": "success",
        "symbol": "IWM",
        "rows": 1258,
        "features": 25,
        "train_size": 1006,
        "test_size": 252,
        "files": {
            "raw": "data/training/IWM_2025-11-08_14-30-00/raw.csv",
            "processed": "...",
            "train": "...",
            "test": "..."
        }
    }
         ↓
[Frontend: Display Success]
    "✅ Downloaded 1258 rows with 25 features
     Train: 1006 samples | Test: 252 samples"
```

---

## 🎓 Training Pipeline

### How PPO/SAC Agents Use the Data

```
[Agent Training Starts]
         ↓
[Load Training Data]
    - Read: data/training/IWM_2025-11-08_14-30-00/train.csv
    - Fast: pandas.read_csv() → Direct to memory
    - No SQL queries during training
         ↓
[Initialize Environment]
    - Create TradingEnv with train_df
    - Observation space: 25 features
    - Action space: [0=HOLD, 1=BUY, 2=SELL]
         ↓
[Training Loop]
    - PPO/SAC learns from train.csv
    - No database access
    - Fast iteration
         ↓
[Validation]
    - Load: data/training/IWM_2025-11-08_14-30-00/test.csv
    - Evaluate agent performance
         ↓
[Save Results]
    - Model weights → ppo_runs/models/
    - Training logs → ppo_runs/logs/
    - Metrics → performance_metrics table (SQL)
```

**Why CSV for Training:**
- ⚡ 10-100x faster than SQL queries
- 💾 All data loaded once (not per batch)
- 🔒 No database locks
- 🔄 Reproducible (same file = same results)

---

## 🔁 Incremental Update Logic

### Core Algorithm: Download Only Missing Data

#### Function: `download_history(symbol, period)`

```python
def download_history(symbol, period="5y", force_redownload=False):
    """
    Smart incremental download: Only fetch data we don't have
    
    Logic:
    1. Check SQL for latest date
    2. If no data OR force_redownload:
       - Download full period
    3. If data exists:
       - Calculate days_missing = today - last_date
       - If days_missing == 0: Use existing data
       - If days_missing > 0: Download only new dates
       - Merge old + new data
    4. Save to SQL
    5. Return complete DataFrame
    """
    
    # Step 1: Check SQL for existing data
    db = DatabaseManager()
    last_date = db.get_latest_market_date(symbol)
    
    if last_date is None or force_redownload:
        # No data exists, download full period
        logger.info(f"No data for {symbol}, downloading full {period}")
        df = yf.download(symbol, period=period)
        db.save_market_data(symbol, df)
        return df
    
    # Step 2: Calculate missing days
    today = datetime.now().date()
    days_missing = (today - last_date).days
    
    if days_missing == 0:
        # Data is up to date
        logger.info(f"{symbol} data is current (updated today)")
        return db.get_market_data(symbol)
    
    # Step 3: Download only new data
    logger.info(f"Downloading {days_missing} missing days for {symbol}")
    
    # Download from last_date to today
    new_df = yf.download(
        symbol,
        start=last_date + timedelta(days=1),
        end=today
    )
    
    # Step 4: Merge with existing data
    old_df = db.get_market_data(symbol)
    merged_df = pd.concat([old_df, new_df])
    merged_df = merged_df.sort_index().drop_duplicates()
    
    # Step 5: Save new data to SQL
    db.save_market_data(symbol, new_df)  # Only save new rows
    
    return merged_df
```

### Example Scenarios:

#### Scenario 1: First Download
```
User: Download IWM 5y
SQL: No data for IWM
Action: Download full 5 years (1258 bars)
Result: 1258 rows saved to SQL
```

#### Scenario 2: Up-to-Date
```
User: Download IWM 5y
SQL: Last date = 2025-11-08 (today)
Action: Return existing data from SQL
Result: 0 new rows downloaded
```

#### Scenario 3: Missing 7 Days
```
User: Download IWM 5y
SQL: Last date = 2025-11-01
Today: 2025-11-08
Action: Download only 2025-11-02 to 2025-11-08 (7 days)
Result: Merge 7 new rows with existing 1251 rows = 1258 total
```

#### Scenario 4: Force Re-download
```
User: Download IWM 5y (force_redownload=true)
Action: Download full 5 years regardless of existing data
Result: 1258 rows downloaded, SQL updated
```

---

## 💾 Caching Strategy

### Three Levels of Caching:

#### Level 1: SQL Database (Permanent)
- **What:** OHLCV, Sentiment, Fundamentals
- **Lifetime:** Permanent (with retention policy)
- **Purpose:** Source of truth, historical data

#### Level 2: JSON Cache (Temporary)
- **What:** Sentiment API responses
- **Lifetime:** 6 hours
- **Purpose:** Avoid API rate limits
- **Location:** `data/cache/sentiment/{SYMBOL}.json`

**Cache Logic:**
```python
def get_sentiment(symbol):
    cache_file = f"data/cache/sentiment/{symbol}.json"
    
    # Check cache
    if cache_file.exists():
        data = json.load(cache_file)
        cached_time = datetime.fromisoformat(data['timestamp'])
        age = datetime.now() - cached_time
        
        if age < timedelta(hours=6):
            # Cache is fresh
            return data['features']
    
    # Cache expired or missing, fetch new data
    sentiment = fetch_sentiment_from_apis(symbol)
    
    # Save to cache
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'features': sentiment
    }, cache_file)
    
    # Also save to SQL for historical record
    db.save_sentiment_data(symbol, sentiment)
    
    return sentiment
```

#### Level 3: CSV Training Files (Semi-Permanent)
- **What:** Exported training datasets
- **Lifetime:** Keep last 10 downloads per symbol
- **Purpose:** Reproducibility, debugging
- **Location:** `data/training/{SYMBOL}_{TIMESTAMP}/`

**Cleanup Logic:**
```python
def cleanup_old_training_files(symbol, keep_latest=10):
    """
    Keep only the most recent 10 training datasets per symbol
    
    Why keep old datasets:
    - Compare model performance across different data versions
    - Debug issues with specific training runs
    - Reproduce results from previous training
    
    Why delete old datasets:
    - Save disk space (each dataset ~50-100MB)
    - Avoid confusion with outdated data
    """
    training_dir = Path(f"data/training")
    pattern = f"{symbol}_*"
    
    # Find all directories for this symbol
    dirs = sorted(training_dir.glob(pattern), reverse=True)
    
    # Keep first N, delete rest
    for old_dir in dirs[keep_latest:]:
        shutil.rmtree(old_dir)
        logger.info(f"Deleted old training data: {old_dir}")
```

---

## 🧹 Cleanup & Retention

### Database Retention Policy

**Current Settings:**
- **Retention Period:** 365 days (1 year)
- **Auto-Cleanup:** On database initialization
- **Manual Trigger:** `db.cleanup_old_data()`

**What Gets Cleaned:**
- ✅ `equity_history` - Old equity points
- ✅ `agent_actions` - Old trading decisions
- ✅ `system_logs` - Old log entries
- ❌ `market_data` - **KEEP** (training needs historical data)
- ❌ `sentiment_data` - **KEEP** (sentiment trends important)
- ❌ `performance_metrics` - **KEEP** (aggregated, small size)
- ❌ `risk_events` - **KEEP** (compliance, small size)

**Special Handling for Market Data:**
```sql
-- Market data is NEVER auto-deleted
-- Reasoning:
-- 1. Training needs years of historical data
-- 2. Data volume is manageable (~1MB per symbol per year)
-- 3. Old data is valuable for backtesting

-- Manual cleanup if needed:
DELETE FROM market_data 
WHERE datetime < '2020-01-01' AND symbol = 'OLD_SYMBOL';
```

### File System Cleanup

**CSV Training Files:**
- Keep: 10 most recent per symbol
- Delete: Older than 10 runs
- Size: ~50-100MB per dataset
- Total: Max ~1GB per symbol

**JSON Sentiment Cache:**
- TTL: 6 hours
- Auto-delete: On next access if expired
- Size: ~10KB per symbol
- Total: Negligible

---

## 🔍 Monitoring & Debugging

### Check What's in SQL

```python
# Get database statistics
db = DatabaseManager()
stats = db.get_database_stats()

print(f"Market data rows: {stats['market_data_count']}")
print(f"Sentiment data rows: {stats['sentiment_data_count']}")
print(f"Oldest market data: {stats['oldest_market_date']}")
print(f"Newest market data: {stats['newest_market_date']}")
print(f"Database size: {stats['db_size_mb']:.2f} MB")
```

### Check Latest Date for Symbol

```python
db = DatabaseManager()
last_date = db.get_latest_market_date('IWM')
print(f"IWM last update: {last_date}")

# Check if up to date
today = datetime.now().date()
days_behind = (today - last_date).days
print(f"Days behind: {days_behind}")
```

### View Training Files

```bash
# List all training datasets for IWM
ls -lh data/training/IWM_*

# Check latest dataset
cd data/training/IWM_2025-11-08_14-30-00/
head -5 raw.csv
head -5 processed.csv
cat metadata.json
```

---

## 🚨 Error Handling

### Download Failures

**Scenario:** Yahoo Finance API fails

```python
try:
    df = yf.download(symbol, period=period)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")
except Exception as e:
    logger.error(f"Download failed: {e}")
    
    # Fallback: Use existing SQL data
    df = db.get_market_data(symbol)
    
    if df.empty:
        raise RuntimeError(f"No data available for {symbol}")
    
    logger.warning(f"Using cached SQL data for {symbol}")
    return df
```

### Sentiment API Rate Limits

**Scenario:** Hit API limits (429 Too Many Requests)

```python
try:
    sentiment = fetch_alpha_vantage_sentiment(symbol)
except APIRateLimitError:
    logger.warning("Alpha Vantage rate limit hit")
    
    # Fallback: Use yesterday's sentiment from SQL
    sentiment = db.get_sentiment_data(
        symbol, 
        date=(datetime.now() - timedelta(days=1)).date()
    )
    
    if sentiment is None:
        # Ultimate fallback: Neutral sentiment
        sentiment = {
            'sentiment_mean': 0.0,
            'sentiment_std': 0.0,
            'sentiment_positive_ratio': 0.5,
            'sentiment_negative_ratio': 0.5,
            'sentiment_neutral_ratio': 0.0,
            'sentiment_source_count': 0.0,
            'google_trends': 0.0
        }
```

### Database Lock

**Scenario:** Multiple processes accessing database

```python
# Use WAL mode for better concurrency
conn.execute("PRAGMA journal_mode=WAL")

# Retry logic for busy database
max_retries = 3
for attempt in range(max_retries):
    try:
        cursor.execute("INSERT INTO market_data ...")
        conn.commit()
        break
    except sqlite3.OperationalError as e:
        if "locked" in str(e) and attempt < max_retries - 1:
            time.sleep(0.1)  # Wait 100ms
            continue
        raise
```

---

## 📈 Performance Considerations

### SQL Queries

**Optimized:**
```sql
-- Use indexed columns
SELECT * FROM market_data 
WHERE symbol = 'IWM' AND timestamp >= ?
ORDER BY timestamp ASC;

-- Batch inserts
INSERT INTO market_data (symbol, timestamp, ...) 
VALUES (?, ?, ...), (?, ?, ...), ...  -- Multiple rows at once
```

**Avoid:**
```sql
-- Full table scan (slow)
SELECT * FROM market_data WHERE datetime LIKE '2025%';

-- Row-by-row inserts (slow)
for row in df:
    cursor.execute("INSERT INTO market_data VALUES (?)", row)
```

### CSV Exports

**Fast:**
```python
# Use pandas native methods
df.to_csv('data/training/IWM/raw.csv', index=True)
```

**Avoid:**
```python
# Row-by-row writing (1000x slower)
with open('file.csv', 'w') as f:
    for row in df.iterrows():
        f.write(...)
```

### Memory Management

**Large Datasets:**
```python
# Stream from SQL in chunks
chunksize = 10000
for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
    process(chunk)
```

**Full Load:**
```python
# Load entire dataset at once (for training)
df = pd.read_csv('train.csv')  # Fast, entire file → RAM
```

---

## 🎯 Summary

### Key Design Decisions:

1. **Hybrid Storage**
   - SQL = Master (incremental, centralized)
   - CSV = Training (fast, reproducible)

2. **Incremental Downloads**
   - Check SQL for last date
   - Download only missing data
   - Merge and save

3. **Data Types**
   - OHLCV → SQL + CSV
   - Technical Indicators → CSV only (computed)
   - Sentiment → SQL + JSON cache + CSV
   - Fundamentals → SQL + CSV (future)

4. **File Structure**
   - Timestamped folders per download
   - Keep 10 most recent
   - Metadata for reproducibility

5. **Performance**
   - SQL for data management
   - CSV for training speed
   - Best of both worlds

---

## 📚 References

- `backend/database/db_manager.py` - Database operations
- `backend/data_download/loader.py` - Download logic
- `backend/data_download/multi_asset_loader.py` - Multi-asset handling
- `backend/data_download/sentiment_features.py` - Sentiment aggregation
- `backend/data_download/feature_engineering.py` - Technical indicators
- `frontend/src/components/TabTraining.jsx` - UI download button

---

**Document Status:** ✅ Complete  
**Last Updated:** November 8, 2025  
**Next Review:** When implementing fundamentals or adding new data sources
