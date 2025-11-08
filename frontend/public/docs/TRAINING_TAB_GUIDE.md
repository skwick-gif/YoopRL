# 📚 Training Tab - Complete User Guide

**Version:** 1.0  
**Last Updated:** November 8, 2025  
**System:** YoopRL Trading System

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [User Flow - Step by Step](#user-flow---step-by-step)
3. [Agent Selection](#agent-selection)
4. [Model Selector](#model-selector)
5. [Hyperparameter Configuration](#hyperparameter-configuration)
6. [Feature Selection](#feature-selection)
7. [Configuration Manager](#configuration-manager)
8. [Training Controls](#training-controls)
9. [Training Progress](#training-progress)
10. [Backtest Results](#backtest-results)
11. [Drift Detection](#drift-detection)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The **Training Tab** is where you create, train, and optimize Reinforcement Learning (RL) agents for automated trading. This is the heart of the YoopRL system - where AI learns to trade by analyzing historical market data.

### What Does This Tab Do?

1. **Downloads** historical market data (prices, volumes, indicators)
2. **Configures** agent hyperparameters (learning rate, batch size, etc.)
3. **Selects** features (RSI, MACD, sentiment, multi-asset correlation)
4. **Trains** RL agents using Optuna hyperparameter optimization
5. **Validates** performance on test data
6. **Saves** trained models for live trading

### Two Agent Types

| Agent | Best For | Risk Level | Characteristics |
|-------|----------|------------|-----------------|
| **PPO** (Proximal Policy Optimization) | Stocks (AAPL, MSFT, etc.) | Low-Medium | Stable, conservative, good for long-term |
| **SAC** (Soft Actor-Critic) | Leveraged ETFs (TQQQ, SOXL, TNA) | Medium-High | Aggressive, handles volatility well |

---

## 🔄 User Flow - Step by Step

### Phase 1: Setup (5 minutes)

```
┌─────────────────────────────────────────┐
│ 1. Select Agent Type (PPO or SAC)      │
│    ↓                                    │
│ 2. Configure Symbol & Date Range       │
│    ↓                                    │
│ 3. Adjust Hyperparameters (or use     │
│    Quick Presets)                       │
│    ↓                                    │
│ 4. Select Features (Technical          │
│    Indicators, Sentiment, Multi-Asset)  │
└─────────────────────────────────────────┘
```

### Phase 2: Data Preparation (30 seconds - 2 minutes)

```
┌─────────────────────────────────────────┐
│ 5. Click "Download Training Data"      │
│    ↓                                    │
│    - System downloads OHLCV from Yahoo  │
│    - Calculates technical indicators    │
│    - Fetches sentiment data (if enabled)│
│    - Downloads multi-asset correlations │
│    - Saves everything to SQL database   │
│    ↓                                    │
│ 6. Wait for "✅ Training Data Ready!"  │
└─────────────────────────────────────────┘
```

### Phase 3: Training (30 minutes - 6 hours)

```
┌─────────────────────────────────────────┐
│ 7. Click "Start Training"              │
│    ↓                                    │
│    - Optuna runs 100 trials (default)   │
│    - Each trial tests different         │
│      hyperparameters                    │
│    - Agent learns from 50,000 episodes  │
│    ↓                                    │
│ 8. Monitor Progress                    │
│    - Episode count                      │
│    - Average reward                     │
│    - Loss values                        │
│    ↓                                    │
│ 9. Wait for "Training Complete!"       │
└─────────────────────────────────────────┘
```

### Phase 4: Validation & Deployment (5 minutes)

```
┌─────────────────────────────────────────┐
│ 10. Review Backtest Results            │
│     - Total P&L                         │
│     - Win Rate                          │
│     - Sharpe Ratio                      │
│     - Max Drawdown                      │
│     ↓                                   │
│ 11. If satisfied → Deploy to Live       │
│     If not → Adjust hyperparameters and │
│              retrain                    │
└─────────────────────────────────────────┘
```

---

## 🤖 Agent Selection

**Location:** Top of the page (first section)

### PPO - Proximal Policy Optimization

**Use When:**
- Trading **stocks** (AAPL, GOOGL, MSFT, TSLA, etc.)
- You want **stable, predictable** behavior
- Risk tolerance is **low to medium**
- Holding periods are **days to weeks**

**Configuration:**
- **Symbol:** Stock ticker (e.g., AAPL)
- **Learning Rate:** 0.0003 (default) - lower = slower learning, more stable
- **Gamma:** 0.99 (default) - how much agent values future rewards
- **Batch Size:** 256 (default) - samples per training update
- **Risk Penalty:** -0.5 (default) - penalizes risky trades
- **Episodes:** 50,000 (default) - total training iterations
- **Retrain:** Weekly (recommended)

**Quick Presets:**
1. **Conservative (Retirement Portfolio)**
   - Risk Penalty: -1.0
   - Episodes: 60,000
   - Focus: Minimize losses, slow steady growth

2. **Balanced (Growth Portfolio)**
   - Default settings
   - Focus: Balance risk and reward

3. **Aggressive (Day Trading)**
   - Risk Penalty: -0.2
   - Episodes: 40,000
   - Focus: Maximize returns, accept higher risk

---

### SAC - Soft Actor-Critic

**Use When:**
- Trading **leveraged ETFs** (TQQQ, SOXL, TNA, UPRO, etc.)
- You want **aggressive, high-volatility** trading
- Risk tolerance is **medium to high**
- Holding periods are **hours to days**

**Configuration:**
- **Symbol:** ETF ticker (e.g., TQQQ - 3x NASDAQ)
- **Learning Rate:** 0.0003 (default)
- **Entropy Coefficient:** 0.2 (default) - encourages exploration
- **Batch Size:** 256 (default)
- **Volatility Penalty:** -0.3 (default) - penalizes excessive volatility
- **Episodes:** 45,000 (default)
- **Retrain:** Weekly (recommended)

**Quick Presets:**
1. **Conservative (2x Leverage)**
   - Volatility Penalty: -0.5
   - Symbol: SSO (2x S&P 500)
   - Focus: Controlled leverage

2. **Balanced (3x Tech)**
   - Default settings
   - Symbol: TQQQ
   - Focus: Tech sector growth

3. **Aggressive (3x Small Cap)**
   - Volatility Penalty: -0.1
   - Symbol: TNA (3x Russell 2000)
   - Focus: Maximum growth potential

---

## 🎛️ Model Selector

**Purpose:** Choose which trained model to use or continue training

### Current Models Display

Shows active models with metadata:
```
PPO: v3.2_20241105 | Trained: 05/11/2024 | Episodes: 50,000
SAC: v2.8_20241105 | Trained: 05/11/2024 | Episodes: 45,000
```

### Model Actions

| Button | Action | When to Use |
|--------|--------|-------------|
| **Load Model** | Load previously saved model | Continue training or deploy to live |
| **New Model** | Start fresh training | Current model underperforming |
| **Compare Models** | Side-by-side performance | Choosing best version |

---

## ⚙️ Hyperparameter Configuration

**Purpose:** Fine-tune how the agent learns

### Training Settings (Universal)

#### Start Date & End Date
- **Default:** 2020-01-01 to Today
- **Minimum:** 2 years of data (730 days)
- **Recommended:** 4-5 years for best generalization
- **Why:** More data = better learning, but slower training

**Example:**
```
Start Date: 2020-01-01  (5 years ago)
End Date:   2025-11-08  (today)
Result:     ~1,250 trading days
```

#### Commission
- **Default:** $1.00 per trade
- **Range:** $0 - $10
- **Why:** Realistic commission costs improve real-world performance
- **Note:** Interactive Brokers charges ~$1-5 per trade

#### Optuna Trials
- **Default:** 100 trials
- **Range:** 10 - 500
- **What it does:** Tests different hyperparameter combinations
- **Time impact:** More trials = better optimization but longer training
- **Recommended:**
  - Quick test: 10 trials (~10 minutes)
  - Standard: 100 trials (~2 hours)
  - Production: 250+ trials (~6 hours)

---

### PPO Hyperparameters

#### Learning Rate
- **Default:** 0.0003
- **Range:** 0.00001 - 0.01
- **What it does:** How fast the agent learns
- **Lower (0.0001):** Slower, more stable learning
- **Higher (0.001):** Faster, but may be unstable

#### Gamma (Discount Factor)
- **Default:** 0.99
- **Range:** 0.9 - 0.999
- **What it does:** How much agent values future rewards
- **0.9:** Short-term focused (day trading)
- **0.99:** Balanced (swing trading)
- **0.999:** Long-term focused (position trading)

#### Batch Size
- **Default:** 256
- **Options:** 64, 128, 256, 512
- **What it does:** Samples processed per update
- **Smaller (64):** Less memory, faster iterations, noisier
- **Larger (512):** More memory, slower iterations, smoother

#### Risk Penalty
- **Default:** -0.5
- **Range:** 0 to -2.0
- **What it does:** Penalizes risky behavior
- **0:** No penalty (very aggressive)
- **-0.5:** Moderate caution
- **-2.0:** Extremely conservative

---

### SAC Hyperparameters

#### Entropy Coefficient
- **Default:** 0.2
- **Range:** 0.01 - 0.5
- **What it does:** Encourages exploration vs exploitation
- **Lower (0.05):** Exploit known strategies (conservative)
- **Higher (0.4):** Explore new strategies (experimental)

#### Volatility Penalty
- **Default:** -0.3
- **Range:** 0 to -1.0
- **What it does:** Penalizes excessive volatility
- **0:** No penalty (wild swings OK)
- **-0.3:** Moderate smoothness
- **-1.0:** Very smooth, low volatility

---

## 🔧 Feature Selection

**Purpose:** Choose which data the agent sees and learns from

### Price Data (Always Enabled)

| Feature | Description | Why Important |
|---------|-------------|---------------|
| **Price** | Current closing price | Core data - always required |
| **Volume** | Trading volume | Indicates liquidity and interest |
| **OHLC** | Open, High, Low, Close | Shows full price range per period |

---

### Technical Indicators (Optional)

#### RSI (Relative Strength Index)
- **What:** Measures if stock is overbought or oversold
- **Range:** 0-100
- **Signals:**
  - RSI > 70: Overbought (might go down)
  - RSI < 30: Oversold (might go up)
- **Period:** 14 days (standard)
- **Use When:** Stock oscillates in range

#### MACD (Moving Average Convergence Divergence)
- **What:** Shows trend direction and momentum
- **Components:**
  - MACD Line: Fast EMA - Slow EMA
  - Signal Line: 9-day EMA of MACD
  - Histogram: MACD - Signal
- **Signals:**
  - MACD crosses above signal: Buy
  - MACD crosses below signal: Sell
- **Parameters:** 12, 26, 9 (standard)
- **Use When:** Trending markets

#### EMA (Exponential Moving Average)
- **What:** Smoothed average price, recent data weighted more
- **Periods:** 10 (short-term), 50 (long-term)
- **Signals:**
  - EMA 10 > EMA 50: Uptrend
  - EMA 10 < EMA 50: Downtrend
- **Use When:** Identifying support/resistance

#### Bollinger Bands
- **What:** Volatility bands around price
- **Components:**
  - Upper Band: SMA + (2 × StdDev)
  - Middle Band: 20-day SMA
  - Lower Band: SMA - (2 × StdDev)
  - Width: (Upper - Lower) / Middle
- **Signals:**
  - Price at upper band: Overbought
  - Price at lower band: Oversold
  - Narrow bands: Low volatility (breakout coming)
  - Wide bands: High volatility
- **Parameters:** 20 days, 2 std dev (standard)

#### Stochastic Oscillator
- **What:** Compares closing price to price range
- **Range:** 0-100
- **Components:**
  - %K: Fast line (14 periods)
  - %D: Slow line (3-period SMA of %K)
- **Signals:**
  - > 80: Overbought
  - < 20: Oversold
  - %K crosses above %D: Buy
  - %K crosses below %D: Sell
- **Use When:** Ranging markets

#### VIX (Volatility Index)
- **What:** Market fear gauge
- **Range:** 10-80+
- **Signals:**
  - VIX < 15: Low volatility (complacent market)
  - VIX 15-25: Normal volatility
  - VIX > 25: High volatility (fearful market)
- **Use When:** Adjusting risk during volatile periods
- **Note:** Requires separate data download

---

### Alternative Data (Advanced)

#### Sentiment (News)
- **What:** AI analysis of news articles
- **Sources:** Alpha Vantage, Finnhub, NewsAPI
- **Range:** -1 (very negative) to +1 (very positive)
- **Features:**
  - `sentiment_mean`: Average sentiment score
  - `sentiment_positive_ratio`: % positive articles
  - `sentiment_negative_ratio`: % negative articles
- **API Keys Required:** Yes (see .env file)
- **Use When:** Trading news-sensitive stocks (TSLA, AAPL)

#### Social Media
- **What:** Sentiment from Twitter, Reddit, StockTwits
- **Sources:** Reddit r/wallstreetbets, StockTwits
- **Features:**
  - `google_trends`: Search interest score
  - Social mentions count
- **Use When:** Trading meme stocks or retail-driven names

#### News Headlines
- **What:** Breaking news count and analysis
- **Features:**
  - `newsapi_sentiment`: NewsAPI sentiment
  - `newsapi_articles`: Article count
- **Use When:** Earnings season, major events

#### Market Events
- **What:** Earnings, dividends, splits, etc.
- **Status:** 🚧 Not implemented yet (Phase 2)
- **Planned Features:**
  - Days until earnings
  - Dividend yield changes
  - Stock split indicators

#### Fundamental Data
- **What:** Company financials (P/E, EPS, revenue)
- **Status:** 🚧 Not implemented yet (Phase 2)
- **Planned Features:**
  - P/E ratio trends
  - EPS growth
  - Revenue growth
  - Debt-to-equity

#### Multi-Asset Correlation
- **What:** Correlation with market indices
- **Symbols:** SPY, QQQ, TLT, GLD (customizable)
- **Features (per symbol):**
  - `spy_close_ratio`: Price ratio
  - `spy_returns`: Daily returns
  - `spy_corr_20`: 20-day rolling correlation
- **Use When:** Understanding market context
- **Why Important:**
  - **SPY** (S&P 500): Overall market sentiment
  - **QQQ** (NASDAQ): Tech sector health
  - **TLT** (Bonds): Risk-off indicator
  - **GLD** (Gold): Safe haven indicator

**Example:** If IWM (small caps) is dropping but SPY is flat, agent learns this is stock-specific weakness.

---

### Agent History (Self-Awareness)

#### Recent Actions
- **What:** Last 5 buy/sell/hold decisions
- **Why:** Agent learns from its own trading patterns
- **Status:** ✅ Implemented

#### Performance History
- **What:** Past P&L over time window
- **Period:** 30 days (default)
- **Status:** ✅ Implemented

#### Position History
- **What:** Entry/exit prices, holding periods
- **Status:** 🚧 Phase 2

#### Reward History
- **What:** Last 100 rewards, statistics
- **Status:** 🚧 Phase 2

---

### LLM Integration (Experimental)

#### Perplexity API
- **What:** Natural language market analysis
- **Use Case:** Generate trading rationale
- **Status:** 🚧 Not implemented

#### Gemini API
- **What:** Multi-modal market analysis
- **Status:** 🚧 Not implemented

---

## 💾 Configuration Manager

**Purpose:** Save and load hyperparameter configurations

### Quick Presets

**PPO Presets (for Stocks):**
1. **Conservative:** High risk penalty, long episodes, stable
2. **Aggressive:** Low risk penalty, short episodes, risky
3. **Balanced:** Default settings, best starting point

**SAC Presets (for ETFs):**
1. **Conservative:** High volatility penalty, 2x ETFs
2. **Aggressive:** Low volatility penalty, 3x ETFs
3. **Balanced:** Default settings, 3x tech

### Save Configuration
- **Button:** "Save Config"
- **Format:** JSON file with all hyperparameters
- **Location:** `configs/[agent]_[symbol]_[date].json`
- **Use Case:** Save successful configurations for future use

### Load Configuration
- **Button:** "Load Config"
- **Effect:** Restores all hyperparameters from file
- **Use Case:** Replicate successful training runs

---

## 🎮 Training Controls

**Location:** Action Bar (below Configuration Manager)

### Download Training Data

**Button:** Blue "Download Training Data"

**What it does:**
1. Downloads OHLCV data from Yahoo Finance
2. Calculates selected technical indicators
3. Fetches sentiment data (if enabled)
4. Downloads multi-asset correlations (if enabled)
5. Saves everything to SQL database (`trading.db`)
6. Exports to CSV files for training

**Time Required:**
- Basic (OHLCV + indicators): 10-30 seconds
- With sentiment: 30-60 seconds
- With multi-asset: 30-90 seconds

**Status Messages:**
```
⬇️ Downloading Data... 50%
✅ Training Data Ready! Symbol: IWM, Rows: 1257, 
   Features: 48, Train: 966 / Test: 242
```

**Troubleshooting:**
- **Timeout:** Network issue, retry
- **Symbol not found:** Check ticker symbol
- **No data:** Symbol may be delisted

---

### Start Training

**Button:** Green "Start Training" (enabled after download)

**What it does:**
1. Validates configuration
2. Builds training environment
3. Runs Optuna hyperparameter optimization
4. Trains agent for specified episodes
5. Saves best model to disk

**Time Required:**
- Quick test (10 trials, 10k episodes): 10-30 minutes
- Standard (100 trials, 50k episodes): 2-4 hours
- Production (250 trials, 100k episodes): 6-12 hours

**Progress Display:**
```
Episode: 12,450 / 50,000 (24.9%)
Average Reward: 0.0234
Policy Loss: 0.0012
Value Loss: 0.0045
Trial: 47 / 100 (Best: 0.0567)
```

**When to Stop:**
- Average reward plateaus for 5,000+ episodes
- Best trial hasn't improved in 20+ trials
- Loss values < 0.001 (converged)

---

### Stop Training

**Button:** Red "Stop Training" (enabled during training)

**What it does:**
- Gracefully stops current training
- Saves partial progress
- Returns best trial so far

**Use When:**
- Reward is decreasing (overfitting)
- Time limit reached
- Emergency stop needed

---

### Load Model

**Button:** "Load Model"

**What it does:**
- Opens file picker
- Loads saved model weights
- Displays model metadata

**Use When:**
- Continuing previous training
- Deploying to live trading
- Comparing model versions

---

## 📊 Training Progress

**Location:** Card below Training Controls

### Real-Time Metrics

#### Download Progress
```
⬇️ Downloading Data... 75%
[████████████░░░░] 75%
```

#### Training Progress
```
🤖 Training in Progress...
Episode: 25,000 / 50,000 (50%)
[██████████░░░░░░░░░░] 50%

Average Reward: 0.0456
Policy Loss: 0.0023
Value Loss: 0.0078

Trial: 67 / 100
Best Trial: #54 (Reward: 0.0789)
```

#### Completion
```
✅ Training Complete!
Total Time: 2h 34m
Best Trial: #54
Best Reward: 0.0789
Model saved: ppo_runs/models/IWM_20241108_143022.zip
```

---

### Training Logs

Real-time log display:
```
[14:30:22] Starting Optuna optimization...
[14:30:23] Trial 1/100: Testing learning_rate=0.0005...
[14:32:15] Trial 1 complete: reward=0.0234
[14:32:16] Trial 2/100: Testing learning_rate=0.0002...
[14:34:08] Trial 2 complete: reward=0.0456 (NEW BEST!)
...
```

---

## 📈 Backtest Results

**Purpose:** Validate agent performance on test data

### Performance Metrics

#### Total P&L
- **What:** Total profit/loss in dollars
- **Good:** Positive P&L
- **Target:** > 10% of starting capital

#### Win Rate
- **What:** Percentage of profitable trades
- **Range:** 0% - 100%
- **Good:** > 50%
- **Excellent:** > 60%

#### Sharpe Ratio
- **What:** Risk-adjusted return
- **Formula:** (Return - RiskFreeRate) / StdDev
- **Interpretation:**
  - < 1: Poor (high risk, low return)
  - 1-2: Good (acceptable risk/return)
  - 2-3: Very good
  - \> 3: Excellent (rare)

#### Sortino Ratio
- **What:** Like Sharpe, but only penalizes downside volatility
- **Better than Sharpe:** Ignores upside volatility
- **Interpretation:** Similar to Sharpe ratio

#### Max Drawdown
- **What:** Largest peak-to-trough decline
- **Good:** < 15%
- **Acceptable:** 15-25%
- **Concerning:** > 25%

#### Total Trades
- **What:** Number of buy/sell actions
- **Too few (< 10):** Agent not active enough
- **Too many (> 500):** Overtrading (commission issues)
- **Good:** 50-200 trades per year

---

### Equity Curve

**What:** Portfolio value over time

**Ideal Shape:**
```
$
│     ╱╲    ╱╲╱
│   ╱    ╲╱    
│  ╱           
│ ╱            
└──────────────> Time
```
- Upward trend
- Smooth growth
- Small drawdowns

**Warning Signs:**
```
$
│╲    ╱╲      
│ ╲  ╱  ╲  ╱╲ 
│  ╲╱    ╲╱  ╲
│            
└──────────────> Time
```
- Volatile
- Large swings
- Downward trend

---

## 🚨 Drift Detection

**Purpose:** Alert when market conditions change significantly

### What is Data Drift?

**Concept:** Market behavior changes over time (regime shift)

**Examples:**
- Bull market → Bear market
- Low volatility → High volatility
- Interest rate changes
- Major economic events

**Why It Matters:** Model trained on old data may underperform in new conditions

---

### Drift Alert Display

```
⚠️ Data Drift Detected!
Symbol: AAPL
Drift Score: 0.78 (High)
Recommendation: Retrain model with recent data

Last Training: 30 days ago
Market Regime Changed: Yes

[Retrain Now] [View Details]
```

---

### When to Retrain

**Automatic Triggers:**
- Drift score > 0.5 (moderate drift)
- Drift score > 0.75 (severe drift)
- 30+ days since last training

**Manual Triggers:**
- Major market event (crash, rally)
- Model performance degrading
- New trading strategy desired

**Best Practice:** Retrain weekly or after major events

---

## ✅ Best Practices

### 1. Start Simple

**First Training:**
- Use **PPO** for stocks
- Select **AAPL** or **MSFT** (liquid, stable)
- Enable only **RSI + MACD**
- Use **Balanced preset**
- Train with **10 Optuna trials** (quick test)

**Why:** Learn the system before complexity

---

### 2. Data Quality

**Good Data:**
- ✅ 3-5 years of history
- ✅ Liquid stocks (volume > 1M/day)
- ✅ No major corporate actions (mergers, delistings)

**Bad Data:**
- ❌ IPOs (< 2 years history)
- ❌ Penny stocks (< $5)
- ❌ Low volume (< 100K/day)

---

### 3. Feature Selection

**Beginner:**
- Price + Volume + RSI + MACD
- Total: ~10-15 features

**Intermediate:**
- Above + EMA + Bollinger
- Total: ~20-25 features

**Advanced:**
- Above + Sentiment + Multi-Asset
- Total: ~40-50 features

**Warning:** More features ≠ better performance
- More features = longer training
- Overfitting risk increases
- Start simple, add gradually

---

### 4. Hyperparameter Tuning

**First Attempt:** Use Quick Presets

**If Underperforming:**
1. Increase Optuna trials (100 → 250)
2. Adjust risk/volatility penalty
3. Try different learning rate
4. Increase episodes

**If Overfitting:** (great backtest, poor live)
1. Increase risk penalty
2. Reduce episodes
3. Add regularization (future feature)

---

### 5. Training Time

**Time Budget:**
- **1 hour:** Quick test (10 trials, 10k episodes)
- **Half day:** Standard (50 trials, 25k episodes)
- **Full day:** Production (100 trials, 50k episodes)
- **Weekend:** Extensive (250 trials, 100k episodes)

**Tip:** Train overnight or on weekends

---

### 6. Model Validation

**Before Live Trading:**
1. ✅ Backtest Sharpe Ratio > 1.5
2. ✅ Win Rate > 50%
3. ✅ Max Drawdown < 20%
4. ✅ Consistent positive returns
5. ✅ Paper trade for 1 week

**If Any Metric Fails:** Retrain with adjustments

---

### 7. Retraining Schedule

**Frequency:**
- **Stable markets:** Every 2 weeks
- **Volatile markets:** Weekly
- **After major events:** Immediately

**Why:** Keep model adapted to current conditions

---

## 🔧 Troubleshooting

### Download Issues

#### Problem: "Download Failed - Timeout"
**Cause:** Network connection issue or Yahoo Finance API slow  
**Solution:**
1. Check internet connection
2. Retry download
3. Try different symbol
4. Wait 5 minutes and retry

---

#### Problem: "Symbol not found"
**Cause:** Invalid ticker symbol  
**Solution:**
1. Verify symbol on Yahoo Finance website
2. Check for typos (APPL ❌ → AAPL ✅)
3. Try adding exchange suffix (e.g., `AAPL.US`)

---

#### Problem: "Insufficient data"
**Cause:** Symbol doesn't have enough history  
**Solution:**
1. Choose older symbol (> 3 years history)
2. Reduce date range requirements
3. Check if symbol was recently listed

---

### Training Issues

#### Problem: Training stuck at 0%
**Cause:** Environment initialization error  
**Solution:**
1. Check browser console (F12) for errors
2. Restart backend: `python api\main.py`
3. Clear browser cache
4. Redownload training data

---

#### Problem: "Average reward is negative"
**Cause:** Model not learning properly  
**Solution:**
1. Reduce learning rate (0.0003 → 0.0001)
2. Increase episodes (50k → 100k)
3. Check feature selection (may have too many)
4. Try different symbol
5. Increase Optuna trials

---

#### Problem: Training very slow
**Cause:** Too many features or large dataset  
**Solution:**
1. Reduce features (disable sentiment/multi-asset)
2. Reduce date range (5y → 3y)
3. Reduce episodes (50k → 30k)
4. Close other programs (RAM issue)

---

#### Problem: High reward in training, poor in backtest
**Cause:** Overfitting  
**Solution:**
1. Increase risk penalty
2. Reduce episodes
3. Use more conservative hyperparameters
4. Add regularization
5. Get more training data

---

### Performance Issues

#### Problem: Low Sharpe Ratio (< 1.0)
**Cause:** Returns not worth the risk  
**Solution:**
1. Increase risk penalty
2. Add Bollinger Bands for volatility awareness
3. Train longer (more episodes)
4. Try different hyperparameters

---

#### Problem: Low Win Rate (< 45%)
**Cause:** Agent making poor decisions  
**Solution:**
1. Add more features (sentiment, multi-asset)
2. Increase Optuna trials
3. Check if symbol is trending or ranging
4. Try different agent type (PPO ↔ SAC)

---

#### Problem: High Max Drawdown (> 25%)
**Cause:** Agent taking too much risk  
**Solution:**
1. **Increase risk/volatility penalty significantly**
2. Add stop-loss logic (future feature)
3. Reduce position sizing
4. Train on more stable symbol

---

## 📞 Getting Help

### Debug Mode

**Enable in config:**
```javascript
// frontend/src/config.js
export const DEBUG_MODE = true;
```

**Shows:**
- API request/response details
- Training metrics in console
- Feature values during training

---

### Log Files

**Backend Logs:**
```
backend/logs/training_[date].log
backend/logs/optuna_[date].log
```

**Frontend Console:**
- Press `F12` in browser
- Go to "Console" tab
- Look for errors (red text)

---

### Community Support

- **GitHub Issues:** Report bugs
- **Discord:** Real-time help
- **Documentation:** This guide + API docs

---

## 📚 Additional Resources

### Related Documentation

- [RL System Specification](../RL_System_Specification.md) - Technical architecture
- [Data Storage Architecture](../backend/data_download/DATA_STORAGE_ARCHITECTURE.md) - How data is stored
- [API Documentation](../backend/api/README.md) - Backend API reference

### External Resources

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/) - RL library
- [Optuna Documentation](https://optuna.readthedocs.io/) - Hyperparameter optimization
- [Yahoo Finance](https://finance.yahoo.com/) - Market data source

---

## 🎓 Glossary

| Term | Definition |
|------|------------|
| **Agent** | AI model that learns to trade |
| **Episode** | One complete simulation from start to finish |
| **Epoch** | One pass through entire training dataset |
| **Trial** | One hyperparameter combination tested by Optuna |
| **Reward** | Numerical score for agent's performance |
| **Policy** | Agent's decision-making strategy |
| **Environment** | Simulated trading world where agent operates |
| **Observation** | Market state that agent sees (prices, indicators) |
| **Action** | Agent's decision (buy, sell, hold) |
| **Hyperparameter** | Configuration value that affects learning |
| **Overfitting** | Model performs well on training but poorly on new data |
| **Drift** | Change in market behavior over time |
| **Sharpe Ratio** | Risk-adjusted return metric |
| **Drawdown** | Peak-to-trough decline in portfolio value |

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 8, 2025 | Initial release - Complete training tab documentation |

---

**Last Updated:** November 8, 2025  
**Author:** YoopRL Development Team  
**License:** Internal Documentation

---

