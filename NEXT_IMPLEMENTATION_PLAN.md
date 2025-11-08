# 🎯 Next Implementation Phase: Backtesting & Evaluation

## Status: After Training Pipeline is Complete

### What's Working Now ✅
- ✅ Data Download (OHLCV, Fundamentals, Events, Macro) - 70+ features
- ✅ Feature Engineering with selective computation
- ✅ SQL caching with 24-hour TTL
- ✅ Training UI (Feature Selection, Hyperparameters)
- ✅ RL Environments (StockEnv, ETFEnv)
- ✅ Training script with Optuna hyperparameter optimization
- ✅ Model versioning and management

---

## 🚀 Phase 1: Backtesting Engine (CRITICAL - NEXT STEP)

**Priority**: HIGH - Cannot deploy to live trading without backtesting
**Estimated Time**: 3-5 days
**Owner**: Backend Team

### Why This is Next:
According to specification Section 7 (Evaluation Framework):
> "The agent will be evaluated on historical data before live deployment"

We have trained models but NO WAY to validate they actually make money before going live!

### Components to Build:

#### 1.1. Backtesting Engine Module
**File**: `backend/evaluation/backtester.py`

```python
class Backtester:
    """
    Backtest a trained RL agent on historical data
    
    Features:
    - Replay historical data through trained agent
    - Track all actions, rewards, positions
    - Calculate performance metrics
    - Generate equity curve
    - Commission and slippage simulation
    """
    
    def __init__(self, model_path, data, config):
        pass
    
    def run(self):
        """Run full backtest and return results"""
        pass
    
    def calculate_metrics(self):
        """Calculate Sharpe, Sortino, max drawdown, etc."""
        pass
    
    def generate_report(self):
        """Generate comprehensive backtest report"""
        pass
```

**Key Features**:
- Load trained model (PPO/SAC)
- Replay test dataset (20% holdout from training)
- Execute agent decisions step-by-step
- Track portfolio value over time
- Apply realistic costs (commission, slippage)
- Generate detailed trade log

#### 1.2. Performance Metrics Calculator
**File**: `backend/evaluation/metrics.py`

```python
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Sharpe: (mean return - risk free) / std"""
    pass

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Sortino: Only penalize downside volatility"""
    pass

def calculate_max_drawdown(equity_curve):
    """Maximum peak-to-trough decline"""
    pass

def calculate_win_rate(trades):
    """% of profitable trades"""
    pass

def calculate_profit_factor(trades):
    """Gross profit / gross loss"""
    pass

def calculate_calmar_ratio(returns, max_drawdown):
    """Return / max drawdown"""
    pass

def calculate_trade_statistics(trades):
    """
    Returns:
    - Total trades
    - Win rate
    - Avg win / Avg loss
    - Largest win / loss
    - Consecutive wins / losses
    """
    pass
```

#### 1.3. Backtesting API Endpoint
**File**: `backend/api/main.py`

```python
@app.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    """
    Run backtest on trained model
    
    Request:
    {
        "model_path": "models/ppo_AAPL_v3.zip",
        "symbol": "AAPL",
        "start_date": "2023-01-01",
        "end_date": "2024-01-01",
        "initial_capital": 100000,
        "commission": 0.001
    }
    
    Response:
    {
        "status": "success",
        "metrics": {
            "total_return": 0.234,
            "sharpe_ratio": 1.45,
            "sortino_ratio": 1.89,
            "max_drawdown": -0.12,
            "win_rate": 0.58,
            "total_trades": 142
        },
        "equity_curve": [...],
        "trades": [...]
    }
    """
    pass
```

#### 1.4. Backtesting UI Component
**File**: `frontend/src/components/TabBacktest.jsx`

**Features**:
- Model selection dropdown (list all saved models)
- Date range picker (backtest period)
- Initial capital input
- Commission/slippage settings
- "Run Backtest" button
- Real-time progress indicator
- Results display:
  - Performance metrics cards
  - Equity curve chart (Recharts)
  - Trade history table
  - Monthly returns heatmap
  - Drawdown chart
- Export report button (PDF/CSV)

**UI Layout**:
```
┌─────────────────────────────────────────┐
│ 📊 Backtest: Evaluate Trained Models   │
├─────────────────────────────────────────┤
│ Model: [ppo_AAPL_v3.zip ▼]             │
│ Date Range: [2023-01-01] to [2024-01-01] │
│ Initial Capital: [$100,000]             │
│ Commission: [0.1%]                      │
│ [Run Backtest]                          │
├─────────────────────────────────────────┤
│ 📈 Performance Metrics                  │
│ ┌──────────┬──────────┬──────────┐     │
│ │ Total    │ Sharpe   │ Max DD   │     │
│ │ Return   │ Ratio    │          │     │
│ │ +23.4%   │ 1.45     │ -12.1%   │     │
│ └──────────┴──────────┴──────────┘     │
├─────────────────────────────────────────┤
│ 📉 Equity Curve                         │
│ [Interactive Line Chart]                │
├─────────────────────────────────────────┤
│ 📋 Trade History                        │
│ [Paginated Table with all trades]      │
└─────────────────────────────────────────┘
```

---

## 🚀 Phase 2: Walk-Forward Analysis (ADVANCED)

**Priority**: MEDIUM - After basic backtesting works
**Estimated Time**: 2-3 days

### Why Important:
Prevents overfitting - tests if model adapts to new data.

### Components:

#### 2.1. Walk-Forward Engine
**File**: `backend/evaluation/walk_forward.py`

```python
class WalkForwardAnalysis:
    """
    Walk-forward testing:
    1. Train on window 1 (e.g., Jan-Jun 2023)
    2. Test on window 2 (Jul-Sep 2023)
    3. Retrain on windows 1+2
    4. Test on window 3 (Oct-Dec 2023)
    5. Repeat...
    
    Simulates real-world retraining schedule
    """
    
    def __init__(self, train_window_months=6, test_window_months=3):
        pass
    
    def run(self):
        """Execute walk-forward analysis"""
        pass
```

---

## 🚀 Phase 3: Paper Trading Mode (LIVE SIMULATION)

**Priority**: HIGH - Final step before live trading
**Estimated Time**: 4-6 days

### Why Critical:
Tests agent in REAL-TIME market conditions without risking money.

### Components:

#### 3.1. Paper Trading Engine
**File**: `backend/execution/paper_trading.py`

```python
class PaperTradingEngine:
    """
    Paper trading with live data:
    - Connects to IBKR Market Data (real-time prices)
    - Agent makes decisions in real-time
    - Simulates order execution (no real money)
    - Tracks performance as if real trading
    """
    
    def __init__(self, model_path, symbol, initial_capital):
        pass
    
    def start(self):
        """Start paper trading session"""
        pass
    
    def stop(self):
        """Stop paper trading session"""
        pass
```

#### 3.2. Paper Trading UI
**File**: `frontend/src/components/TabPaperTrading.jsx`

**Features**:
- Model selection
- Symbol input
- Start/Stop controls
- Real-time status display:
  - Current position
  - Portfolio value
  - Today's P&L
  - Recent actions
- Performance metrics (updated live)
- Chart with agent actions overlaid

---

## 🚀 Phase 4: Model Comparison & A/B Testing

**Priority**: MEDIUM
**Estimated Time**: 2-3 days

### Components:

#### 4.1. Model Comparison Tool
**File**: `backend/evaluation/model_comparator.py`

```python
def compare_models(model_paths, test_data):
    """
    Compare multiple models on same test data
    
    Returns:
    - Side-by-side metrics
    - Statistical significance tests
    - Best model recommendation
    """
    pass
```

#### 4.2. Comparison UI
Show multiple models' performance side-by-side in table and charts.

---

## 📊 Implementation Priority Order

### Week 1: Core Backtesting (MUST HAVE)
1. ✅ Day 1-2: Backtester class + metrics calculator
2. ✅ Day 3: API endpoint for backtesting
3. ✅ Day 4-5: Backtesting UI component
4. ✅ Day 5: Testing with AAPL trained model

### Week 2: Paper Trading (MUST HAVE)
1. ✅ Day 1-2: Paper trading engine
2. ✅ Day 3-4: Real-time data integration
3. ✅ Day 4-5: Paper trading UI
4. ✅ Day 5: End-to-end testing

### Week 3: Advanced Features (NICE TO HAVE)
1. Walk-forward analysis
2. Model comparison tool
3. Advanced analytics dashboard

---

## 🎯 Success Criteria

**Before moving to live trading, we MUST have:**

✅ **Backtesting**:
- [x] Can load trained model
- [x] Can replay historical data
- [x] Calculates all key metrics (Sharpe, Sortino, max DD, win rate)
- [x] Generates equity curve
- [x] Shows all trades with entry/exit prices
- [x] UI displays results clearly

✅ **Paper Trading**:
- [x] Connects to live market data
- [x] Agent makes real-time decisions
- [x] Simulates order execution
- [x] Tracks performance live
- [x] Can run for multiple days without issues
- [x] UI shows real-time status

✅ **Validation**:
- [x] Backtested model shows positive returns
- [x] Sharpe ratio > 1.0
- [x] Max drawdown < 20%
- [x] Paper trading confirms backtest results
- [x] No technical issues during paper trading

---

## 📝 Related Specification Sections

From `RL_System_Specification.md`:

**Section 7: Evaluation Framework**
> "Continuous evaluation is critical for RL agent reliability:
> - Backtesting: The agent will be evaluated on historical data before live deployment
> - Paper Trading: A paper trading mode will be available for live simulation without real capital risk
> - Performance Metrics: Key metrics (Sharpe, Sortino, max drawdown, win rate, etc.) will be tracked and visualized
> - A/B Testing: Multiple agent versions can be evaluated in parallel to select the best performer"

**Section 15: Retraining Workflow**
> "5. Backtesting: Evaluate the new model version on historical data to ensure performance improvements or stability
> 6. Redeployment: Deploy the validated model to live trading if backtest results meet predefined criteria"

---

## 🔧 Technical Dependencies

**Python Packages** (add to requirements.txt):
```
backtrader>=1.9.78  # Backtesting framework (optional, can build custom)
quantstats>=0.0.62  # Performance analytics
empyrical>=0.5.5    # Financial metrics
```

**Frontend Packages**:
```
recharts  # Already installed
react-table  # For trade history table
date-fns  # Date formatting
```

---

## 📚 Documentation to Create

1. **BACKTESTING_GUIDE.md** - How to backtest models
2. **PAPER_TRADING_GUIDE.md** - How to run paper trading
3. **PERFORMANCE_METRICS.md** - Explanation of all metrics
4. **API_EVALUATION_ENDPOINTS.md** - API documentation for evaluation endpoints

---

## 🚨 Critical Notes

**DO NOT SKIP BACKTESTING!**
- Training a model ≠ model works
- Must validate on unseen data
- Must test in realistic conditions
- Paper trading is mandatory before live deployment

**Risk Management**:
- Even with good backtest, start with small capital in live trading
- Monitor closely for first few weeks
- Have kill switch ready

---

## Next Action Item

**START HERE**: Create `backend/evaluation/backtester.py` skeleton and implement basic backtesting workflow.

```bash
# Create evaluation directory
mkdir -p backend/evaluation

# Create initial files
touch backend/evaluation/__init__.py
touch backend/evaluation/backtester.py
touch backend/evaluation/metrics.py
touch backend/evaluation/walk_forward.py

# Start implementation
code backend/evaluation/backtester.py
```
