# System Architecture - Training Module

Simple explanation of how everything connects.

---

## Big Picture

```
User (Browser)
    ↓
Frontend (React)
    ↓ HTTP
Backend API (Flask)
    ↓
Training System
    ↓
Saved Models
```

---

## Component Breakdown

### 1. Frontend (React App)

**What it does:** User interface for training and monitoring

**Main parts:**
- `TabTraining.jsx` - Main training screen
- `TrainingProgress.jsx` - Shows training progress bars
- `ModelSelector.jsx` - Pick which model to use
- `BacktestResults.jsx` - Shows backtest metrics
- `DriftAlert.jsx` - Warns when market changes

**How it works:**
1. User fills in training form (symbol, dates, params)
2. Clicks "Start Training"
3. Frontend sends request to backend
4. Polls every 5 seconds for progress updates
5. Shows real-time stats and graphs

---

### 2. Backend API (Flask)

**What it does:** REST API that handles requests

**Location:** `backend/api/main.py`

**Endpoints:**
```
POST /api/training/train          → Start training
GET  /api/training/progress/{id}  → Get progress
POST /api/training/stop           → Stop training
GET  /api/training/models         → List models
POST /api/training/backtest       → Run backtest
GET  /api/training/drift_status   → Check drift
```

**How it works:**
1. Receives HTTP request from frontend
2. Validates input data
3. Calls training functions
4. Returns JSON response

---

### 3. Training System

The brain of the operation. Here's the flow:

```
Data → Environment → Agent → Training Loop → Model
```

#### 3.1 Data Loading

**File:** `training/train.py` (function: `load_data`)

**What it does:**
1. Downloads data from Yahoo Finance
2. Calculates technical indicators (RSI, MACD, etc.)
3. Normalizes values (z-score)
4. Splits into train/test sets

**Data structure:**
```
Date  | Open | High | Low | Close | Volume | RSI | MACD | ...
2020  | 150  | 152  | 149 | 151   | 1M     | 65  | 0.5  | ...
```

#### 3.2 Environments (Gym)

**Files:** 
- `environments/base_env.py` - Base logic
- `environments/stock_env.py` - PPO version
- `environments/etf_env.py` - SAC version

**What it does:** Simulates trading

**State (what the agent sees):**
- Portfolio: cash, stock value, total value, positions
- Market: prices, volume, indicators
- History: last 5 returns

**Actions (what the agent can do):**
- PPO: 0=Hold, 1=Buy, 2=Sell
- SAC: continuous value -1.0 to +1.0 (position size)

**Rewards (what the agent gets):**
- Positive: portfolio gains
- Negative: risk penalties, drawdowns
- Goal: maximize Sharpe ratio

**Trading rules:**
- Commission: IBKR tiered (0.01 $/share, $2.50 min, 1% cap)
- No short selling
- Max 100% of capital

**Example step:**
```python
observation = env.reset()           # Start
action = agent.predict(observation) # Agent decides
observation, reward, done, info = env.step(action)
# reward = return - risk_penalty - drawdown_penalty
```

#### 3.3 Agents (RL Algorithms)

**Files:**
- `agents/base_agent.py` - Interface
- `agents/ppo_agent.py` - PPO wrapper
- `agents/sac_agent.py` - SAC wrapper
- `agents/agent_factory.py` - Creates agents

**What they do:** Learn to trade

**PPO (Proximal Policy Optimization):**
- For stocks with discrete actions
- Neural network: [64, 64] hidden layers
- On-policy (learns from current experience)
- Stable and safe updates

**SAC (Soft Actor-Critic):**
- For ETFs with continuous actions
- Neural network: [256, 256] hidden layers
- Off-policy (learns from replay buffer)
- Good for exploration

**How they learn:**
1. Start with random actions
2. Collect experience (states, actions, rewards)
3. Update neural network to maximize rewards
4. Repeat until performance plateaus

#### 3.4 Training Loop

**File:** `training/train.py` (function: `train_agent`)

**Flow:**
```
1. Load config
2. Load and prepare data
3. Create environment
4. Create agent
5. Train for N timesteps
6. Save model + metadata
7. Return results
```

**During training:**
- Callback logs progress every 1000 steps
- Tracks: episode rewards, steps, losses
- Saves progress to JSON file
- Can be stopped anytime

**Typical training:**
- Timesteps: 50,000 - 200,000
- Time: 5-30 minutes (CPU)
- Episodes: 100-500
- Final Sharpe: 1.0-2.5

---

### 4. Model Management

**File:** `models/model_manager.py`

**What it does:**
- Saves trained models with version numbers
- Stores metadata (Sharpe, date, params)
- Loads models for backtesting or live trading
- Compares different versions
- Archives old models

**Model structure:**
```
backend/models/
├── ppo/
│   ├── ppo_AAPL_v001_20251108.zip  ← Model file
│   ├── ppo_AAPL_v001_20251108.json ← Metadata
│   ├── ppo_AAPL_v002_20251109.zip
│   └── archive/                     ← Old versions
└── sac/
    ├── sac_TQQQ_v001_20251108.zip
    └── sac_TQQQ_v001_20251108.json
```

**Metadata example:**
```json
{
  "model_id": "ppo_AAPL_v001_20251108",
  "agent_type": "PPO",
  "symbol": "AAPL",
  "version": 1,
  "created_at": "2025-11-08 14:30:00",
  "metrics": {
    "sharpe_ratio": 1.85,
    "total_return": 0.42,
    "max_drawdown": -0.15,
    "win_rate": 0.58
  },
  "config": { ... }
}
```

---

### 5. Evaluation System

#### 5.1 Metrics

**File:** `evaluation/metrics.py`

**8 metrics calculated:**
1. **Sharpe Ratio** - Risk-adjusted return (target: 1.5+)
2. **Sortino Ratio** - Like Sharpe but only downside risk
3. **Max Drawdown** - Biggest loss from peak (lower is better)
4. **Win Rate** - % of profitable trades
5. **Profit Factor** - Total gains / Total losses
6. **Total Return** - % gain/loss
7. **Calmar Ratio** - Return / Max Drawdown
8. **Alpha** - Return vs Buy & Hold

**Example output:**
```python
{
  "sharpe_ratio": 1.85,
  "sortino_ratio": 2.12,
  "max_drawdown": -0.15,
  "win_rate": 0.58,
  "profit_factor": 1.45,
  "total_return": 0.42,
  "calmar_ratio": 2.80,
  "alpha": 0.18
}
```

#### 5.2 Backtesting

**File:** `evaluation/backtester.py`

**What it does:**
1. Loads test data (not used in training)
2. Runs model through environment
3. Records every trade
4. Calculates all metrics
5. Compares to Buy & Hold

**Backtest report:**
```python
{
  "agent_metrics": { ... },      # Your model's performance
  "buy_and_hold_metrics": { ... }, # Benchmark
  "trades": [
    {"date": "2024-01-05", "action": "BUY", "price": 150, ...},
    {"date": "2024-01-12", "action": "SELL", "price": 155, ...}
  ],
  "portfolio_value": [10000, 10050, 10100, ...], # Daily values
  "comparison": {
    "sharpe_improvement": 0.35,
    "return_improvement": 0.08
  }
}
```

---

### 6. Drift Detection

**File:** `utils/state_normalizer.py`

**What it does:** Checks if market behavior changed

**How it works:**
1. Save data distribution during training (mean, std)
2. Later, check new data against saved distribution
3. Use statistical tests (KS test, PSI)
4. Alert if distribution shifted

**When to retrain:**
- High drift (0.5-0.7) - Consider retraining
- Critical drift (>0.7) - Definitely retrain

**Example:**
```python
# During training
normalizer.fit(train_data)
normalizer.save_params("normalizer_AAPL_v1.json")

# Later, check drift
drift = normalizer.detect_drift(recent_data)
if drift['severity'] == 'critical':
    trigger_retraining()
```

---

### 7. Auto-Retraining

**File:** `training/retraining_scheduler.py`

**What it does:** Automatically retrains models

**Schedule options:**
- Daily (every morning)
- Weekly (every Monday)
- Monthly (1st of month)

**Retraining flow:**
```
1. Load historical + recent data
2. Train new model
3. Backtest new model
4. Compare with current best model
5. If better → deploy new model
6. If worse → archive new model
```

**Example usage:**
```python
scheduler = RetrainingScheduler(
    symbol="AAPL",
    agent_type="PPO",
    frequency="Weekly"
)
scheduler.schedule()  # Runs every Monday
```

---

## Data Flow Diagram

### Training Flow
```
User Input
    ↓
Frontend Form (symbol, dates, params)
    ↓
POST /api/training/train
    ↓
Backend API
    ↓
train_agent()
    ↓
load_data() → Yahoo Finance
    ↓
Normalize Data
    ↓
Create Environment
    ↓
Create Agent (PPO/SAC)
    ↓
Training Loop (50k-200k steps)
    │
    ├─→ Agent observes state
    ├─→ Agent picks action
    ├─→ Environment executes trade
    ├─→ Environment returns reward
    └─→ Agent learns from experience
    ↓
Save Model + Metadata
    ↓
Return results
    ↓
Frontend shows completion
```

### Backtesting Flow
```
User clicks "Backtest"
    ↓
POST /api/training/backtest
    ↓
Load Model
    ↓
Load Test Data (unseen)
    ↓
Create Environment
    ↓
Run Episode
    │
    └─→ For each timestep:
        ├─→ Agent predicts action
        ├─→ Environment executes
        └─→ Record trade
    ↓
Calculate Metrics
    ↓
Compare to Buy & Hold
    ↓
Return report
    ↓
Frontend shows results
```

### Drift Detection Flow
```
Timer (every 5 minutes)
    ↓
Frontend polls drift status
    ↓
GET /api/training/drift_status?symbol=AAPL&days=30
    ↓
Load normalizer (from training)
    ↓
Download recent 30 days data
    ↓
Compare distributions (KS test + PSI)
    ↓
Calculate severity
    ↓
Return drift status
    ↓
Frontend shows alert if needed
```

---

## Key Design Decisions

### 1. Why Two Agents?

**PPO for stocks:**
- Simple discrete actions (buy/sell/hold)
- More stable learning
- Good for less volatile assets

**SAC for leveraged ETFs:**
- Continuous position sizing needed
- Better exploration
- Handles high volatility

### 2. Why Gym Environments?

Standard interface that works with any RL library. Easy to:
- Test different algorithms
- Swap environments
- Debug trading logic

### 3. Why Flask Not FastAPI?

Already using Flask for other modules. Could switch to FastAPI later for:
- Auto API docs
- Better performance
- Type validation

### 4. Why Client-Side Polling?

Training can take 5-30 minutes. Instead of:
- Long HTTP requests (timeout issues)
- WebSockets (more complex)

We use:
- Background thread for training
- Client polls every 5 seconds
- Simple and reliable

### 5. Why JSON Metadata?

Model files (ZIP) + Metadata (JSON) separated because:
- Easy to read metadata without loading model
- Can query/filter models quickly
- Simple to add new metadata fields

---

## Performance Characteristics

### Training Speed
- PPO: ~2000 steps/second (CPU)
- SAC: ~1500 steps/second (CPU)
- 100k steps = ~1-2 minutes

### Memory Usage
- Training: ~500MB RAM
- Model size: ~1-5MB each
- Data: ~50MB per year per symbol

### Accuracy
- Sharpe ratio: 1.0-2.5 typical
- Win rate: 50-60% typical
- Max drawdown: 10-20% typical

---

## Extending The System

### Add New Agent Type

1. Create `agents/new_agent.py`
2. Inherit from `BaseAgent`
3. Implement `train()`, `predict()`, `save()`, `load()`
4. Add to `agent_factory.py`

### Add New Metric

1. Add function to `evaluation/metrics.py`
2. Update `calculate_all_metrics()`
3. Frontend will show it automatically

### Add New Feature

1. Add to data loading in `train.py`
2. Update environment state space
3. Update normalizer

### Change Reward Function

Edit `_calculate_reward()` in:
- `environments/stock_env.py` (PPO)
- `environments/etf_env.py` (SAC)

---

## Common Patterns

### Error Handling
All API endpoints return:
```json
{
  "success": true/false,
  "data": { ... },
  "error": "error message if failed"
}
```

### Session Tracking
Training sessions use UUID:
```python
session_id = str(uuid.uuid4())
training_sessions[session_id] = {
    "status": "running",
    "progress": {...}
}
```

### Model Versioning
Auto-increment version numbers:
```
ppo_AAPL_v001
ppo_AAPL_v002
ppo_AAPL_v003
```

### Config Management
Three ways to configure:
1. UI form (saved to localStorage)
2. JSON file (saved/loaded via API)
3. Python code (presets in config.py)

---

## Testing Strategy

**Unit tests** (not implemented yet):
- Test each component separately
- Mock dependencies
- Files: `backend/tests/test_*.py`

**Integration tests** (not implemented yet):
- Test components together
- Use small datasets
- Verify full workflows

**Manual testing** (current approach):
- Train small models (10k steps)
- Run backtests
- Check metrics make sense

---

## Security Notes

**Current setup** (development):
- No authentication
- Open CORS
- Local only

**For production:**
- Add API keys
- Restrict CORS
- Use HTTPS
- Rate limiting
- Input validation

---

## Deployment

**Development:**
```bash
# Backend
cd backend
python api/main.py

# Frontend
cd frontend
npm start
```

**Production:**
- Backend: Gunicorn + Nginx
- Frontend: Build + serve static files
- Database: Keep SQLite or migrate to PostgreSQL
- Models: Store on S3 or file server

---

**That's it!** The system is simpler than it looks. Each component does one thing well, and they all connect through clean interfaces.
