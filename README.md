# YoopRL Trading System - Training Module

A complete reinforcement learning system for automated stock and ETF trading.

---

## What This Does

This system trains AI agents to trade stocks and ETFs using reinforcement learning. It learns from historical data and makes buy/sell/hold decisions.

**Two types of agents:**
- **PPO** - For regular stocks (discrete actions: buy/sell/hold)
- **SAC** - For leveraged ETFs (continuous actions: position sizing)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Backend

```bash
cd backend
python api/main.py
```

Backend runs on `http://localhost:5000`

### 3. Start Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs on `http://localhost:3000`

---

## How To Train An Agent

### Option 1: Using the UI

1. Open `http://localhost:3000`
2. Go to **Training** tab
3. Select agent type (PPO or SAC)
4. Pick a stock symbol (e.g., AAPL)
5. Set date range for training data
6. Click **Start Training**
7. Watch progress in real-time

### Option 2: Using Python Code

```python
from training.train import train_agent
from config.training_config import TrainingConfig

# Create config
config = TrainingConfig.get_conservative_preset()
config.symbol = "AAPL"
config.agent_type = "PPO"
config.training_settings.total_timesteps = 100000

# Train
result = train_agent(config)
print(f"Training done! Model saved at: {result['model_path']}")
```

### Option 3: Using API

```bash
curl -X POST http://localhost:5000/api/training/train \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "agent_type": "PPO",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "total_timesteps": 100000
  }'
```

---

## Key Features

### Training
- **Real market data** from Yahoo Finance
- **Custom rewards** - Sharpe ratio optimization, risk penalties, drawdown penalties
- **Progress tracking** - See training stats in real-time
- **Model versioning** - Auto-saves models with metadata

### Evaluation
- **Backtesting** - Test your model on historical data
- **8 performance metrics** - Sharpe, Sortino, Max Drawdown, Win Rate, etc.
- **Benchmark comparison** - Compare against Buy & Hold strategy

### Management
- **Drift detection** - Alerts when market conditions change
- **Auto-retraining** - Schedule daily/weekly/monthly retraining
- **Model comparison** - Compare multiple model versions

---

## File Structure

```
backend/
├── environments/       # Trading environments (Gym)
│   ├── base_env.py    # Base trading logic
│   ├── stock_env.py   # PPO environment (stocks)
│   └── etf_env.py     # SAC environment (ETFs)
│
├── agents/            # RL agents
│   ├── base_agent.py  # Abstract agent interface
│   ├── ppo_agent.py   # PPO wrapper (Stable-Baselines3)
│   ├── sac_agent.py   # SAC wrapper (Stable-Baselines3)
│   └── agent_factory.py
│
├── training/          # Training loop
│   ├── train.py       # Main training function
│   └── retraining_scheduler.py
│
├── evaluation/        # Performance evaluation
│   ├── metrics.py     # 8 performance metrics
│   └── backtester.py  # Backtesting framework
│
├── models/            # Saved models
│   ├── model_manager.py
│   ├── ppo/          # PPO models
│   └── sac/          # SAC models
│
├── utils/
│   └── state_normalizer.py  # Data normalization + drift detection
│
├── config/
│   └── training_config.py    # Config + presets
│
└── api/
    └── main.py        # Flask REST API

frontend/
└── src/
    ├── components/
    │   └── TabTraining.jsx    # Training UI
    └── services/
        └── trainingAPI.js     # API calls
```

---

## API Endpoints

### Training
- `POST /api/training/train` - Start training
- `GET /api/training/progress/{session_id}` - Get training progress
- `POST /api/training/stop` - Stop training

### Models
- `GET /api/training/models` - List saved models
- `POST /api/training/load_model` - Load a model
- `DELETE /api/training/models/{model_id}` - Delete a model

### Evaluation
- `POST /api/training/backtest` - Run backtest

### Management
- `GET /api/training/drift_status` - Check for distribution drift
- `POST /api/training/config/save` - Save config
- `GET /api/training/config/load` - Load config

Full API docs: `http://localhost:5000/docs` (when backend is running)

---

## Configuration Presets

**Conservative** (Low risk):
- Learning rate: 0.0001
- Gamma: 0.99
- Risk penalty: 0.2
- Good for stable stocks

**Aggressive** (High risk):
- Learning rate: 0.001
- Gamma: 0.95
- Risk penalty: 0.05
- Good for volatile stocks

**Balanced** (Medium risk):
- Learning rate: 0.0003
- Gamma: 0.98
- Risk penalty: 0.1
- Good starting point

You can customize all parameters in the UI or config file.

---

## Training Tips

1. **More data is better** - Use at least 2 years of training data
2. **Start with presets** - Try Conservative first, then adjust
3. **Check drift alerts** - Retrain if market conditions change
4. **Compare models** - Run backtests to pick the best version
5. **Watch the Sharpe ratio** - Target 1.5+ for good performance

---

## Common Issues

**Training is slow**
- Reduce `total_timesteps` (try 50,000 first)
- Use fewer features
- CPU is fine, no GPU needed

**Model performs poorly**
- Try different hyperparameters
- Use more training data
- Check if data quality is good
- Consider switching agent type (PPO ↔ SAC)

**API errors**
- Check backend is running (`python api/main.py`)
- Verify port 5000 is not in use
- Check CORS settings in `main.py`

---

## Requirements

- Python 3.9+
- Node.js 16+
- 8GB RAM (16GB recommended)
- Internet connection (for downloading market data)

---

## Tech Stack

**Backend:**
- Flask (web server)
- Stable-Baselines3 (RL algorithms)
- Gym (trading environments)
- NumPy/Pandas (data processing)
- SciPy (drift detection)

**Frontend:**
- React 18
- Material-UI
- Recharts (graphs)

---

## Next Steps

1. Train your first model with default settings
2. Run a backtest to see performance
3. Try different hyperparameters
4. Set up auto-retraining
5. Connect to IBKR for live trading (separate module)

---

## Documentation

- **Architecture**: See `Docs/ARCHITECTURE.md`
- **Training Plan**: See `Docs/TRAINING_IMPLEMENTATION_PLAN.md`
- **Setup Guide**: See `Docs/SETUP_INSTRUCTIONS.md`

---

## License

MIT License - Do whatever you want with this code.

---

**Questions?** Check the docs or read the code - it's well commented.

**Happy Trading!** 🚀
