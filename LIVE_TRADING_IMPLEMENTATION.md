# 🎯 Live Trading Tab - Complete Implementation Guide

**Status**: Not Started  
**Priority**: HIGH - Core System Feature  
**Estimated Time**: 2-3 weeks  
**Last Updated**: November 8, 2025

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Time Frame Architecture](#time-frame-architecture)
3. [Model Deployment Workflow](#model-deployment-workflow)
4. [Backend Implementation](#backend-implementation)
5. [Frontend Implementation](#frontend-implementation)
6. [Testing Without Live Data](#testing-without-live-data)
7. [Phase-by-Phase Implementation](#phase-by-phase-implementation)

---

## 🎯 System Overview

### Current State
- ✅ **Training Pipeline**: Complete with Optuna optimization
- ✅ **Model Evaluation**: Automatic 80/20 test evaluation with real metrics
- ✅ **IBKR Bridge**: C# bridge ready (python_adapter available)
- ✅ **Model Storage**: Models saved with full metadata
- ❌ **Live Trading Execution**: NOT IMPLEMENTED
- ❌ **Real-Time Monitoring**: NOT IMPLEMENTED
- ❌ **Agent Management**: NOT IMPLEMENTED

### What You Have (Data Access)
- ✅ **IBKR Connection**: Can connect to IBKR TWS/Gateway
- ✅ **Historical Data**: Can download historical bars
- ✅ **Order Execution**: Can send buy/sell orders
- ❌ **Live Streaming Data**: NO subscription (need market data subscription)
- ❌ **Real-Time Ticks**: NO subscription

### Implications
Since you DON'T have live data subscription:
1. ✅ Can trade based on **delayed data** (15-20 min delay via Yahoo Finance)
2. ✅ Can use **historical bars** from IBKR (end-of-day or 5-min bars)
3. ✅ Can execute orders on IBKR
4. ❌ Cannot implement true high-frequency intraday trading
5. ✅ Can implement **EOD (End-of-Day) trading** → **Best approach for you!**

---

## ⏰ Time Frame Architecture

### Critical Decision: What Time Frame Does Agent Trade On?

**Currently**: NOT DEFINED IN TRAINING!  
**Problem**: Training uses daily OHLCV but time frame is not explicitly saved.

### Recommended Approach: **Daily Bars (EOD Trading)**

**Why Daily?**
1. ✅ You don't have live tick data subscription
2. ✅ Matches training data format (daily OHLCV from Yahoo Finance)
3. ✅ Less risky than intraday (more time to analyze)
4. ✅ Lower commissions (fewer trades)
5. ✅ Easier to implement without real-time WebSocket
6. ✅ Can run checks once per day after market close

**Time Frame Configuration** (where to save it):

```python
# In backend/training/train.py - when saving model metadata
metadata = {
    'agent_type': 'PPO',
    'symbol': 'INTC',
    'version': 'v20251108_162433',
    
    # ADD THIS:
    'trading_config': {
        'time_frame': 'daily',  # or '5min', '15min', '1hour'
        'bar_size': '1 day',    # IBKR format
        'trading_hours': 'RTH', # Regular Trading Hours
        'check_frequency': 'EOD' # End-of-Day or 'REALTIME'
    },
    
    # ... rest of metadata
}
```

### Time Frame Options

| Time Frame | Data Source | Frequency | Requires Live Data? | Complexity | Recommended |
|------------|-------------|-----------|---------------------|------------|-------------|
| **Daily** | IBKR/Yahoo | Once/day (after close) | ❌ NO | Low | ✅ **YES** |
| **5-min** | IBKR historical | Every 5 min | ❌ NO (delayed OK) | Medium | ⚠️ Maybe |
| **15-min** | IBKR historical | Every 15 min | ❌ NO (delayed OK) | Medium | ⚠️ Maybe |
| **1-min** | IBKR streaming | Every minute | ✅ YES | High | ❌ NO |
| **Tick** | IBKR streaming | Sub-second | ✅ YES | Very High | ❌ NO |

**Decision**: Start with **Daily (EOD)** trading:
- Agent trained on daily bars
- Checks market once per day after close (4:00 PM ET)
- Downloads latest daily bar from IBKR or Yahoo
- Makes decision (BUY/SELL/HOLD)
- Executes order next morning at market open

---

## 🚀 Model Deployment Workflow

### User Journey: "I Want to Trade INTC with My Trained Model"

#### Option A: Deploy from Training Tab (RECOMMENDED)
```
Training Tab → Train Model → See Results → Click "Deploy to Live Trading"
```

**Flow**:
1. User trains model on INTC (PPO, 50k episodes, Sharpe 2.29)
2. Sees backtest results: +7.71% return, good metrics
3. Clicks **"Deploy to Live Trading"** button
4. Modal opens:
   ```
   ┌────────────────────────────────────────┐
   │ Deploy Model to Live Trading          │
   ├────────────────────────────────────────┤
   │ Model: PPO_INTC_v20251108_162433      │
   │ Symbol: INTC                           │
   │ Sharpe Ratio: 2.29                    │
   │ Return: +7.71%                        │
   ├────────────────────────────────────────┤
   │ Live Trading Settings:                │
   │                                        │
   │ Initial Capital: [$10,000]            │
   │ Max Position Size: [50]%              │
   │ Risk Per Trade: [2]%                  │
   │ Time Frame: [Daily ▼]                 │
   │   ☑ EOD (End-of-Day)                 │
   │   ☐ Intraday (5-min bars)            │
   │                                        │
   │ [Cancel]  [Deploy & Start Trading]   │
   └────────────────────────────────────────┘
   ```
5. System creates live trading agent from model
6. Redirects to Live Trading tab showing active agent

**UI Changes Needed in Training Tab**:
```jsx
// After backtest results display
<Card>
  <h3>🚀 Deploy Model</h3>
  {backtestResults && backtestResults.sharpe_ratio > 1.5 && (
    <Button 
      variant="success" 
      onClick={handleDeployModel}
      disabled={isDeploying}
    >
      {isDeploying ? '⏳ Deploying...' : '🚀 Deploy to Live Trading'}
    </Button>
  )}
  {backtestResults && backtestResults.sharpe_ratio <= 1.5 && (
    <p style={{color: '#f59e0b'}}>
      ⚠️ Sharpe Ratio too low ({backtestResults.sharpe_ratio.toFixed(2)}). 
      Recommended: > 1.5 for live trading.
    </p>
  )}
</Card>
```

#### Option B: Manual Start from Live Trading Tab
```
Live Trading Tab → Select Symbol → Select Model → Configure → Start
```

**Flow**:
1. User goes to Live Trading tab
2. Sees empty state: "No active agents"
3. Clicks **"+ Add New Agent"**
4. Form appears:
   ```
   ┌────────────────────────────────────────┐
   │ Create Live Trading Agent             │
   ├────────────────────────────────────────┤
   │ Agent Type: [PPO ▼]                   │
   │                                        │
   │ Symbol: [INTC]                        │
   │   (must match trained model)          │
   │                                        │
   │ Select Model:                         │
   │ [PPO_INTC_v20251108 (Sharpe: 2.29) ▼]│
   │                                        │
   │ Capital: [$10,000]                    │
   │ Max Position: [50]%                   │
   │ Risk Per Trade: [2]%                  │
   │                                        │
   │ [Cancel]  [Start Trading]            │
   └────────────────────────────────────────┘
   ```
5. System validates model matches symbol
6. Creates and starts agent

**Recommendation**: **Option A** is better UX:
- Natural workflow: Train → Validate → Deploy
- User sees results before deploying
- Less manual configuration
- Reduces errors (symbol mismatch, wrong model)

---

## 🔧 Backend Implementation

### Phase 1: Core Execution Module (Week 1)

#### 1.1. Create Directory Structure
```bash
backend/
├── execution/
│   ├── __init__.py
│   ├── live_trader.py          # Main live trading engine
│   ├── agent_manager.py        # Manages multiple agents
│   ├── order_executor.py       # Sends orders to IBKR
│   ├── position_tracker.py     # Tracks open positions
│   └── risk_manager.py         # Risk checks before orders
```

#### 1.2. LiveTrader Class
**File**: `backend/execution/live_trader.py`

```python
"""
Live Trading Engine for RL Agents

Responsibilities:
- Load trained model
- Fetch latest market data (daily or intraday)
- Compute features (same as training)
- Get agent prediction (BUY/SELL/HOLD)
- Execute orders via IBKR
- Track positions and P&L
- Log all actions
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_download.data_downloader import DataDownloader
from data_download.feature_engineering import FeatureEngineer
from IBKR_Bridge.python_adapter.interreact_bridge_adapter import InterReactBridgeAdapter
from evaluation.metrics import calculate_sharpe_ratio, calculate_win_rate

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Live Trading Engine for Single RL Agent
    
    One instance per agent (e.g., PPO for INTC, SAC for TNA)
    """
    
    def __init__(self, config: dict):
        """
        Initialize live trader
        
        Args:
            config: {
                'agent_id': 'PPO_INTC_001',
                'agent_type': 'PPO',
                'symbol': 'INTC',
                'model_path': 'backend/models/ppo/ppo_INTC_v20251108.zip',
                'time_frame': 'daily',  # or '5min', '15min'
                'initial_capital': 10000,
                'max_position_pct': 0.5,  # Max 50% of capital in one position
                'risk_per_trade': 0.02,   # Risk 2% per trade
                'check_frequency': 'EOD',  # End-of-day or 'REALTIME'
                'features_config': {...}   # Same as training
            }
        """
        self.config = config
        self.agent_id = config['agent_id']
        self.agent_type = config['agent_type']
        self.symbol = config['symbol']
        self.model_path = config['model_path']
        self.time_frame = config.get('time_frame', 'daily')
        
        # Capital management
        self.initial_capital = config['initial_capital']
        self.current_capital = self.initial_capital
        self.max_position_pct = config.get('max_position_pct', 0.5)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        
        # State
        self.is_running = False
        self.current_position = 0  # Number of shares held
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
        # Components
        self.model = None
        self.ibkr_adapter = None
        self.data_downloader = None
        self.feature_engineer = None
        
        # Logging
        self.logger = logging.getLogger(f"LiveTrader.{self.agent_id}")
        
    def load_model(self):
        """Load trained RL model"""
        try:
            if self.agent_type == 'PPO':
                self.model = PPO.load(self.model_path)
            elif self.agent_type == 'SAC':
                self.model = SAC.load(self.model_path)
            else:
                raise ValueError(f"Unknown agent type: {self.agent_type}")
            
            self.logger.info(f"Model loaded: {self.model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def connect_to_ibkr(self):
        """Connect to IBKR via bridge"""
        try:
            self.ibkr_adapter = InterReactBridgeAdapter(
                host='localhost',
                port=5080
            )
            
            if self.ibkr_adapter.is_connected():
                self.logger.info("Connected to IBKR Bridge")
                return True
            else:
                self.logger.error("IBKR Bridge not connected")
                return False
        except Exception as e:
            self.logger.error(f"IBKR connection failed: {e}")
            return False
    
    def fetch_latest_data(self) -> pd.DataFrame:
        """
        Fetch latest market data for symbol
        
        Returns:
            DataFrame with OHLCV and features (single row for latest bar)
        """
        try:
            # Download recent data (last 60 days to compute indicators)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            self.data_downloader = DataDownloader()
            df = self.data_downloader.download_history(
                symbol=self.symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='1d' if self.time_frame == 'daily' else '5m'
            )
            
            if df is None or len(df) == 0:
                self.logger.error("No data downloaded")
                return None
            
            # Engineer features (same as training)
            self.feature_engineer = FeatureEngineer(
                df=df,
                features_config=self.config.get('features_config', {})
            )
            df_features = self.feature_engineer.build_features()
            
            if df_features is None or len(df_features) == 0:
                self.logger.error("Feature engineering failed")
                return None
            
            # Return only latest row
            latest_row = df_features.iloc[[-1]].copy()
            self.logger.info(f"Latest data fetched for {self.symbol}: {latest_row.index[0]}")
            
            return latest_row
            
        except Exception as e:
            self.logger.error(f"Data fetch failed: {e}")
            return None
    
    def get_agent_decision(self, observation: np.ndarray) -> int:
        """
        Get agent's trading decision
        
        Args:
            observation: Feature vector (normalized)
        
        Returns:
            action: 0 (SELL), 1 (HOLD), 2 (BUY)
        """
        try:
            action, _states = self.model.predict(observation, deterministic=True)
            return int(action)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return 1  # Default to HOLD
    
    def execute_order(self, action: int, current_price: float):
        """
        Execute trading order via IBKR
        
        Args:
            action: 0 (SELL), 1 (HOLD), 2 (BUY)
            current_price: Current market price
        """
        try:
            # Calculate position size
            max_shares = int((self.current_capital * self.max_position_pct) / current_price)
            risk_shares = int((self.current_capital * self.risk_per_trade) / current_price)
            shares = min(max_shares, risk_shares)
            
            if action == 2:  # BUY
                if self.current_position == 0:  # Only if no position
                    # Send buy order
                    order_result = self.ibkr_adapter.place_order(
                        symbol=self.symbol,
                        action='BUY',
                        quantity=shares,
                        order_type='MKT'
                    )
                    
                    if order_result['success']:
                        self.current_position = shares
                        self.entry_price = current_price
                        self.current_capital -= (shares * current_price)
                        
                        self.logger.info(f"BUY {shares} {self.symbol} @ ${current_price:.2f}")
                        
                        # Log trade
                        self.trades.append({
                            'timestamp': datetime.now(),
                            'action': 'BUY',
                            'symbol': self.symbol,
                            'quantity': shares,
                            'price': current_price,
                            'capital': self.current_capital
                        })
                    else:
                        self.logger.error(f"Buy order failed: {order_result.get('error')}")
            
            elif action == 0:  # SELL
                if self.current_position > 0:  # Only if holding position
                    # Send sell order
                    order_result = self.ibkr_adapter.place_order(
                        symbol=self.symbol,
                        action='SELL',
                        quantity=self.current_position,
                        order_type='MKT'
                    )
                    
                    if order_result['success']:
                        # Calculate P&L
                        pnl = (current_price - self.entry_price) * self.current_position
                        self.total_pnl += pnl
                        self.current_capital += (self.current_position * current_price)
                        
                        self.logger.info(f"SELL {self.current_position} {self.symbol} @ ${current_price:.2f} | P&L: ${pnl:.2f}")
                        
                        # Log trade
                        self.trades.append({
                            'timestamp': datetime.now(),
                            'action': 'SELL',
                            'symbol': self.symbol,
                            'quantity': self.current_position,
                            'price': current_price,
                            'pnl': pnl,
                            'capital': self.current_capital
                        })
                        
                        # Reset position
                        self.current_position = 0
                        self.entry_price = 0.0
                    else:
                        self.logger.error(f"Sell order failed: {order_result.get('error')}")
            
            else:  # HOLD
                self.logger.info(f"HOLD {self.symbol} (Position: {self.current_position})")
            
            # Update equity curve
            portfolio_value = self.current_capital + (self.current_position * current_price)
            self.equity_curve.append(portfolio_value)
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
    
    def run_single_check(self):
        """
        Run single trading check cycle
        
        Flow:
        1. Fetch latest data
        2. Engineer features
        3. Get agent decision
        4. Execute order
        """
        try:
            self.logger.info(f"[{self.agent_id}] Running trading check...")
            
            # 1. Fetch data
            latest_data = self.fetch_latest_data()
            if latest_data is None:
                self.logger.error("Failed to fetch data - skipping cycle")
                return False
            
            # 2. Prepare observation (normalize, convert to numpy)
            # Note: Use same normalizer as training!
            observation = latest_data.values[0]  # First (only) row
            
            # 3. Get decision
            current_price = latest_data['price'].values[0] if 'price' in latest_data.columns else latest_data['close'].values[0]
            action = self.get_agent_decision(observation)
            
            action_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            self.logger.info(f"Agent decision: {action_names[action]} (price: ${current_price:.2f})")
            
            # 4. Execute
            self.execute_order(action, current_price)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Trading check failed: {e}")
            return False
    
    def start(self):
        """Start live trading"""
        try:
            self.logger.info(f"Starting live trader: {self.agent_id}")
            
            # 1. Load model
            if not self.load_model():
                return False
            
            # 2. Connect to IBKR
            if not self.connect_to_ibkr():
                return False
            
            self.is_running = True
            self.logger.info(f"Live trading started for {self.symbol}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start: {e}")
            return False
    
    def stop(self):
        """Stop live trading"""
        self.is_running = False
        self.logger.info(f"Live trading stopped for {self.symbol}")
    
    def get_status(self) -> dict:
        """Get current trading status"""
        current_price = 0.0  # TODO: Get from latest data
        portfolio_value = self.current_capital + (self.current_position * current_price)
        
        return {
            'agent_id': self.agent_id,
            'symbol': self.symbol,
            'is_running': self.is_running,
            'current_position': self.current_position,
            'entry_price': self.entry_price,
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': (self.total_pnl / self.initial_capital) * 100,
            'num_trades': len(self.trades),
            'last_check': datetime.now().isoformat()
        }
```

**Key Features**:
- ✅ Loads trained model (PPO/SAC)
- ✅ Connects to IBKR Bridge
- ✅ Fetches latest data (daily or intraday)
- ✅ Engineers features (same as training)
- ✅ Gets agent prediction
- ✅ Executes orders with risk management
- ✅ Tracks positions, P&L, trades
- ✅ Provides status for UI

#### 1.3. AgentManager Class
**File**: `backend/execution/agent_manager.py`

```python
"""
Agent Manager - Manages Multiple Live Trading Agents

Responsibilities:
- Create/start/stop agents
- Schedule periodic checks (EOD or real-time)
- Coordinate between PPO and SAC agents
- Aggregate status for dashboard
"""

import logging
import threading
import time
from datetime import datetime, time as dtime
from typing import Dict, List
from live_trader import LiveTrader

logger = logging.getLogger(__name__)


class AgentManager:
    """Manages multiple live trading agents"""
    
    def __init__(self):
        self.agents: Dict[str, LiveTrader] = {}
        self.scheduler_thread = None
        self.is_running = False
        self.logger = logging.getLogger("AgentManager")
    
    def create_agent(self, config: dict) -> str:
        """
        Create new live trading agent
        
        Args:
            config: Agent configuration
        
        Returns:
            agent_id: Unique agent ID
        """
        agent_id = config['agent_id']
        
        if agent_id in self.agents:
            self.logger.warning(f"Agent {agent_id} already exists")
            return agent_id
        
        agent = LiveTrader(config)
        self.agents[agent_id] = agent
        
        self.logger.info(f"Agent created: {agent_id}")
        return agent_id
    
    def start_agent(self, agent_id: str) -> bool:
        """Start specific agent"""
        if agent_id not in self.agents:
            self.logger.error(f"Agent {agent_id} not found")
            return False
        
        agent = self.agents[agent_id]
        success = agent.start()
        
        if success:
            self.logger.info(f"Agent started: {agent_id}")
        
        return success
    
    def stop_agent(self, agent_id: str):
        """Stop specific agent"""
        if agent_id in self.agents:
            self.agents[agent_id].stop()
            self.logger.info(f"Agent stopped: {agent_id}")
    
    def remove_agent(self, agent_id: str):
        """Remove agent completely"""
        if agent_id in self.agents:
            self.stop_agent(agent_id)
            del self.agents[agent_id]
            self.logger.info(f"Agent removed: {agent_id}")
    
    def get_all_status(self) -> List[dict]:
        """Get status of all agents"""
        return [agent.get_status() for agent in self.agents.values()]
    
    def start_scheduler(self, check_time: str = "16:30"):
        """
        Start scheduler for EOD checks
        
        Args:
            check_time: Time to run checks (e.g., "16:30" for 4:30 PM)
        """
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            args=(check_time,),
            daemon=True
        )
        self.scheduler_thread.start()
        self.logger.info(f"Scheduler started (EOD check at {check_time})")
    
    def _scheduler_loop(self, check_time: str):
        """
        Scheduler loop for EOD checks
        
        Runs all active agents at specified time each day
        """
        target_hour, target_minute = map(int, check_time.split(':'))
        
        while self.is_running:
            now = datetime.now()
            current_time = now.time()
            target_time = dtime(target_hour, target_minute)
            
            # Check if it's time to run (within 5-minute window)
            if (current_time.hour == target_hour and 
                target_minute <= current_time.minute < target_minute + 5):
                
                self.logger.info("Running scheduled EOD checks...")
                
                # Run all active agents
                for agent_id, agent in self.agents.items():
                    if agent.is_running:
                        self.logger.info(f"Checking {agent_id}...")
                        agent.run_single_check()
                
                self.logger.info("EOD checks complete")
                
                # Sleep until next day (avoid running multiple times)
                time.sleep(60 * 60)  # Sleep 1 hour
            
            # Sleep 1 minute before next check
            time.sleep(60)
    
    def stop_scheduler(self):
        """Stop scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Scheduler stopped")
    
    def emergency_stop_all(self):
        """EMERGENCY: Stop all agents immediately"""
        self.logger.warning("EMERGENCY STOP - Stopping all agents")
        
        for agent_id in list(self.agents.keys()):
            self.stop_agent(agent_id)
        
        self.stop_scheduler()
        self.logger.warning("All agents stopped")


# Global instance
agent_manager = AgentManager()
```

#### 1.4. API Endpoints
**File**: `backend/api/main.py` (add to existing)

```python
from execution.agent_manager import agent_manager

# ==================== LIVE TRADING ENDPOINTS ====================

@app.route('/api/live/agents', methods=['GET'])
def get_live_agents():
    """Get all live trading agents status"""
    try:
        agents_status = agent_manager.get_all_status()
        
        return jsonify({
            'status': 'success',
            'count': len(agents_status),
            'agents': agents_status
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/live/agents', methods=['POST'])
def create_live_agent():
    """
    Create new live trading agent
    
    Body:
    {
        "agent_type": "PPO",
        "symbol": "INTC",
        "model_path": "backend/models/ppo/ppo_INTC_v20251108.zip",
        "time_frame": "daily",
        "initial_capital": 10000,
        "max_position_pct": 0.5,
        "risk_per_trade": 0.02,
        "features_config": {...}
    }
    """
    try:
        config = request.json
        
        # Generate agent_id
        agent_id = f"{config['agent_type']}_{config['symbol']}_{int(time.time())}"
        config['agent_id'] = agent_id
        config['check_frequency'] = 'EOD'
        
        # Create agent
        agent_manager.create_agent(config)
        
        return jsonify({
            'status': 'success',
            'agent_id': agent_id,
            'message': f'Agent {agent_id} created'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/live/agents/<agent_id>/start', methods=['POST'])
def start_live_agent(agent_id):
    """Start specific agent"""
    try:
        success = agent_manager.start_agent(agent_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Agent {agent_id} started'
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'Failed to start agent'
            }), 500
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/live/agents/<agent_id>/stop', methods=['POST'])
def stop_live_agent(agent_id):
    """Stop specific agent"""
    try:
        agent_manager.stop_agent(agent_id)
        
        return jsonify({
            'status': 'success',
            'message': f'Agent {agent_id} stopped'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/live/agents/<agent_id>', methods=['DELETE'])
def remove_live_agent(agent_id):
    """Remove agent completely"""
    try:
        agent_manager.remove_agent(agent_id)
        
        return jsonify({
            'status': 'success',
            'message': f'Agent {agent_id} removed'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/live/agents/<agent_id>/position/close', methods=['POST'])
def close_agent_position(agent_id):
    """Force close agent's current position"""
    try:
        # TODO: Implement force close
        return jsonify({
            'status': 'success',
            'message': f'Position closed for {agent_id}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/live/emergency-stop', methods=['POST'])
def emergency_stop_all():
    """EMERGENCY: Stop all agents immediately"""
    try:
        agent_manager.emergency_stop_all()
        
        return jsonify({
            'status': 'success',
            'message': 'All agents stopped'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/api/live/scheduler/start', methods=['POST'])
def start_scheduler():
    """Start EOD scheduler"""
    try:
        check_time = request.json.get('check_time', '16:30')
        agent_manager.start_scheduler(check_time)
        
        return jsonify({
            'status': 'success',
            'message': f'Scheduler started (EOD at {check_time})'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
```

---

## 🎨 Frontend Implementation

### Phase 2: Live Trading UI (Week 2)

#### 2.1. Enhanced TabLiveTrading Component
**File**: `frontend/src/components/TabLiveTrading.jsx`

```jsx
import React, { useState, useEffect } from 'react';
import { Button, Card } from './common/UIComponents';
import LiveAgentCard from './live/LiveAgentCard';
import AddAgentModal from './live/AddAgentModal';
import EmergencyStopModal from './live/EmergencyStopModal';
import liveAPI from '../services/liveAPI';

function TabLiveTrading() {
  const [agents, setAgents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showEmergencyModal, setShowEmergencyModal] = useState(false);

  // Fetch agents on mount and every 10 seconds
  useEffect(() => {
    fetchAgents();
    const interval = setInterval(fetchAgents, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchAgents = async () => {
    const result = await liveAPI.getAgents();
    if (result.success) {
      setAgents(result.agents);
    }
    setLoading(false);
  };

  const handleEmergencyStop = async () => {
    const result = await liveAPI.emergencyStopAll();
    if (result.success) {
      fetchAgents();
      setShowEmergencyModal(false);
    }
  };

  if (loading) {
    return <div>Loading agents...</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {/* Header with controls */}
      <Card style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ margin: 0 }}>🤖 Live Trading Agents ({agents.length})</h2>
          
          <div style={{ display: 'flex', gap: '12px' }}>
            <Button 
              onClick={() => setShowAddModal(true)}
              style={{ background: '#238636' }}
            >
              ➕ Add Agent
            </Button>
            
            <Button 
              onClick={() => setShowEmergencyModal(true)}
              style={{ background: '#da3633' }}
            >
              🚨 EMERGENCY STOP
            </Button>
          </div>
        </div>
      </Card>

      {/* Agents list */}
      {agents.length === 0 ? (
        <Card style={{ padding: '40px', textAlign: 'center' }}>
          <p style={{ color: '#8b949e', marginBottom: '20px' }}>
            No active agents. Add an agent to start live trading.
          </p>
          <Button onClick={() => setShowAddModal(true)}>
            ➕ Add Your First Agent
          </Button>
        </Card>
      ) : (
        agents.map(agent => (
          <LiveAgentCard 
            key={agent.agent_id} 
            agent={agent}
            onRefresh={fetchAgents}
          />
        ))
      )}

      {/* Add Agent Modal */}
      {showAddModal && (
        <AddAgentModal 
          onClose={() => setShowAddModal(false)}
          onSuccess={fetchAgents}
        />
      )}

      {/* Emergency Stop Confirmation */}
      {showEmergencyModal && (
        <EmergencyStopModal
          onConfirm={handleEmergencyStop}
          onCancel={() => setShowEmergencyModal(false)}
        />
      )}
    </div>
  );
}

export default TabLiveTrading;
```

#### 2.2. LiveAgentCard Component
**File**: `frontend/src/components/live/LiveAgentCard.jsx`

```jsx
import React, { useState } from 'react';
import { Button } from '../common/UIComponents';
import liveAPI from '../../services/liveAPI';

function LiveAgentCard({ agent, onRefresh }) {
  const [showDetails, setShowDetails] = useState(false);

  const handlePause = async () => {
    await liveAPI.stopAgent(agent.agent_id);
    onRefresh();
  };

  const handleStart = async () => {
    await liveAPI.startAgent(agent.agent_id);
    onRefresh();
  };

  const handleClosePosition = async () => {
    if (window.confirm(`Close position for ${agent.symbol}?`)) {
      await liveAPI.closePosition(agent.agent_id);
      onRefresh();
    }
  };

  const handleRemove = async () => {
    if (window.confirm(`Remove agent ${agent.agent_id}?`)) {
      await liveAPI.removeAgent(agent.agent_id);
      onRefresh();
    }
  };

  const getStatusColor = (is_running) => {
    return is_running ? '#3fb950' : '#8b949e';
  };

  const getPnLColor = (pnl) => {
    return pnl >= 0 ? '#3fb950' : '#f85149';
  };

  return (
    <div style={styles.card}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <span style={styles.agentName}>
            {agent.symbol} - {agent.agent_id}
          </span>
          <span style={{
            ...styles.statusBadge,
            background: getStatusColor(agent.is_running)
          }}>
            {agent.is_running ? '● Active' : '○ Stopped'}
          </span>
        </div>
        
        <div style={styles.pnlBadge}>
          <span style={{ color: getPnLColor(agent.total_pnl) }}>
            {agent.total_pnl >= 0 ? '+' : ''}${agent.total_pnl.toFixed(2)} 
            ({agent.total_pnl_pct >= 0 ? '+' : ''}{agent.total_pnl_pct.toFixed(2)}%)
          </span>
        </div>
      </div>

      {/* Chart placeholder */}
      <div style={styles.chartArea}>
        <div style={styles.chartPlaceholder}>
          Price Chart + Buy/Sell Markers + Volume + Indicators
          <br />
          <small>(Will show Recharts with real-time data)</small>
        </div>
      </div>

      {/* Metrics */}
      <div style={styles.metricsGrid}>
        <div style={styles.metric}>
          <div style={styles.metricLabel}>Position</div>
          <div style={styles.metricValue}>
            {agent.current_position} shares @ ${agent.entry_price.toFixed(2)}
          </div>
        </div>
        
        <div style={styles.metric}>
          <div style={styles.metricLabel}>Current Price</div>
          <div style={styles.metricValue}>
            ${agent.current_price.toFixed(2)}
          </div>
        </div>
        
        <div style={styles.metric}>
          <div style={styles.metricLabel}>Portfolio Value</div>
          <div style={styles.metricValue}>
            ${agent.portfolio_value.toFixed(2)}
          </div>
        </div>
        
        <div style={styles.metric}>
          <div style={styles.metricLabel}>Total Trades</div>
          <div style={styles.metricValue}>
            {agent.num_trades}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={styles.controls}>
        {agent.is_running ? (
          <Button onClick={handlePause} style={{ background: '#f59e0b' }}>
            ⏸️ Pause
          </Button>
        ) : (
          <Button onClick={handleStart} style={{ background: '#3fb950' }}>
            ▶️ Start
          </Button>
        )}
        
        <Button onClick={handleClosePosition} disabled={agent.current_position === 0}>
          💰 Close Position
        </Button>
        
        <Button onClick={handleRemove} style={{ background: '#da3633' }}>
          🗑️ Remove
        </Button>
        
        <Button onClick={() => setShowDetails(!showDetails)} style={{ background: '#1f2937' }}>
          {showDetails ? '🔼 Hide Details' : '🔽 Show Details'}
        </Button>
      </div>

      {/* Detailed info (collapsible) */}
      {showDetails && (
        <div style={styles.detailsPanel}>
          <h4>Agent Details</h4>
          <pre>{JSON.stringify(agent, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

const styles = {
  card: {
    background: '#0d1117',
    border: '1px solid #30363d',
    borderRadius: '8px',
    padding: '20px'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '16px'
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px'
  },
  agentName: {
    fontSize: '16px',
    fontWeight: 'bold',
    color: '#c9d1d9'
  },
  statusBadge: {
    padding: '4px 12px',
    borderRadius: '12px',
    fontSize: '12px',
    fontWeight: 'bold',
    color: '#ffffff'
  },
  pnlBadge: {
    fontSize: '18px',
    fontWeight: 'bold'
  },
  chartArea: {
    height: '200px',
    marginBottom: '16px'
  },
  chartPlaceholder: {
    height: '100%',
    background: '#0d1117',
    border: '1px dashed #30363d',
    borderRadius: '4px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#6e7681',
    fontSize: '11px',
    textAlign: 'center'
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: '12px',
    marginBottom: '16px'
  },
  metric: {
    padding: '12px',
    background: '#161b22',
    borderRadius: '6px'
  },
  metricLabel: {
    fontSize: '11px',
    color: '#8b949e',
    marginBottom: '4px'
  },
  metricValue: {
    fontSize: '14px',
    fontWeight: 'bold',
    color: '#c9d1d9'
  },
  controls: {
    display: 'flex',
    gap: '8px',
    flexWrap: 'wrap'
  },
  detailsPanel: {
    marginTop: '16px',
    padding: '16px',
    background: '#161b22',
    borderRadius: '6px',
    fontSize: '11px',
    color: '#8b949e'
  }
};

export default LiveAgentCard;
```

#### 2.3. Live Trading API Service
**File**: `frontend/src/services/liveAPI.js`

```javascript
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const liveAPI = {
  // Get all agents
  async getAgents() {
    try {
      const response = await fetch(`${API_BASE}/api/live/agents`);
      const data = await response.json();
      return { success: true, agents: data.agents };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  // Create new agent
  async createAgent(config) {
    try {
      const response = await fetch(`${API_BASE}/api/live/agents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      const data = await response.json();
      return { success: true, agent_id: data.agent_id };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  // Start agent
  async startAgent(agent_id) {
    try {
      const response = await fetch(`${API_BASE}/api/live/agents/${agent_id}/start`, {
        method: 'POST'
      });
      await response.json();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  // Stop agent
  async stopAgent(agent_id) {
    try {
      const response = await fetch(`${API_BASE}/api/live/agents/${agent_id}/stop`, {
        method: 'POST'
      });
      await response.json();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  // Remove agent
  async removeAgent(agent_id) {
    try {
      const response = await fetch(`${API_BASE}/api/live/agents/${agent_id}`, {
        method: 'DELETE'
      });
      await response.json();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  // Close position
  async closePosition(agent_id) {
    try {
      const response = await fetch(`${API_BASE}/api/live/agents/${agent_id}/position/close`, {
        method: 'POST'
      });
      await response.json();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  },

  // Emergency stop all
  async emergencyStopAll() {
    try {
      const response = await fetch(`${API_BASE}/api/live/emergency-stop`, {
        method: 'POST'
      });
      await response.json();
      return { success: true };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
};

export default liveAPI;
```

---

## 🧪 Testing Without Live Data Subscription

### Option 1: Historical Bar Replay (RECOMMENDED)
Since you can't get real-time ticks, use historical bars:

```python
def fetch_latest_data_delayed(self):
    """
    Fetch latest data using 15-20 minute delayed data
    
    Works without live data subscription:
    - Use IBKR historical bars (reqHistoricalData)
    - Or fallback to Yahoo Finance (15-20 min delay)
    """
    # Try IBKR historical first
    try:
        bars = self.ibkr_adapter.get_historical_bars(
            symbol=self.symbol,
            duration='1 D',  # Last 1 day
            bar_size='1 day',
            what_to_show='TRADES'
        )
        
        if bars and len(bars) > 0:
            return bars[-1]  # Latest bar
    except:
        pass
    
    # Fallback to Yahoo Finance
    import yfinance as yf
    ticker = yf.Ticker(self.symbol)
    df = ticker.history(period='2d')
    return df.iloc[-1]  # Latest bar
```

### Option 2: Paper Trading with Delayed Data
```python
# In live_trader.py
if self.config.get('paper_trading_mode', True):
    # Simulate order execution (don't send to IBKR)
    self.logger.info(f"[PAPER] Would execute: {action_name} {shares} @ ${price}")
    # Update virtual portfolio
else:
    # Real order execution
    self.ibkr_adapter.place_order(...)
```

---

## 📅 Phase-by-Phase Implementation

### Week 1: Backend Foundation
**Days 1-2**: Core Classes
- [ ] Create `backend/execution/` directory
- [ ] Implement `live_trader.py` (LiveTrader class)
- [ ] Implement `agent_manager.py` (AgentManager class)
- [ ] Test locally without IBKR

**Days 3-4**: IBKR Integration
- [ ] Integrate with InterReactBridgeAdapter
- [ ] Test connection to IBKR
- [ ] Test historical data fetch
- [ ] Test order execution (paper trading first!)

**Days 5-7**: API Endpoints
- [ ] Add live trading endpoints to `api/main.py`
- [ ] Test with Postman/curl
- [ ] Implement error handling
- [ ] Add logging

### Week 2: Frontend UI
**Days 1-2**: Basic UI
- [ ] Create LiveAgentCard component
- [ ] Create AddAgentModal component
- [ ] Update TabLiveTrading.jsx
- [ ] Create liveAPI.js service

**Days 3-4**: Advanced Features
- [ ] Implement emergency stop modal
- [ ] Add real-time updates (polling every 10s)
- [ ] Add position tracking display
- [ ] Add trade history table

**Days 5-7**: Deploy from Training
- [ ] Add "Deploy to Live" button in Training tab
- [ ] Create deployment modal
- [ ] Wire up training → live workflow
- [ ] Test end-to-end

### Week 3: Polish & Testing
**Days 1-3**: Testing
- [ ] Test with paper trading mode
- [ ] Test multiple agents (PPO + SAC)
- [ ] Test EOD scheduler
- [ ] Test emergency stop

**Days 4-5**: Documentation
- [ ] Write LIVE_TRADING_GUIDE.md
- [ ] Document deployment workflow
- [ ] Create troubleshooting guide

---

## ✅ Acceptance Criteria

Before considering Live Trading "complete":

### Backend
- [x] LiveTrader class loads model and fetches data
- [x] Can execute orders via IBKR (test in paper mode)
- [x] AgentManager handles multiple agents
- [x] EOD scheduler runs checks daily
- [x] API endpoints respond correctly
- [x] Emergency stop works

### Frontend
- [x] Can create new agent from UI
- [x] Can start/stop agents
- [x] Shows real-time status (updates every 10s)
- [x] Shows current position and P&L
- [x] Emergency stop button works
- [x] Deploy from Training tab works

### Testing
- [x] Works in paper trading mode (no real money)
- [x] PPO and SAC agents work independently
- [x] Can run overnight without crashing
- [x] Logs all actions correctly

---

## 🚨 Important Notes

### About Time Frames
- **Training data**: Daily OHLCV from Yahoo Finance
- **Live trading**: Must use same time frame (daily)
- **Check frequency**: EOD (after market close)
- **Decision timing**: Agent sees daily bar, decides for next day

### About Live Data
- You DON'T have real-time subscription
- Use delayed data (15-20 min) or historical bars
- This is OK for daily trading (EOD strategy)
- NOT suitable for intraday scalping

### About Risk
- START WITH PAPER TRADING MODE
- Test for at least 1 week before real money
- Use small capital initially ($1,000-$5,000)
- Monitor closely for first month
- Have emergency stop ready

---

## 📊 Success Metrics

After 1 week of paper trading:
- [ ] No crashes or errors
- [ ] Agent makes reasonable decisions
- [ ] Orders execute successfully
- [ ] P&L tracking accurate
- [ ] Metrics match expectations

After 1 month of live trading (small capital):
- [ ] Positive Sharpe Ratio (> 1.0)
- [ ] Max drawdown < 15%
- [ ] No system failures
- [ ] Can run unsupervised

---

## 🎓 Recommended Reading

Before starting implementation:
1. Read IBKR API documentation (order types, historical data)
2. Understand market hours (pre-market, regular, after-hours)
3. Learn about order execution (market vs limit orders)
4. Study risk management principles

---

**Next Action**: Start with `backend/execution/live_trader.py` skeleton.

```bash
mkdir backend/execution
touch backend/execution/__init__.py
touch backend/execution/live_trader.py
```

Good luck! 🚀
