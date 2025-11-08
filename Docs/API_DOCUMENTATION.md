# API Documentation - Training Module

Complete list of all training API endpoints with examples.

---

## Base URL

```
http://localhost:5000/api/training
```

---

## Endpoints

### 1. Start Training

**POST** `/train`

Starts training a new model in the background.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "agent_type": "PPO",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "total_timesteps": 100000,
  "hyperparameters": {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 64
  },
  "features": ["close", "volume", "rsi", "macd"],
  "normalization_method": "zscore"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Training started"
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid input
- `500` - Server error

**Example:**
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

### 2. Get Training Progress

**GET** `/progress/{session_id}`

Get real-time training progress for a session.

**URL Parameters:**
- `session_id` - Session ID from start training

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "running",
    "progress": {
      "current_step": 45000,
      "total_steps": 100000,
      "percent_complete": 45,
      "episode": 150,
      "episode_reward": 0.082,
      "mean_reward": 0.065,
      "episode_length": 300,
      "loss": 0.0023,
      "sharpe_ratio": 1.45,
      "elapsed_time": 420
    }
  }
}
```

**Status Values:**
- `running` - Training in progress
- `completed` - Training finished
- `stopped` - User stopped training
- `error` - Training failed

**Example:**
```bash
curl http://localhost:5000/api/training/progress/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

---

### 3. Stop Training

**POST** `/stop`

Stops a running training session.

**Request Body:**
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Training stopped"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/training/stop \
  -H "Content-Type: application/json" \
  -d '{"session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}'
```

---

### 4. List Models

**GET** `/models`

Get list of saved models with metadata.

**Query Parameters (optional):**
- `agent_type` - Filter by agent type (PPO/SAC)
- `symbol` - Filter by symbol
- `limit` - Max number of results (default: 50)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "model_id": "ppo_AAPL_v003_20251108",
      "agent_type": "PPO",
      "symbol": "AAPL",
      "version": 3,
      "created_at": "2025-11-08 14:30:00",
      "metrics": {
        "sharpe_ratio": 1.85,
        "total_return": 0.42,
        "max_drawdown": -0.15,
        "win_rate": 0.58
      },
      "config": { ... }
    },
    { ... }
  ]
}
```

**Example:**
```bash
# All models
curl http://localhost:5000/api/training/models

# Filter by symbol
curl http://localhost:5000/api/training/models?symbol=AAPL

# Filter by agent type
curl http://localhost:5000/api/training/models?agent_type=PPO
```

---

### 5. Load Model

**POST** `/load_model`

Load a saved model for inference or backtesting.

**Request Body:**
```json
{
  "model_id": "ppo_AAPL_v003_20251108"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Model loaded successfully",
  "data": {
    "model_id": "ppo_AAPL_v003_20251108",
    "agent_type": "PPO",
    "symbol": "AAPL",
    "metrics": { ... }
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/training/load_model \
  -H "Content-Type: application/json" \
  -d '{"model_id": "ppo_AAPL_v003_20251108"}'
```

---

### 6. Delete Model

**DELETE** `/models/{model_id}`

Delete a saved model and its metadata.

**URL Parameters:**
- `model_id` - Model ID to delete

**Response:**
```json
{
  "success": true,
  "message": "Model deleted"
}
```

**Example:**
```bash
curl -X DELETE http://localhost:5000/api/training/models/ppo_AAPL_v001_20251101
```

---

### 7. Run Backtest

**POST** `/backtest`

Run backtest on a trained model with test data.

**Request Body:**
```json
{
  "model_id": "ppo_AAPL_v003_20251108",
  "test_start_date": "2024-01-01",
  "test_end_date": "2024-12-31",
  "initial_balance": 10000,
  "save_results": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "agent_metrics": {
      "sharpe_ratio": 1.92,
      "sortino_ratio": 2.18,
      "max_drawdown": -0.12,
      "win_rate": 0.61,
      "profit_factor": 1.58,
      "total_return": 0.38,
      "calmar_ratio": 3.17,
      "alpha": 0.15
    },
    "buy_and_hold_metrics": {
      "sharpe_ratio": 1.45,
      "total_return": 0.28,
      "max_drawdown": -0.18
    },
    "comparison": {
      "sharpe_improvement": 0.47,
      "return_improvement": 0.10
    },
    "trades": [
      {
        "date": "2024-01-05",
        "action": "BUY",
        "price": 150.25,
        "shares": 50,
        "portfolio_value": 10050
      },
      { ... }
    ],
    "portfolio_values": [10000, 10050, 10100, ...]
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/training/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "ppo_AAPL_v003_20251108",
    "test_start_date": "2024-01-01",
    "test_end_date": "2024-12-31"
  }'
```

---

### 8. Check Drift Status

**GET** `/drift_status`

Check if market distribution has drifted since training.

**Query Parameters:**
- `symbol` - Stock symbol (required)
- `agent_type` - Agent type (default: PPO)
- `days` - Days of recent data to check (default: 30)

**Response:**
```json
{
  "success": true,
  "data": {
    "drift_detected": true,
    "severity": "high",
    "affected_features": ["rsi", "volume"],
    "drift_scores": {
      "rsi": 0.68,
      "volume": 0.55,
      "close": 0.32
    },
    "needs_retraining": true,
    "recommendation": "High drift detected in 2 features. Retraining recommended within 7 days."
  }
}
```

**Severity Levels:**
- `low` (0-0.4) - No action needed
- `medium` (0.4-0.5) - Monitor closely
- `high` (0.5-0.7) - Consider retraining
- `critical` (>0.7) - Retrain immediately

**Example:**
```bash
curl "http://localhost:5000/api/training/drift_status?symbol=AAPL&days=30"
```

---

### 9. Save Config

**POST** `/config/save`

Save training configuration for reuse.

**Request Body:**
```json
{
  "config_name": "my_conservative_config",
  "config": {
    "agent_type": "PPO",
    "hyperparameters": {
      "learning_rate": 0.0001,
      "gamma": 0.99
    },
    "features": ["close", "volume", "rsi"],
    "normalization_method": "zscore"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Config saved",
  "config_id": "my_conservative_config"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/training/config/save \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "aggressive_etf",
    "config": {...}
  }'
```

---

### 10. Load Config

**GET** `/config/load`

Load a saved configuration.

**Query Parameters:**
- `config_name` - Name of config to load

**Response:**
```json
{
  "success": true,
  "data": {
    "config_name": "my_conservative_config",
    "config": {
      "agent_type": "PPO",
      "hyperparameters": { ... },
      "features": [ ... ]
    }
  }
}
```

**Example:**
```bash
curl "http://localhost:5000/api/training/config/load?config_name=my_conservative_config"
```

---

## Error Responses

All endpoints return consistent error format:

```json
{
  "success": false,
  "error": "Description of what went wrong"
}
```

**Common Error Codes:**
- `400` - Bad request (invalid input)
- `404` - Not found (session/model doesn't exist)
- `500` - Internal server error

---

## Rate Limiting

**Current:** No rate limiting (development mode)

**Production:** Implement rate limiting:
- Training: 10 requests/hour per IP
- Other endpoints: 100 requests/minute per IP

---

## Authentication

**Current:** No authentication (development mode)

**Production:** Add API key authentication:
```bash
curl http://localhost:5000/api/training/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## CORS

**Current:** Open CORS (allows all origins)

**Production:** Restrict to your domain:
```python
CORS(app, origins=["https://yourdomain.com"])
```

---

## WebSocket (Future)

For real-time training updates, consider WebSocket:
```
ws://localhost:5000/ws/training/{session_id}
```

Currently using HTTP polling (every 5 seconds).

---

## Testing Endpoints

**Simple health check:**
```bash
curl http://localhost:5000/api/training/models
```

Should return list of models (or empty array if none).

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:5000/api/training"

# Start training
response = requests.post(f"{BASE_URL}/train", json={
    "symbol": "AAPL",
    "agent_type": "PPO",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "total_timesteps": 100000
})
session_id = response.json()["session_id"]

# Poll progress
import time
while True:
    progress = requests.get(f"{BASE_URL}/progress/{session_id}").json()
    status = progress["data"]["status"]
    
    if status == "completed":
        print("Training done!")
        break
    elif status == "error":
        print("Training failed!")
        break
    
    print(f"Progress: {progress['data']['progress']['percent_complete']}%")
    time.sleep(5)

# List models
models = requests.get(f"{BASE_URL}/models?symbol=AAPL").json()
print(f"Found {len(models['data'])} models")

# Run backtest
backtest = requests.post(f"{BASE_URL}/backtest", json={
    "model_id": models["data"][0]["model_id"],
    "test_start_date": "2024-01-01",
    "test_end_date": "2024-12-31"
}).json()
print(f"Sharpe: {backtest['data']['agent_metrics']['sharpe_ratio']}")
```

---

## JavaScript Client Example

```javascript
const BASE_URL = "http://localhost:5000/api/training";

// Start training
async function trainModel() {
  const response = await fetch(`${BASE_URL}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      symbol: "AAPL",
      agent_type: "PPO",
      start_date: "2020-01-01",
      end_date: "2023-12-31",
      total_timesteps: 100000
    })
  });
  
  const data = await response.json();
  return data.session_id;
}

// Poll progress
async function pollProgress(sessionId) {
  const response = await fetch(`${BASE_URL}/progress/${sessionId}`);
  const data = await response.json();
  return data.data;
}

// Use it
const sessionId = await trainModel();
const interval = setInterval(async () => {
  const progress = await pollProgress(sessionId);
  
  if (progress.status === "completed") {
    clearInterval(interval);
    console.log("Training complete!");
  } else {
    console.log(`Progress: ${progress.progress.percent_complete}%`);
  }
}, 5000);
```

---

**Need more details?** Check the source code in `backend/api/main.py` - it's well commented.
