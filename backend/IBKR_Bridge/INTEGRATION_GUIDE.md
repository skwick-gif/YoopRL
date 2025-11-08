# IBKR Bridge Portable - Quick Integration Guide

This guide shows how to integrate the IBKR Bridge into your trading application on a new computer.

## 5-Minute Setup

### Step 1: Start C# Bridge (Terminal 1)
```bash
cd IBKR_Bridge_Portable/csharp_bridge
dotnet restore
dotnet run
```

### Step 2: Start Python Adapter (Terminal 2)
```bash
cd IBKR_Bridge_Portable/python_adapter
pip install -r requirements.txt
python interreact_bridge_adapter.py
```

### Step 3: Connect from Your Application

**Python Application:**
```python
from interreact_bridge_adapter import IBKRAdapter

# Initialize connection
adapter = IBKRAdapter(
    bridge_url="http://localhost:5000",
    ws_url="ws://localhost:5000/hub/trading"
)

# Get accounts
accounts = adapter.get_accounts()

# Get positions
positions = adapter.get_positions()

# Place order
order = adapter.place_order(
    contract_id="AAPL",
    action="BUY",
    quantity=100,
    price=150.00
)
```

## Connection Architecture

```
Your Application
    ↓
Python Adapter (WebSocket)
    ↓
C# IBKR Bridge (localhost:5000)
    ↓
IBKR / InterReact / Paper Trading Account
```

## Configuration

### Change Bridge Port (if 5000 is taken)

Edit `csharp_bridge/appsettings.json`:
```json
"Kestrel": {
  "Endpoints": {
    "Http": {
      "Url": "http://localhost:5001"  // Change to 5001
    }
  }
}
```

Then update Python connection:
```python
adapter = IBKRAdapter(
    bridge_url="http://localhost:5001",
    ws_url="ws://localhost:5001/hub/trading"
)
```

### Remote Connection (Different Computer)

If C# Bridge is on computer with IP `192.168.1.100`:

```python
adapter = IBKRAdapter(
    bridge_url="http://192.168.1.100:5000",
    ws_url="ws://192.168.1.100:5000/hub/trading"
)
```

**Note:** Ensure firewall allows port 5000 on the bridge computer.

## API Endpoints

### REST API (C# Bridge)

```
GET /api/accounts
    Response: List of trading accounts

GET /api/positions
    Response: Current positions

GET /api/orders
    Response: Open orders

POST /api/orders
    Body: { action, contract, quantity, price }
    Response: Order confirmation

GET /api/status
    Response: Bridge status and connection info
```

### WebSocket (Real-time)

Connected via Python adapter automatically. Receives:
- Real-time price updates
- Position changes
- Order executions
- Account updates

## Troubleshooting

### Bridge won't start
```bash
# Check .NET version
dotnet --version

# Restore packages
dotnet restore

# Try clean build
dotnet clean
dotnet build
```

### Python adapter can't connect
```bash
# Test bridge is running
curl http://localhost:5000/api/status

# Check firewall allows port 5000
netstat -ano | findstr :5000

# Test Python connection
python -c "import requests; print(requests.get('http://localhost:5000/api/status').json())"
```

### Port 5000 already in use
```bash
# Find what's using port 5000
netstat -ano | findstr :5000

# Change bridge port in appsettings.json to 5001-5100 and try again
```

## File Reference

| File | Purpose |
|------|---------|
| `csharp_bridge/Program.cs` | C# entry point - handles HTTP/WebSocket |
| `csharp_bridge/appsettings.json` | Bridge configuration (port, logging) |
| `python_adapter/interreact_bridge_adapter.py` | Python client for bridge |
| `python_adapter/requirements.txt` | Python dependencies |

## Example: Full Trading System

```python
import time
from interreact_bridge_adapter import IBKRAdapter

# Start bridge connection
adapter = IBKRAdapter(
    bridge_url="http://localhost:5000",
    ws_url="ws://localhost:5000/hub/trading"
)

# Check accounts
accounts = adapter.get_accounts()
print(f"Available accounts: {accounts}")

# Check positions
positions = adapter.get_positions()
print(f"Current positions: {positions}")

# Place test order
order_response = adapter.place_order(
    contract_id="AAPL",
    action="BUY",
    quantity=10,
    price=150.00
)
print(f"Order placed: {order_response}")

# Wait for updates
time.sleep(5)

# Get updated positions
updated_positions = adapter.get_positions()
print(f"Updated positions: {updated_positions}")
```

## Next Steps

1. Copy `IBKR_Bridge_Portable` folder to new computer
2. Follow "5-Minute Setup" above
3. Integrate adapter into your application
4. Test with small orders first (use paper trading)
5. Monitor logs in both terminals for issues

---

**Questions?** See the detailed `SETUP_INSTRUCTIONS.md` or `DEPLOYMENT_CHECKLIST.md`
