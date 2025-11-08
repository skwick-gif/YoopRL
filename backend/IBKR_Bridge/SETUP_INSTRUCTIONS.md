# IBKR Bridge - Setup Instructions for New Computer

This folder contains everything needed to run the IBKR Bridge on a new computer.

## Prerequisites

### For C# Bridge (Windows)
- **.NET SDK 6.0 or higher** - Download from: https://dotnet.microsoft.com/download
- **Visual Studio or VS Code** (optional, for development)

### For Python Adapter
- **Python 3.9 or higher**
- **pip** (Python package manager)

## Setup Steps

### Step 1: Set Up C# Bridge

```bash
cd csharp_bridge

# Restore NuGet packages
dotnet restore

# Build the project
dotnet build

# Or directly run
dotnet run
```

**Configuration:**
- Edit `appsettings.json` to set the server port (default: 5000)
- Edit `appsettings.Development.json` for development settings

### Step 2: Verify Bridge is Running

After running `dotnet run`, you should see:
```
info: Microsoft.Hosting.Lifetime[14]
      Now listening on: http://localhost:5000
```

The bridge will be available at:
- **HTTP**: http://localhost:5000
- **WebSocket**: ws://localhost:5000/hub/trading

### Step 3: Set Up Python Adapter

In a separate terminal/command prompt:

```bash
cd python_adapter

# Install dependencies
pip install dash requests websocket-client

# Run the adapter
python interreact_bridge_adapter.py
```

This will establish a connection to the C# bridge and make it available to your Python application.

## Configuration Files

### appsettings.json
```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Information"
    }
  },
  "Kestrel": {
    "Endpoints": {
      "Http": {
        "Url": "http://localhost:5000"
      }
    }
  }
}
```

Key settings:
- **Url**: Change port from 5000 if needed
- **LogLevel**: Set to "Debug" for more verbose logging

### appsettings.Development.json
Override settings for development environment

## Connecting from Another Interface

Once the bridge is running, connect from your other interface using:

**From Python:**
```python
from interreact_bridge_adapter import IBKRAdapter

adapter = IBKRAdapter(
    bridge_url="http://localhost:5000",  # Or remote IP:port
    ws_url="ws://localhost:5000/hub/trading"
)
```

**From HTTP:**
```
POST http://localhost:5000/api/accounts
GET http://localhost:5000/api/positions
POST http://localhost:5000/api/orders
```

## Troubleshooting

### Bridge won't start
1. Check .NET SDK is installed: `dotnet --version`
2. Try `dotnet restore` to fix missing packages
3. Check port 5000 is not in use: `netstat -ano | findstr :5000`

### Python adapter can't connect
1. Verify C# bridge is running
2. Check connection URL matches bridge listening address
3. Try connecting with `curl` first: `curl http://localhost:5000`

### Port already in use
Change port in `appsettings.json`:
```json
"Url": "http://localhost:5001"
```

## File Structure

```
IBKR_Bridge_Portable/
├── README.md                          # Overview
├── CHANGELOG.md                       # Version history
├── SETUP_INSTRUCTIONS.md              # This file
├── csharp_bridge/
│   ├── InterReactBridge.csproj       # Project file
│   ├── Program.cs                     # Entry point
│   ├── appsettings.json               # Configuration
│   ├── Hubs/                          # SignalR hubs
│   ├── Models/                        # Data models
│   ├── Services/                      # Business logic
│   ├── Properties/                    # Project properties
│   └── scripts/                       # Helper scripts
└── python_adapter/
    ├── interreact_bridge_adapter.py   # Python connection code
    └── QUICK_START.md                 # Quick start guide
```

## Next Steps

1. Copy this entire folder to your new computer
2. Follow the Setup Steps above
3. Test the connection by accessing the bridge URL
4. Integrate with your trading interface/application

## Support

- C# Bridge Details: See `csharp_bridge/README.md` and `csharp_bridge/STATUS_REPORT.md`
- Python Adapter Details: See `python_adapter/QUICK_START.md`
- Original IBKR Documentation: See `README.md` and `CHANGELOG.md`

---

**Note:** This portable folder contains all source code needed. Build artifacts (bin/, obj/) are excluded to reduce size. They will be regenerated when you run `dotnet build`.
