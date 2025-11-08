# InterReactBridge - C# IBKR Bridge Server

## Quick Start

### Requirements
- .NET 8.0 SDK
- IBKR TWS or Gateway (running with API enabled)

### Build & Run

```powershell
# Build
dotnet build

# Run
dotnet run

# Or for production
dotnet run --configuration Release
```

Server will start on: `http://localhost:5080`

### Test Health
```powershell
curl http://localhost:5080/health
```

Expected response:
```json
{"status":"Healthy"}
```

### Configuration

Edit `appsettings.json`:
```json
{
  "Urls": "http://localhost:5080",
  "Logging": {
    "LogLevel": {
      "Default": "Information"
    }
  }
}
```

### API Documentation

Once running, visit: `http://localhost:5080/swagger`

---

For full documentation, see the main README.md in the parent folder.
