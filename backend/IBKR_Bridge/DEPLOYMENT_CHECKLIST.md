# IBKR Bridge Deployment Checklist

Use this checklist when deploying to a new computer.

## Pre-Deployment

- [ ] Source computer has all files copied
- [ ] Folder size is reasonable (< 100 MB without build artifacts)
- [ ] All source files are readable

## Target Computer Setup

- [ ] .NET SDK 6.0+ installed: `dotnet --version`
- [ ] Python 3.9+ installed: `python --version`
- [ ] Git installed (optional, for version control)

## C# Bridge Deployment

- [ ] Navigate to `csharp_bridge` folder
- [ ] Run: `dotnet restore`
  - [ ] All NuGet packages downloaded successfully
- [ ] Run: `dotnet build`
  - [ ] Build completes with no errors
- [ ] Review `appsettings.json` and update if needed
  - [ ] Port is accessible (default: 5000)
  - [ ] Logging level is appropriate
- [ ] Run: `dotnet run`
  - [ ] Bridge starts successfully
  - [ ] Message shows: "Now listening on: http://localhost:5000"

## Python Adapter Deployment

- [ ] Open new terminal/command prompt
- [ ] Navigate to `python_adapter` folder
- [ ] Install dependencies: `pip install dash requests websocket-client`
- [ ] Verify adapter can connect:
  ```bash
  python -c "import requests; print(requests.get('http://localhost:5000').status_code)"
  ```
  - [ ] Response shows 200 or similar (not connection error)

## Integration Testing

- [ ] Test C# Bridge REST API:
  ```bash
  curl http://localhost:5000/api/accounts
  ```
  - [ ] Returns data (or appropriate response)

- [ ] Test WebSocket connection:
  ```bash
  python interreact_bridge_adapter.py
  ```
  - [ ] Adapter connects without errors
  - [ ] Can send/receive test messages

## Optional: Remote Access

If accessing from different computer on same network:

- [ ] Update Python connection URL to remote IP:
  ```python
  adapter = IBKRAdapter(
      bridge_url="http://REMOTE_IP:5000",
      ws_url="ws://REMOTE_IP:5000/hub/trading"
  )
  ```
- [ ] Test connection from client machine
- [ ] Verify firewall allows port 5000 (or configured port)

## Documentation Review

- [ ] Read `csharp_bridge/README.md` for C# specifics
- [ ] Read `python_adapter/QUICK_START.md` for Python integration
- [ ] Review `SETUP_INSTRUCTIONS.md` for detailed steps
- [ ] Check `CHANGELOG.md` for version information

## Final Validation

- [ ] Both C# bridge and Python adapter running
- [ ] Connection established without errors
- [ ] Ready to integrate with trading interface

## Troubleshooting Commands

```bash
# Check .NET version
dotnet --version

# Clean and rebuild
cd csharp_bridge
dotnet clean
dotnet build

# Check if port is in use (Windows)
netstat -ano | findstr :5000

# Test HTTP connection (Windows)
powershell -Command "(Invoke-WebRequest -Uri http://localhost:5000).StatusCode"

# Python connection test
python -c "from interreact_bridge_adapter import IBKRAdapter; print('OK')"
```

## Notes

- Build artifacts (bin/, obj/) will be created on first build
- Log files are generated in `csharp_bridge/bin/` during runtime
- Python packages will be installed in your Python environment
- Keep this checklist for future deployments

---

**Next Step:** Start with "Target Computer Setup" section
