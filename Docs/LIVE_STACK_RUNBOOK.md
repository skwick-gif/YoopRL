# Live Stack Runbook
> File Note: Describes operational sequence for bridge, backend, scheduler, and UI.

## Startup Order
1. Launch Interactive Brokers TWS or Gateway and confirm paper-trading login.
2. Run `start_bridge.bat` (opens the InterReact bridge on port 5080).
3. Execute `start_backend.bat`; the backend attaches to the bridge and registers routes.
4. Start the frontend (`npm start` inside `frontend/`) and open the Live Data tab.
5. Verify the scheduler is active by checking system logs (component `LIVE_SCHEDULER`).

## Health Checks
- Run `python backend/scripts/validate_live_stack.py` to confirm bridge, backend, database, and PyQt6 availability.
- Monitor `/api/monitoring/summary` for recent actions and alerts.

## Paper-Trading Dry Run
- Deploy a model from the Training tab using the 15-minute template.
- In the Live tab, create an agent with `start_immediately=true` and observe scheduled checks.
- Confirm `agent_actions` and `risk_events` tables receive entries.
- Test controls: `Run Once`, `Start`, `Stop`, and `Emergency Stop`.

## Shutdown
1. Stop all live agents via the Live tab.
2. Close the frontend and backend terminal windows.
3. Stop the bridge process.
4. Log out of TWS/Gateway.
