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

## Training Maintenance (Phase 0 Steps 6–7)
- Run the normalization smoke test before promoting a new SAC checkpoint to ensure `_normalize_feature_frames` is saving/using the latest scaler parameters. Expect a quick pass with TensorBoard `LooseVersion` warnings only.

```powershell
py -3.11 -m pytest backend\tests\test_intraday_normalization.py
```

- Verify observability by summarizing the most recent TensorBoard runs and metadata snapshot. Confirm actor/critic losses are flowing and Sharpe/return figures line up with the promoted model.

```powershell
py -3.11 scripts\analyze_training_logs.py --runs 2 --limit 50 --metadata backend\models\sac\sac_TNA_v20251114_094733_metadata.json
```

- Keep loader refreshes bounded to the walk-forward window to avoid hammering Twelve Data. When in doubt, confirm cache coverage before triggering `run_walk_forward`.

```powershell
py -3.11 scripts\tmp_dataset_check.py --symbols TNA IWM --interval 15m
```

- After any config changes, rerun the multi-window command with `tmp/mini_config.json` so the new guardrails (training start/end, capped `ensure_intraday_data_up_to`) are exercised end-to-end.

```powershell
py -3.11 -m backend.scripts.run_walk_forward --symbol TNA --config tmp\mini_config.json --benchmark IWM --interval 15m --quiet
```
