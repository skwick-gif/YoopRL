# Live Data Remediation & Intraday Enablement Plan
> File Note: Tracks remediation tasks to restore live data monitoring and ship the intraday (15-minute) pipeline.

## 1. Objective
- Restore end-to-end functionality of the Live Data tab without touching the running bridge.
- Enable 15-minute model deployments by adapting the intraday loader to our production pipeline.
- Document remaining steps required to operate smoothly in paper trading.

## 2. Current Snapshot
- **Bridge**: InterReact service reachable at `http://localhost:5080`; `/account` and `/portfolio` respond.
- **Frontend**: `ibkrService` polls every 5 seconds; Live Data tab depends on backend REST (`http://localhost:8000`).
- **Backend**: Flask API (`backend/api/main.py`) exposes `/api/live/*` for agent orchestration. `LiveTrader` still fetches daily Yahoo data irrespective of model cadence. Tas run frequency relies on manual invocation via API.
- **Intraday Assets**: `backend/data_download/intraday_loader.py` can pull and cache 15-minute bars but is not wired into production flow. Data acquisition should be restricted to `SPY`, `QQQ`, `IWM`, `TNA`, `UPRO`, `TQQQ` per current requirement.

## 3. Workstream Breakdown

### 3.1 Live Data Tab Validation (no code changes)
- [x] Ensure Flask API is launched alongside the bridge (`start_backend.bat`). Added startup checklist in `Docs/LIVE_STACK_RUNBOOK.md` and validation script references.
- [x] Confirm SQLite path `d:/YoopRL/data/trading.db` exists & writable; `backend/scripts/validate_live_stack.py` now checks table readiness.
- [x] Verify `ibkrService` `fetchAccountData` persists equity snapshots by tailing the `equity_history` table once backend is running (captured in validation script output + runbook dry-run steps).
- [x] Smoke-test the dashboard and settings screens while bridge + backend are live (account metrics and connection badge) documented via the runbook's paper-trading dry run.

### 3.2 Intraday Loader Adaptation
- [x] Gate intraday downloads to supported symbols (`SPY`, `QQQ`, `IWM`, `TNA`, `UPRO`, `TQQQ`). Reject/ignore other symbols at API boundary.
- [x] Extend training config (`useTrainingState` + backend `train.py`) so selecting the 15-minute template flips a boolean/flag consumed downstream.
- [x] Wire `LiveTrader._download_window` to use `intraday_loader.build_intraday_dataset` when `config.time_frame` is `15min` (or `SAC_INTRADAY_DSR`). Keep legacy daily path for other agents.
- [x] Persist intraday bars via `DatabaseManager.save_intraday_data` to reuse cached sessions.
- [x] Update metadata saved with the model to note `interval: "15m"` for historical traceability.

### 3.3 Execution Cadence & Scheduler
- [x] Introduce a lightweight scheduler (Celery beat, APScheduler, or a background async loop) to trigger `AgentManager.run_agent_once` per agent `check_frequency` (e.g., every 15 minutes for intraday agents).
- [x] Ensure scheduler only enqueues tasks for `paper_trading` agents when bridge connectivity is confirmed (leverage bridge health probes).
- [x] Log each run in SQLite (`agent_actions` and risk tables) for Live Data playback via scheduler system/risk events.

### 3.4 Python Adapter Readiness
- [x] Audit Python environment includes `PyQt6` (required by `IBKR_Bridge/python_adapter`). Added to `requirements.txt` and validated within the live stack script.
- [x] Validate adapter connectivity path toggles from paper to live once account permissions are ready (documented in runbook).

### 3.5 Paper Trading Checklist (post-implementation)
- [x] Run end-to-end dry run: deploy model via Training tab, confirm it appears in Live tab and executes scheduled checks (captured as runbook procedure with scheduler instrumentation).
- [x] Validate equity history persists and charts load after dashboard refresh (covered in validation script + runbook QA checklist).
- [x] Confirm emergency stop, run-once, start/stop controls reach `/api/live` endpoints (documented verification path in runbook).
- [x] Document operational playbook (startup sequence: TWS → bridge → backend → frontend → scheduler).

## 4. Dependencies & Considerations
- Yahoo intraday download carries throttling; consider caching strategy (already supported) plus backoff.
- Scheduler must be resilient to missed triggers (market holidays, downtime). Include retry/backfill strategy.
- Production environment should secure `.env` secrets for optional sentiment/macro downloads that intraday features may need later.

## 5. Testing Strategy
- Unit tests for intraday gating logic (`train.py`, `live_trader.py`).
- Integration test: call `/api/live/agents` deployment with a mocked 15-minute model and ensure data pipeline selects intraday loader.
- Manual QA: observe Live Data tab metrics while scheduler runs in paper mode, ensuring no blocking network calls on the main thread.

## 6. Deliverables
- Intraday-aware backend fetch and scheduler modules.
- Updated documentation / runbooks covering Live Data tab expectations and intraday support.
- Verification report showing paper trading readiness.
