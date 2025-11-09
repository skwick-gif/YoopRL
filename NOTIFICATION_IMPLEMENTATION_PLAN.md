# Notification & Alert Integration Plan

Comprehensive workplan for adding dual-channel (WhatsApp + Telegram) alerting to YoopRL.

---

## 1. Goals & Outcomes
- Forward critical risk events and trade executions to external messaging channels in real time.
- Allow switching or combining WhatsApp and Telegram without code changes.
- Surface channel status and test controls inside the Monitoring tab.

---

## 2. Backend Tasks

### 2.1 Configuration
- **File:** `backend/config/notifications.py` *(new)*
  - Define dataclasses / dictionaries for channel settings (API keys, endpoints, severity routing).
  - Read environment variables: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `WHATSAPP_FROM`, `WHATSAPP_TO`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `NOTIFICATION_CHANNELS`.
  - Expose helper `load_notification_settings()` returning active channels and secrets.

### 2.2 Notification Dispatcher
- **Folder:** `backend/monitoring/notifications/`
  - `__init__.py` *(new)*: export `NotificationDispatcher`, channel registry, shared alert schema.
  - `dispatcher.py` *(new)*:
    - Accept list of channel instances; expose `dispatch_alert(alert: dict)` with async (ThreadPoolExecutor) send to avoid blocking.
    - Implement routing rules: severity >= WARNING → WhatsApp + Telegram; BUY/SELL actions → Telegram default.
  - `channels/base.py` *(new)*: abstract base class with `send_alert(self, alert: dict) -> bool` and optional health reporting.
  - `channels/whatsapp.py` *(new)*: Twilio REST call via `requests.post`, format message text, return success flag.
  - `channels/telegram.py` *(new)*: Telegram Bot API POST, support Markdown formatting; return success flag.
  - `worker.py` *(new optional)*: background loop polling DB queue (see 2.3) if deferred delivery preferred.

### 2.3 Database Integration
- **File:** `backend/database/db_manager.py`
  - Extend `log_risk_event` and `log_agent_action` to drop event records into a lightweight queue table `notification_queue` (id, payload JSON, status, retries, created_at).
  - Provide helper `fetch_pending_notifications(limit=50)` and `mark_notification_sent(id, status)` for worker use.
  - Ensure schema migration creates the queue table (see `initialize_schema`).

### 2.4 Event Hooking
- **File:** `backend/execution/live_trader.py`
  - Verify `_persist_risk_event` and `_persist_action_event` call the updated DB helper so notifications enqueue automatically.
  - Add optional context (e.g., price, qty, pnl) to enrich message payloads.

### 2.5 Monitoring API Extensions
- **File:** `backend/monitoring/routes.py`
  - Add endpoints:
    - `GET /api/monitoring/notifications/status` → return active channels, last error per channel, backlog size.
    - `POST /api/monitoring/notifications/test` → trigger dispatcher with synthetic alert for manual verification.
  - Wire endpoints to dispatcher and queue helpers.

### 2.6 Background Worker (optional but recommended)
- **Entry Point:** `backend/monitoring/notifications/run_worker.py` *(new)*
  - Loop every N seconds: fetch queued events, send via dispatcher, update status.
  - Allow CLI flags for run interval and dry-run testing.
  - Integrate with `start_backend.bat` (optional) or document manual launch.

### 2.7 Environment & Security
- Update `.env.example` *(if exists)* with new keys and guidance.
- Document secrets handling in `Docs/SETUP_INSTRUCTIONS.md` (new section: “Configuring Notification Channels”).

---

## 3. Frontend Tasks

### 3.1 API Client
- **File:** `frontend/src/services/monitoringAPI.js`
  - Add methods: `getNotificationStatus()` and `sendNotificationTest(payload)`.

### 3.2 Monitoring Tab UI
- **File:** `frontend/src/components/TabMonitoring.jsx`
  - Introduce “Notification Channels” card showing each channel with status badges (OK / Error / Disabled).
  - Provide toggles (checkbox) for enabling WhatsApp / Telegram on-the-fly; call new endpoint to persist preference (if implemented) or use local state with instructions.
  - Add “Send Test Alert” button to hit the POST test endpoint and display toast/inline feedback.
  - Optionally show log of last sent alert (timestamp + outcome) from API response.

### 3.3 Styles
- **File:** `frontend/src/App.css`
  - Add `.notification-card`, `.notification-channel`, `.notification-status` classes to match existing Monitoring aesthetic.

---

## 4. Testing Plan

### 4.1 Backend
- Unit tests in `backend/test_monitoring_notifications.py` *(new)*
  - Mock Twilio/Telegram calls; assert dispatcher routes correctly based on severity/action type.
  - Verify queue interactions: enqueue on log, worker marks sent.
- Integration smoke script calling `/api/monitoring/notifications/test` with env vars stubbed.

### 4.2 Frontend
- Extend existing Monitoring component tests (e.g., `frontend/src/components/__tests__/TabMonitoring.test.jsx`) to cover new card render and button handlers (mock API).
- Manual QA: run backend + frontend, confirm status indicators update and test button triggers WhatsApp/Telegram (using sandbox numbers/bots).

---

## 5. Deployment Checklist
- Obtain Twilio sandbox approvals and Telegram bot token.
- Populate `.env` on staging/production with channel secrets.
- Run DB migration to add `notification_queue` table.
- Start notification worker alongside backend service.
- Confirm `GET /api/monitoring/notifications/status` shows both channels healthy before enabling auto dispatch.

---

## 6. Future Enhancements (Optional)
- Support email/SMS channels via same dispatcher interface.
- Add rate limiting & deduplication (e.g., suppress duplicate alerts within X minutes).
- Allow per-agent notification preferences stored in DB.
- Persist notification history table for audit trail and UI display.

---

## 7. Task Sequencing Summary
1. Implement config & dispatcher skeleton (Sections 2.1–2.2).
2. Add DB queue + schema updates (2.3).
3. Hook dispatcher into event logging (2.4).
4. Build monitoring API endpoints (2.5).
5. Create worker loop (2.6) and environment/docs updates (2.7).
6. Update frontend service + UI + styles (3.x).
7. Write automated tests (4.x) and run manual verification.
8. Follow deployment checklist (5) once all tests pass.
