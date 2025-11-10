"""File Note: Coordinates periodic live-trading checks for deployed agents."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, UTC
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

import requests

if TYPE_CHECKING:  # pragma: no cover - imports only for type checking
    from database.db_manager import DatabaseManager
    from .agent_manager import AgentManager
    from .live_trader import LiveTrader


class BridgeHealthClient:
    """Caches bridge connectivity checks to avoid hammering the adapter."""

    def __init__(self, timeout: float = 0.5, ttl_seconds: float = 30.0) -> None:
        self._timeout = timeout
        self._ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple[float, bool]] = {}
        self._lock = threading.Lock()

    def is_available(self, host: str, port: int) -> bool:
        key = f"{host}:{port}"
        now = time.time()

        with self._lock:
            cached = self._cache.get(key)
            if cached and (now - cached[0]) < self._ttl_seconds:
                return cached[1]

        status = self._probe_bridge(host, port)

        with self._lock:
            self._cache[key] = (now, status)

        return status

    def _probe_bridge(self, host: str, port: int) -> bool:
        url = f"http://{host}:{port}/health"
        try:
            response = requests.get(url, timeout=self._timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False


class LiveAgentScheduler:
    """Background scheduler that triggers live agents based on their cadence."""

    def __init__(
        self,
        agent_manager: "AgentManager",
        poll_interval_seconds: int = 30,
        bridge_client: Optional[BridgeHealthClient] = None,
    ) -> None:
        self._agent_manager = agent_manager
        self._poll_interval = max(5, int(poll_interval_seconds))
        self._bridge_client = bridge_client or BridgeHealthClient()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._registered_ids: Set[str] = set()
        self._lock = threading.Lock()
        self._logger = logging.getLogger("LiveAgentScheduler")
        self._db: Optional["DatabaseManager"] = None

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def attach_database(self, db_manager: "DatabaseManager") -> None:
        """Provide shared database handle for logging run metadata."""

        self._db = db_manager

    def start(self) -> None:
        """Start the scheduler loop if it is not already running."""

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="LiveAgentScheduler", daemon=True)
        self._thread.start()
        self._logger.info("Live agent scheduler started (poll=%ss)", self._poll_interval)

    def stop(self) -> None:
        """Signal the background thread to exit and wait for it to finish."""

        self._stop_event.set()
        self._wake_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._poll_interval * 2)
            self._logger.info("Live agent scheduler stopped")

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_agent(self, trader: "LiveTrader") -> None:
        """Add an agent to the scheduling roster."""

        if not trader.is_running:
            self._logger.debug("Skipping scheduler registration for stopped agent %s", trader.config.agent_id)
            return

        with self._lock:
            self._registered_ids.add(trader.config.agent_id)
        self._wake_event.set()
        self._logger.debug("Registered agent %s for scheduling", trader.config.agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the scheduling roster."""

        with self._lock:
            self._registered_ids.discard(agent_id)
        self._logger.debug("Unregistered agent %s from scheduling", agent_id)

    def wake(self, agent_id: Optional[str] = None) -> None:
        """Wake the scheduler loop so it can evaluate agents immediately."""

        if agent_id:
            self._logger.debug("Wake requested for agent %s", agent_id)
        self._wake_event.set()

    def has_agents(self) -> bool:
        """Return True if any agents are currently scheduled."""

        with self._lock:
            return bool(self._registered_ids)

    def shutdown(self) -> None:
        """Stop the scheduler and clear all registrations."""

        self.stop()
        with self._lock:
            self._registered_ids.clear()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._run_pending()
            self._wait_for_next_tick()

    def _wait_for_next_tick(self) -> None:
        try:
            self._wake_event.wait(timeout=self._poll_interval)
        finally:
            self._wake_event.clear()

    def _run_pending(self) -> None:
        with self._lock:
            agent_ids = list(self._registered_ids)

        for agent_id in agent_ids:
            if self._stop_event.is_set():
                break

            try:
                trader = self._agent_manager.get_agent(agent_id)
            except KeyError:
                continue

            if not trader.is_running:
                self._log_scheduler_skip(trader, reason="agent_stopped")
                continue

            if trader.paper_trading and not self._bridge_available(trader):
                self._log_scheduler_skip(trader, reason="bridge_unavailable")
                continue

            if hasattr(trader, "should_run_now") and not trader.should_run_now():
                self._log_scheduler_skip(trader, reason="outside_trading_window")
                continue

            interval_seconds = self._determine_interval_seconds(trader)
            last_run = trader.last_run_at
            now = datetime.now(UTC)

            if last_run is None or (now - last_run).total_seconds() >= interval_seconds:
                self._trigger_agent(trader)

    def _trigger_agent(self, trader: "LiveTrader") -> None:
        agent_id = trader.config.agent_id
        self._logger.debug("Triggering live check for %s", agent_id)

        start_time = time.time()
        success = False
        error_detail: Optional[str] = None

        try:
            success, reason = self._agent_manager.run_agent_once(agent_id)
            if reason == "outside_trading_window":
                self._log_scheduler_skip(trader, reason=reason)
                return
            if not success:
                error_detail = "Agent run returned False"
                self._logger.warning("Live agent %s run failed", agent_id)
        except Exception as exc:  # pragma: no cover - defensive logging
            success = False
            error_detail = str(exc)
            self._logger.exception("Unhandled error while executing live agent %s", agent_id)

        duration = time.time() - start_time
        self._log_scheduler_result(trader, success, duration, error_detail)

    def _bridge_available(self, trader: "LiveTrader") -> bool:
        host = getattr(trader.config, "bridge_host", "localhost") or "localhost"
        port = int(getattr(trader.config, "bridge_port", 5080) or 5080)
        status = self._bridge_client.is_available(host, port)
        if not status:
            self._logger.debug(
                "Bridge unavailable for agent %s at %s:%s",
                trader.config.agent_id,
                host,
                port,
            )
        return status

    # ------------------------------------------------------------------
    # Interval resolution helpers
    # ------------------------------------------------------------------
    def _determine_interval_seconds(self, trader: "LiveTrader") -> int:
        config = trader.config
        frequency = str(config.check_frequency or "").strip().lower()
        if not frequency:
            frequency = str(config.data_frequency or "").strip().lower()

        interval = self._parse_frequency_to_seconds(frequency)
        if interval is None:
            interval = self._default_interval_for_trader(config.data_frequency)

        return max(self._poll_interval, interval)

    @staticmethod
    def _default_interval_for_trader(data_frequency: Optional[str]) -> int:
        normalized = (data_frequency or "").lower()
        if normalized in {"intraday", "15m", "15min"}:
            return 15 * 60
        return 24 * 60 * 60

    def _parse_frequency_to_seconds(self, frequency: str) -> Optional[int]:
        if not frequency:
            return None

        mapping: Dict[str, int] = {
            "15m": 15 * 60,
            "15min": 15 * 60,
            "5m": 5 * 60,
            "5min": 5 * 60,
            "30m": 30 * 60,
            "30min": 30 * 60,
            "1h": 60 * 60,
            "hourly": 60 * 60,
            "1d": 24 * 60 * 60,
            "daily": 24 * 60 * 60,
            "eod": 24 * 60 * 60,
        }

        if frequency in mapping:
            return mapping[frequency]

        try:
            if frequency.endswith("min"):
                return int(frequency[:-3]) * 60
            if frequency.endswith("m"):
                return int(frequency[:-1]) * 60
            if frequency.endswith("h"):
                return int(frequency[:-1]) * 60 * 60
            if frequency.endswith("d"):
                return int(frequency[:-1]) * 24 * 60 * 60
            if frequency.endswith("s"):
                return int(frequency[:-1])
        except ValueError:
            self._logger.debug("Unable to parse check_frequency '%s'", frequency)

        return None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _log_scheduler_result(
        self,
        trader: "LiveTrader",
        success: bool,
        duration_seconds: float,
        error_detail: Optional[str],
    ) -> None:
        if self._db is None:
            return

        details: Dict[str, Any] = {
            "agent_id": trader.config.agent_id,
            "symbol": trader.config.symbol,
            "duration_seconds": round(duration_seconds, 4),
            "paper_trading": bool(trader.paper_trading),
        }

        level = "INFO" if success else "WARNING"
        message = "Live agent run completed" if success else "Live agent run failed"
        if error_detail:
            details["error"] = error_detail

        self._db.log_system_event(
            component="LIVE_SCHEDULER",
            level=level,
            message=message,
            details=details,
        )

        if not success:
            self._db.log_risk_event(
                event_type="SCHEDULER_RUN_FAILED",
                severity="WARNING",
                description=error_detail or "run_agent_once returned False",
                agent_name=trader.config.agent_id,
                symbol=trader.config.symbol,
            )

    def _log_scheduler_skip(self, trader: "LiveTrader", *, reason: str) -> None:
        if self._db is None:
            return

        self._db.log_system_event(
            component="LIVE_SCHEDULER",
            level="INFO",
            message="Live agent run skipped",
            details={
                "agent_id": trader.config.agent_id,
                "symbol": trader.config.symbol,
                "reason": reason,
            },
        )
