"""
Live Trading Agent Manager

Coordinates multiple `LiveTrader` instances:
- Creates and tracks agent lifecycles for the live trading service
- Provides start/stop/check primitives for API and scheduler layers
- Aggregates status snapshots for the frontend dashboard

Author: YoopRL System
Date: November 8, 2025
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from database.db_manager import DatabaseManager

from .live_trader import LiveTrader, LiveTraderConfig
from .live_scheduler import LiveAgentScheduler

logger = logging.getLogger(__name__)


class AgentManager:
    """Simple in-memory registry for live trading agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, LiveTrader] = {}
        self.logger = logging.getLogger(f"{__name__}.AgentManager")
        self._db: Optional[DatabaseManager] = None
        self._scheduler = LiveAgentScheduler(self)
        self._bootstrapped = False

    def attach_database(self, db_manager: DatabaseManager) -> None:
        """Inject shared persistence layer used for monitoring telemetry."""

        self._db = db_manager
        self._scheduler.attach_database(db_manager)
        self.logger.info("Database manager attached to AgentManager")
        self._bootstrap_agents()

    # ------------------------------------------------------------------
    # Agent lifecycle helpers
    # ------------------------------------------------------------------
    def create_agent(
        self,
        config: Union[LiveTraderConfig, Dict[str, object]],
        start_immediately: bool = False,
        auto_restart: Optional[bool] = None,
    ) -> LiveTrader:
        """Instantiate a new live agent and optionally start it."""

        trader_config = config if isinstance(config, LiveTraderConfig) else LiveTraderConfig(**config)  # type: ignore[arg-type]
        agent_id = trader_config.agent_id

        if agent_id in self._agents:
            self.logger.info("Replacing existing agent %s", agent_id)
            self.remove_agent(agent_id)

        if auto_restart is not None:
            trader_config.auto_restart = bool(auto_restart)
            trader_config.extras = dict(trader_config.extras or {})
            trader_config.extras["auto_restart"] = trader_config.auto_restart

        trader = LiveTrader(trader_config, db_manager=self._db)
        self._agents[agent_id] = trader
        self.logger.info("Agent %s registered", agent_id)

        status = "stopped"
        if start_immediately:
            started = trader.start()
            if started:
                self._register_with_scheduler(trader)
                status = "running"
            else:
                self.logger.error("Failed to start live agent %s: %s", agent_id, trader.last_error)
                status = "error"

        self._persist_agent_record(trader_config, status)
        return trader

    def start_agent(self, agent_id: str) -> tuple[bool, Optional[str]]:
        trader = self._require_agent(agent_id)
        started = trader.start()
        if started:
            self._register_with_scheduler(trader)
            self._persist_agent_record(trader.config, "running")
            return True, None
        self._persist_agent_record(trader.config, "error")
        return False, trader.last_error

    def stop_agent(self, agent_id: str) -> None:
        trader = self._require_agent(agent_id)
        self._scheduler.unregister_agent(agent_id)
        trader.stop()
        self._persist_agent_record(trader.config, "stopped")
        self._maybe_stop_scheduler()

    def remove_agent(self, agent_id: str) -> None:
        trader = self._agents.pop(agent_id, None)
        if trader is None:
            message = f"Agent '{agent_id}' not found"
            self.logger.error(message)
            raise KeyError(message)

        self._scheduler.unregister_agent(agent_id)
        trader.stop()
        self._maybe_stop_scheduler()
        self.logger.info("Agent %s removed", agent_id)
        if self._db is not None:
            try:
                self._db.remove_live_agent(agent_id)
            except Exception as exc:  # pragma: no cover - persistence should not break removal
                self.logger.error("Failed to remove persisted agent %s: %s", agent_id, exc)

    def stop_all(self) -> None:
        for agent_id, trader in list(self._agents.items()):
            self._scheduler.unregister_agent(agent_id)
            trader.stop()
            self._persist_agent_record(trader.config, "stopped")
        self._scheduler.shutdown()
        self.logger.info("All agents stopped")

    def emergency_stop(self) -> None:
        self.stop_all()
        self.logger.warning("Emergency stop invoked for all agents")

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def run_agent_once(self, agent_id: str, *, force: bool = False) -> tuple[bool, Optional[str]]:
        trader = self._require_agent(agent_id)

        if not force and hasattr(trader, "should_run_now") and not trader.should_run_now():
            return False, "outside_trading_window"

        return trader.run_single_check(), None

    def run_all(self, agent_ids: Optional[Iterable[str]] = None) -> Dict[str, bool]:
        ids = list(agent_ids) if agent_ids else list(self._agents.keys())
        results = {}
        for agent_id in ids:
            try:
                results[agent_id] = self.run_agent_once(agent_id)[0]
            except KeyError:
                results[agent_id] = False
        return results

    def close_position(self, agent_id: str) -> bool:
        trader = self._require_agent(agent_id)
        return trader.close_position()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def get_agent(self, agent_id: str) -> LiveTrader:
        return self._require_agent(agent_id)

    def list_agents(self) -> Dict[str, LiveTrader]:
        return dict(self._agents)

    def get_all_status(self) -> Dict[str, Dict[str, object]]:
        return {agent_id: trader.get_status() for agent_id, trader in self._agents.items()}

    def _require_agent(self, agent_id: str) -> LiveTrader:
        if agent_id not in self._agents:
            message = f"Agent '{agent_id}' not found"
            self.logger.error(message)
            raise KeyError(message)
        return self._agents[agent_id]

    def wake_agent(self, agent_id: Optional[str] = None) -> None:
        """Expose scheduler wake helper for API triggers."""

        self._scheduler.wake(agent_id)

    def _register_with_scheduler(self, trader: LiveTrader) -> None:
        self._scheduler.register_agent(trader)
        self._scheduler.start()

    def _maybe_stop_scheduler(self) -> None:
        if not self._scheduler.has_agents():
            self._scheduler.stop()

    def _persist_agent_record(self, config: LiveTraderConfig, status: str) -> None:
        if self._db is None:
            return

        status_normalized = (status or "stopped").lower()
        auto_restart = bool(getattr(config, "auto_restart", True))

        def _serialise(value):  # lightweight JSON sanitation
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: _serialise(val) for key, val in value.items()}
            if isinstance(value, list):
                return [_serialise(item) for item in value]
            return value

        payload = _serialise(config.as_dict())

        try:
            self._db.persist_live_agent(
                agent_id=config.agent_id,
                agent_type=config.agent_type,
                symbol=config.symbol,
                config=payload,
                status=status_normalized,
                auto_restart=auto_restart,
            )
        except Exception as exc:  # pragma: no cover - persistence must not halt trading
            self.logger.error("Failed to persist live agent %s: %s", config.agent_id, exc)

    def _bootstrap_agents(self) -> None:
        if self._bootstrapped or self._db is None:
            return

        try:
            records = self._db.load_live_agents(only_auto_restart=True)
        except Exception as exc:  # pragma: no cover - bootstrapping should fail gracefully
            self.logger.error("Failed to load persisted live agents: %s", exc)
            self._bootstrapped = True
            return

        if not records:
            self._bootstrapped = True
            return

        for record in records:
            agent_id = record.get("agent_id")
            try:
                config_payload = record.get("config") or {}
                config = self._deserialize_config(config_payload)
            except Exception as exc:  # pragma: no cover - malformed persisted config
                self.logger.error("Failed to deserialize config for agent %s: %s", agent_id, exc)
                if self._db is not None and agent_id:
                    try:
                        self._db.update_live_agent_status(agent_id, "error")
                    except Exception:  # pragma: no cover - best effort
                        pass
                continue

            if config.agent_id in self._agents:
                continue

            trader = LiveTrader(config, db_manager=self._db)
            self._agents[config.agent_id] = trader

            persisted_status = str(record.get("status") or "stopped").lower()
            should_restart = persisted_status == "running" and bool(record.get("auto_restart", 1))

            if should_restart:
                started = trader.start()
                if started:
                    self._register_with_scheduler(trader)
                    self._persist_agent_record(config, "running")
                    self.logger.info("Bootstrapped live agent %s (auto-restart)", config.agent_id)
                else:
                    self.logger.error(
                        "Failed to restart live agent %s during bootstrap: %s",
                        config.agent_id,
                        trader.last_error,
                    )
                    self._persist_agent_record(config, "error")
            else:
                self._persist_agent_record(config, persisted_status or "stopped")
                self.logger.info("Bootstrapped live agent %s in stopped state", config.agent_id)

        self._bootstrapped = True

    def _deserialize_config(self, payload: Dict[str, object]) -> LiveTraderConfig:
        data = dict(payload)

        for field_name in ("model_path", "normalizer_path", "metadata_path"):
            raw_value = data.get(field_name)
            if raw_value:
                data[field_name] = Path(str(raw_value))
            elif field_name != "model_path":
                data[field_name] = None

        if data.get("extras") is None:
            data["extras"] = {}

        if data.get("features_used") is None:
            data["features_used"] = []

        return LiveTraderConfig(**data)


agent_manager = AgentManager()
"""Module-level singleton used by API endpoints."""
