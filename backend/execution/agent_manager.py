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

    def attach_database(self, db_manager: DatabaseManager) -> None:
        """Inject shared persistence layer used for monitoring telemetry."""

        self._db = db_manager
        self._scheduler.attach_database(db_manager)
        self.logger.info("Database manager attached to AgentManager")

    # ------------------------------------------------------------------
    # Agent lifecycle helpers
    # ------------------------------------------------------------------
    def create_agent(
        self,
        config: Union[LiveTraderConfig, Dict[str, object]],
        start_immediately: bool = False,
    ) -> LiveTrader:
        """Instantiate a new live agent and optionally start it."""

        trader_config = config if isinstance(config, LiveTraderConfig) else LiveTraderConfig(**config)  # type: ignore[arg-type]
        agent_id = trader_config.agent_id

        if agent_id in self._agents:
            self.logger.info("Replacing existing agent %s", agent_id)
            self.remove_agent(agent_id)

        trader = LiveTrader(trader_config, db_manager=self._db)
        self._agents[agent_id] = trader
        self.logger.info("Agent %s registered", agent_id)

        if start_immediately:
            started = trader.start()
            if started:
                self._register_with_scheduler(trader)
            else:
                self.logger.error("Failed to start live agent %s: %s", agent_id, trader.last_error)
        return trader

    def start_agent(self, agent_id: str) -> tuple[bool, Optional[str]]:
        trader = self._require_agent(agent_id)
        started = trader.start()
        if started:
            self._register_with_scheduler(trader)
            return True, None
        return False, trader.last_error

    def stop_agent(self, agent_id: str) -> None:
        trader = self._require_agent(agent_id)
        self._scheduler.unregister_agent(agent_id)
        trader.stop()
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

    def stop_all(self) -> None:
        for agent_id, trader in list(self._agents.items()):
            self._scheduler.unregister_agent(agent_id)
            trader.stop()
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


agent_manager = AgentManager()
"""Module-level singleton used by API endpoints."""
