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

logger = logging.getLogger(__name__)


class AgentManager:
    """Simple in-memory registry for live trading agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, LiveTrader] = {}
        self.logger = logging.getLogger(f"{__name__}.AgentManager")
        self._db: Optional[DatabaseManager] = None

    def attach_database(self, db_manager: DatabaseManager) -> None:
        """Inject shared persistence layer used for monitoring telemetry."""

        self._db = db_manager
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
            trader.start()
        return trader

    def start_agent(self, agent_id: str) -> bool:
        trader = self._require_agent(agent_id)
        return trader.start()

    def stop_agent(self, agent_id: str) -> None:
        trader = self._require_agent(agent_id)
        trader.stop()

    def remove_agent(self, agent_id: str) -> None:
        trader = self._agents.pop(agent_id, None)
        if trader is None:
            message = f"Agent '{agent_id}' not found"
            self.logger.error(message)
            raise KeyError(message)

        trader.stop()
        self.logger.info("Agent %s removed", agent_id)

    def stop_all(self) -> None:
        for trader in self._agents.values():
            trader.stop()
        self.logger.info("All agents stopped")

    def emergency_stop(self) -> None:
        self.stop_all()
        self.logger.warning("Emergency stop invoked for all agents")

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def run_agent_once(self, agent_id: str) -> bool:
        trader = self._require_agent(agent_id)
        return trader.run_single_check()

    def run_all(self, agent_ids: Optional[Iterable[str]] = None) -> Dict[str, bool]:
        ids = list(agent_ids) if agent_ids else list(self._agents.keys())
        results = {}
        for agent_id in ids:
            try:
                results[agent_id] = self.run_agent_once(agent_id)
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


agent_manager = AgentManager()
"""Module-level singleton used by API endpoints."""
