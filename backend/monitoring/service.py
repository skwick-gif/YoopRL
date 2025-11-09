"""Monitoring aggregation helpers for API endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database.db_manager import DatabaseManager
from execution.agent_manager import agent_manager as live_agent_manager


def _parse_iso8601(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.rstrip("Z")
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def build_monitoring_summary(
    db: DatabaseManager,
    hours: int = 24,
    action_limit: int = 5,
    alert_limit: int = 5,
    log_limit: int = 5,
) -> Dict[str, Any]:
    """Collect core monitoring metrics for the dashboard header."""

    equity = db.get_equity_summary(hours=hours)
    actions = db.get_recent_agent_actions(limit=action_limit)
    alerts = db.get_recent_risk_events(limit=alert_limit)
    logs = db.get_recent_system_logs(limit=log_limit)
    agent_snapshot = _build_agent_snapshot()

    return {
        "equity": equity,
        "agents": agent_snapshot,
        "recent_actions": actions,
        "recent_alerts": alerts,
        "recent_logs": logs,
    }


def get_recent_actions(
    db: DatabaseManager,
    limit: int = 50,
    agent_name: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Expose recent agent actions with optional filters."""

    actions = db.get_recent_agent_actions(limit=limit, agent_name=agent_name, symbol=symbol)
    return {
        "count": len(actions),
        "actions": actions,
    }


def get_recent_alerts(
    db: DatabaseManager,
    limit: int = 50,
    severity: Optional[str] = None,
) -> Dict[str, Any]:
    """Return recent risk alerts."""

    alerts = db.get_recent_risk_events(limit=limit, severity=severity)
    return {
        "count": len(alerts),
        "alerts": alerts,
    }


def get_system_diagnostics(
    db: DatabaseManager,
    log_limit: int = 50,
) -> Dict[str, Any]:
    """Return system-level diagnostics including agent heartbeat and logs."""

    agent_snapshot = _build_agent_snapshot()
    logs = db.get_recent_system_logs(limit=log_limit)
    return {
        "agents": agent_snapshot,
        "logs": logs,
    }


def _build_agent_snapshot() -> Dict[str, Any]:
    """Generate agent status summary for monitoring endpoints."""

    statuses_raw = live_agent_manager.get_all_status()
    now = datetime.utcnow().replace(tzinfo=timezone.utc)

    formatted_statuses: List[Dict[str, Any]] = []
    running = 0

    for agent_id, payload in statuses_raw.items():
        last_run_raw = payload.get("last_run_at")
        last_run_dt = _parse_iso8601(last_run_raw)
        seconds_since_last_run: Optional[float] = None

        if last_run_dt is not None:
            if last_run_dt.tzinfo is None:
                last_run_dt = last_run_dt.replace(tzinfo=timezone.utc)
            seconds_since_last_run = (now - last_run_dt).total_seconds()
            seconds_since_last_run = max(seconds_since_last_run, 0.0)

        entry = {
            **payload,
            "agent_id": agent_id,
            "seconds_since_last_run": seconds_since_last_run,
        }

        if payload.get("is_running"):
            running += 1

        formatted_statuses.append(entry)

    total = len(formatted_statuses)
    return {
        "total": total,
        "running": running,
        "stopped": max(total - running, 0),
        "statuses": formatted_statuses,
    }
