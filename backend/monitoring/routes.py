"""Flask blueprint exposing monitoring endpoints."""

from __future__ import annotations

import logging
from typing import Optional

from flask import Blueprint, jsonify, request

from database.db_manager import DatabaseManager
from monitoring.service import (
    build_monitoring_summary,
    get_recent_actions as monitoring_get_recent_actions,
    get_recent_alerts as monitoring_get_recent_alerts,
    get_system_diagnostics,
)

_monitoring_bp = Blueprint("monitoring", __name__)
_DB: Optional[DatabaseManager] = None
_LOGGER: logging.Logger = logging.getLogger(__name__)


def register_monitoring_routes(app, db: DatabaseManager, logger: Optional[logging.Logger] = None) -> None:
    """Attach monitoring routes to the provided Flask app."""

    global _DB, _LOGGER  # pylint: disable=global-statement

    _DB = db
    if logger is not None:
        _LOGGER = logger

    app.register_blueprint(_monitoring_bp, url_prefix="/api/monitoring")


@_monitoring_bp.route("/summary", methods=["GET"])
def monitoring_summary():
    """Return aggregated monitoring metrics for dashboard consumption."""

    try:
        if _DB is None:
            raise RuntimeError("Monitoring routes not initialised with DatabaseManager")

        hours = request.args.get("hours", default=24, type=int)
        action_limit = request.args.get("actions_limit", default=5, type=int)
        alert_limit = request.args.get("alerts_limit", default=5, type=int)
        log_limit = request.args.get("logs_limit", default=5, type=int)

        summary = build_monitoring_summary(
            db=_DB,
            hours=hours,
            action_limit=max(1, action_limit),
            alert_limit=max(1, alert_limit),
            log_limit=max(1, log_limit),
        )

        return jsonify({"status": "success", "data": summary}), 200
    except Exception as exc:  # pylint: disable=broad-except
        _LOGGER.error("Monitoring summary failed: %s", exc, exc_info=True)
        return jsonify({"status": "error", "error": str(exc)}), 500


@_monitoring_bp.route("/actions", methods=["GET"])
def monitoring_actions():
    """Return recent agent actions."""

    try:
        if _DB is None:
            raise RuntimeError("Monitoring routes not initialised with DatabaseManager")
        limit = request.args.get("limit", default=50, type=int)
        agent_name = request.args.get("agent")
        symbol = request.args.get("symbol")

        payload = monitoring_get_recent_actions(
            db=_DB,
            limit=max(1, limit),
            agent_name=agent_name,
            symbol=symbol,
        )

        return jsonify({"status": "success", "data": payload}), 200
    except Exception as exc:  # pylint: disable=broad-except
        _LOGGER.error("Monitoring actions failed: %s", exc, exc_info=True)
        return jsonify({"status": "error", "error": str(exc)}), 500


@_monitoring_bp.route("/alerts", methods=["GET"])
def monitoring_alerts():
    """Return recent risk alerts."""

    try:
        if _DB is None:
            raise RuntimeError("Monitoring routes not initialised with DatabaseManager")
        limit = request.args.get("limit", default=50, type=int)
        severity = request.args.get("severity")

        payload = monitoring_get_recent_alerts(
            db=_DB,
            limit=max(1, limit),
            severity=severity,
        )

        return jsonify({"status": "success", "data": payload}), 200
    except Exception as exc:  # pylint: disable=broad-except
        _LOGGER.error("Monitoring alerts failed: %s", exc, exc_info=True)
        return jsonify({"status": "error", "error": str(exc)}), 500


@_monitoring_bp.route("/system", methods=["GET"])
def monitoring_system():
    """Return system diagnostics (agent heartbeat, logs)."""

    try:
        if _DB is None:
            raise RuntimeError("Monitoring routes not initialised with DatabaseManager")
        log_limit = request.args.get("logs_limit", default=50, type=int)

        payload = get_system_diagnostics(
            db=_DB,
            log_limit=max(1, log_limit),
        )

        return jsonify({"status": "success", "data": payload}), 200
    except Exception as exc:  # pylint: disable=broad-except
        _LOGGER.error("Monitoring system diagnostics failed: %s", exc, exc_info=True)
        return jsonify({"status": "error", "error": str(exc)}), 500
