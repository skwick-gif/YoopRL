"""Flask blueprint encapsulating live trading routes."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from flask import Blueprint, jsonify, request

from execution import LiveTraderConfig
from execution.agent_manager import AgentManager
from models.model_manager import ModelManager
from database.db_manager import DatabaseManager

_live_bp = Blueprint("live", __name__)
_LOGGER: logging.Logger = logging.getLogger(__name__)
_AGENT_MANAGER: Optional[AgentManager] = None
_DB: Optional[DatabaseManager] = None
_MODEL_MANAGER: Optional[ModelManager] = None
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _coerce_bool(value: Any) -> bool:
    """Best-effort conversion of payload values to boolean flags."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    if value is None:
        return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def register_live_routes(
    app,
    db: DatabaseManager,
    agent_manager: AgentManager,
    model_manager: ModelManager,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Attach live trading routes to the provided Flask app."""

    global _LOGGER, _AGENT_MANAGER, _DB, _MODEL_MANAGER  # pylint: disable=global-statement

    _LOGGER = logger or _LOGGER
    _AGENT_MANAGER = agent_manager
    _DB = db
    _MODEL_MANAGER = model_manager

    app.register_blueprint(_live_bp, url_prefix="/api/live")


def _load_model_metadata(model_id: str) -> Dict[str, Any]:
    if not model_id:
        raise ValueError("model_id is required")

    if _MODEL_MANAGER is None:
        raise RuntimeError("Live routes not initialised with model manager")

    agent_prefix = model_id.split('_')[0].lower()
    metadata_path = Path(_MODEL_MANAGER.base_dir) / agent_prefix / f"{model_id}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found for model_id {model_id}")

    with metadata_path.open('r', encoding='utf-8') as handle:
        metadata = json.load(handle)

    metadata.setdefault('metadata_path', str(metadata_path))
    if 'model_path' not in metadata and metadata.get('file_path'):
        metadata['model_path'] = metadata['file_path']
    if 'model_path' not in metadata and metadata.get('model_id'):
        agent_prefix = metadata.get('model_id')
        candidate = metadata_path.with_name(f"{agent_prefix}.zip")
        metadata['model_path'] = str(candidate)
    return metadata


@_live_bp.route('/agents', methods=['GET'])
def list_live_agents():
    if _AGENT_MANAGER is None:
        raise RuntimeError("Live routes not initialised with agent manager")

    try:
        statuses = _AGENT_MANAGER.get_all_status()
        return jsonify({
            'status': 'success',
            'count': len(statuses),
            'agents': list(statuses.values()),
        })
    except Exception as exc:  # pylint: disable=broad-except
        _LOGGER.error("Failed to list live agents: %s", exc, exc_info=True)
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@_live_bp.route('/agents', methods=['POST'])
def create_live_agent():
    if _AGENT_MANAGER is None or _DB is None:
        raise RuntimeError("Live routes not initialised properly")

    try:
        data = request.get_json(silent=True) or {}

        metadata = data.get('metadata')
        if metadata is None:
            model_id = data.get('model_id')
            metadata = _load_model_metadata(model_id)

        features_used = metadata.get('features_used') or []
        if not features_used:
            features_used = []
            feature_config = metadata.get('features', {})
            if isinstance(feature_config, dict):
                for name, value in feature_config.items():
                    if isinstance(value, dict):
                        if value.get('enabled'):
                            features_used.append(name)
                    elif value:
                        features_used.append(name)
        if not features_used:
            return jsonify({
                'status': 'error',
                'error': 'features_used missing from metadata and could not be inferred',
            }), 400

        metadata['features_used'] = features_used

        overrides = data.get('overrides', {}) or {}

        scalar_overrides = {
            'initial_capital': data.get('initial_capital'),
            'max_position_pct': data.get('max_position_pct'),
            'risk_per_trade': data.get('risk_per_trade'),
            'time_frame': data.get('time_frame'),
            'bar_size': data.get('bar_size'),
            'lookback_days': data.get('lookback_days'),
            'check_frequency': data.get('check_frequency'),
            'paper_trading': data.get('paper_trading'),
            'bridge_host': data.get('bridge_host'),
            'bridge_port': data.get('bridge_port'),
            'allow_premarket': data.get('allow_premarket'),
            'allow_afterhours': data.get('allow_afterhours'),
            'data_frequency': data.get('data_frequency'),
            'interval': data.get('interval'),
            'benchmark_symbol': data.get('benchmark_symbol'),
            'features_used': data.get('features_used'),
            'features_config': data.get('features_config'),
            'normalizer_path': data.get('normalizer_path'),
            'model_path': data.get('model_path'),
            'extras': data.get('extras'),
        }
        overrides.update({k: v for k, v in scalar_overrides.items() if v is not None})

        if 'model_path' in overrides:
            model_override = Path(overrides['model_path'])
            if not model_override.is_absolute():
                model_override = (_PROJECT_ROOT / model_override).resolve()
            else:
                model_override = model_override.resolve()
            overrides['model_path'] = model_override
        if 'normalizer_path' in overrides:
            normalizer_override = Path(overrides['normalizer_path'])
            if not normalizer_override.is_absolute():
                normalizer_override = (_PROJECT_ROOT / normalizer_override).resolve()
            else:
                normalizer_override = normalizer_override.resolve()
            overrides['normalizer_path'] = normalizer_override
        if 'features_used' in overrides and isinstance(overrides['features_used'], str):
            overrides['features_used'] = [overrides['features_used']]
        if 'features_used' in overrides and not isinstance(overrides['features_used'], list):
            raise ValueError("features_used override must be a list of feature names")
        if 'features_config' in overrides and overrides['features_config'] is None:
            overrides.pop('features_config', None)
        if 'initial_capital' in overrides:
            overrides['initial_capital'] = float(overrides['initial_capital'])
        if 'max_position_pct' in overrides:
            overrides['max_position_pct'] = float(overrides['max_position_pct'])
        if 'risk_per_trade' in overrides:
            overrides['risk_per_trade'] = float(overrides['risk_per_trade'])
        if 'lookback_days' in overrides:
            overrides['lookback_days'] = int(overrides['lookback_days'])
        if 'bridge_port' in overrides:
            overrides['bridge_port'] = int(overrides['bridge_port'])
        if 'paper_trading' in overrides:
            overrides['paper_trading'] = _coerce_bool(overrides['paper_trading'])
        if 'allow_premarket' in overrides:
            overrides['allow_premarket'] = _coerce_bool(overrides['allow_premarket'])
        if 'allow_afterhours' in overrides:
            overrides['allow_afterhours'] = _coerce_bool(overrides['allow_afterhours'])
        if 'data_frequency' in overrides and overrides['data_frequency'] is not None:
            overrides['data_frequency'] = str(overrides['data_frequency']).lower()
        if 'interval' in overrides and overrides['interval'] is not None:
            overrides['interval'] = str(overrides['interval']).lower()
        if 'benchmark_symbol' in overrides and overrides['benchmark_symbol']:
            overrides['benchmark_symbol'] = str(overrides['benchmark_symbol']).upper()

        agent_id = data.get('agent_id')
        if not agent_id:
            agent_prefix = str(metadata.get('agent_type', 'AGENT')).upper()
            agent_symbol = str(metadata.get('symbol', 'ASSET')).upper()
            agent_id = f"{agent_prefix}_{agent_symbol}_{uuid.uuid4().hex[:6]}"

        start_immediately = bool(data.get('start_immediately', True))

        config = LiveTraderConfig.from_metadata(
            agent_id=agent_id,
            metadata=metadata,
            overrides=overrides,
        )

        _AGENT_MANAGER.create_agent(config=config, start_immediately=start_immediately)
        status = _AGENT_MANAGER.get_agent(agent_id).get_status()
        started_flag = bool(status.get('is_running'))
        if started_flag:
            _AGENT_MANAGER.wake_agent(agent_id)

        def _serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {key: _serialize(val) for key, val in obj.items()}
            if isinstance(obj, list):
                return [_serialize(item) for item in obj]
            return obj

        _DB.log_system_event(
            component='LIVE_DEPLOYMENT',
            level='INFO',
            message=f"Live agent {agent_id} created",
            details={
                'agent_id': agent_id,
                'start_immediately': start_immediately,
                'config': _serialize(config.as_dict()),
                'metadata_path': metadata.get('metadata_path'),
                'overrides': _serialize(overrides),
            },
        )

        return jsonify({
            'status': 'success',
            'agent_id': agent_id,
            'started': started_flag,
            'agent': status,
        }), 201

    except (ValueError, FileNotFoundError) as exc:
        _LOGGER.warning("Failed to create live agent: %s", exc)
        return jsonify({'status': 'error', 'error': str(exc)}), 400
    except Exception as exc:  # pylint: disable=broad-except
        _LOGGER.error("Unexpected error creating live agent: %s", exc, exc_info=True)
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@_live_bp.route('/agents/<agent_id>/start', methods=['POST'])
def start_live_agent(agent_id):
    if _AGENT_MANAGER is None:
        raise RuntimeError("Live routes not initialised with agent manager")

    try:
        success, reason = _AGENT_MANAGER.start_agent(agent_id)
        if not success:
            error_message = reason or 'failed to start agent'
            return jsonify({'status': 'error', 'error': error_message}), 500
        _AGENT_MANAGER.wake_agent(agent_id)
        status = _AGENT_MANAGER.get_agent(agent_id).get_status()
        return jsonify({'status': 'success', 'agent_id': agent_id, 'agent': status}), 200
    except KeyError as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 404


@_live_bp.route('/agents/<agent_id>/stop', methods=['POST'])
def stop_live_agent(agent_id):
    if _AGENT_MANAGER is None:
        raise RuntimeError("Live routes not initialised with agent manager")

    try:
        _AGENT_MANAGER.stop_agent(agent_id)
        return jsonify({'status': 'success', 'agent_id': agent_id}), 200
    except KeyError as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 404


@_live_bp.route('/agents/<agent_id>/run', methods=['POST'])
def run_live_agent_once(agent_id):
    if _AGENT_MANAGER is None:
        raise RuntimeError("Live routes not initialised with agent manager")

    try:
        success, reason = _AGENT_MANAGER.run_agent_once(agent_id, force=True)
        status = 'success' if success else ('skipped' if reason else 'error')
        return jsonify({'status': status, 'agent_id': agent_id, 'ran': success, 'reason': reason}), 200
    except KeyError as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 404


@_live_bp.route('/agents/<agent_id>/position/close', methods=['POST'])
def close_live_agent_position(agent_id):
    if _AGENT_MANAGER is None:
        raise RuntimeError("Live routes not initialised with agent manager")

    try:
        success = _AGENT_MANAGER.close_position(agent_id)
        return jsonify({'status': 'success' if success else 'error', 'agent_id': agent_id}), 200
    except KeyError as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 404


@_live_bp.route('/agents/<agent_id>/trading-hours', methods=['PATCH'])
def update_trading_hours(agent_id: str):
    if _AGENT_MANAGER is None:
        raise RuntimeError("Live routes not initialised with agent manager")

    try:
        trader = _AGENT_MANAGER.get_agent(agent_id)
    except KeyError as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 404
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({'status': 'error', 'error': 'Invalid payload'}), 400

    updated: Dict[str, bool] = {}

    if 'allow_premarket' in payload:
        flag = _coerce_bool(payload['allow_premarket'])
        trader.allow_premarket = flag
        trader.config.allow_premarket = flag
        trader.config.extras = dict(trader.config.extras or {})
        trader.config.extras['allow_premarket'] = flag
        updated['allow_premarket'] = flag

    if 'allow_afterhours' in payload:
        flag = _coerce_bool(payload['allow_afterhours'])
        trader.allow_afterhours = flag
        trader.config.allow_afterhours = flag
        trader.config.extras = dict(trader.config.extras or {})
        trader.config.extras['allow_afterhours'] = flag
        updated['allow_afterhours'] = flag

    if not updated:
        return jsonify({'status': 'error', 'error': 'No trading window fields provided'}), 400

    if _DB is not None:
        _DB.log_system_event(  # type: ignore[call-arg]
            component='LIVE_DEPLOYMENT',
            level='INFO',
            message='Updated trading hours settings',
            details={
                'agent_id': agent_id,
                'updates': updated,
            },
        )

    return jsonify({
        'status': 'success',
        'agent_id': agent_id,
        'updated': updated,
        'agent': trader.get_status(),
    }), 200


@_live_bp.route('/agents/<agent_id>', methods=['DELETE'])
def remove_live_agent(agent_id):
    if _AGENT_MANAGER is None:
        raise RuntimeError("Live routes not initialised with agent manager")

    try:
        _AGENT_MANAGER.remove_agent(agent_id)
        return jsonify({'status': 'success', 'agent_id': agent_id}), 200
    except KeyError as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 404


@_live_bp.route('/emergency-stop', methods=['POST'])
def emergency_stop_agents():
    if _AGENT_MANAGER is None:
        raise RuntimeError("Live routes not initialised with agent manager")

    _AGENT_MANAGER.emergency_stop()
    return jsonify({'status': 'success', 'message': 'All agents stopped'}), 200


@_live_bp.route('/agents/<agent_id>/ticks', methods=['GET'])
def get_agent_ticks(agent_id: str):
    if _AGENT_MANAGER is None:
        raise RuntimeError("Live routes not initialised with agent manager")

    try:
        trader = _AGENT_MANAGER.get_agent(agent_id)
    except KeyError as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 404
    duration = request.args.get('duration', default=20, type=int)
    duration = max(1, min(duration, 120))

    sec_type = request.args.get('secType', default='STK')
    exchange = request.args.get('exchange', default='SMART')

    bridge_host = request.args.get('bridge_host') or getattr(trader.config, 'bridge_host', 'localhost') or 'localhost'
    bridge_port = request.args.get('bridge_port') or getattr(trader.config, 'bridge_port', 5080) or 5080

    try:
        bridge_port = int(bridge_port)
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'error': 'Invalid bridge_port value'}), 400

    url = f"http://{bridge_host}:{bridge_port}/marketdata"
    params = {
        'symbol': trader.config.symbol,
        'secType': sec_type,
        'exchange': exchange,
        'durationSeconds': duration,
    }

    try:
        response = requests.get(url, params=params, timeout=3)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.Timeout:
        return jsonify({'status': 'error', 'error': 'Bridge request timed out'}), 504
    except requests.exceptions.RequestException as exc:
        _LOGGER.debug("Bridge market data request failed: %s", exc)
        return jsonify({'status': 'error', 'error': str(exc)}), 502
    except ValueError:
        return jsonify({'status': 'error', 'error': 'Bridge returned non-JSON payload'}), 502

    ticks = payload if isinstance(payload, list) else payload.get('ticks', []) if isinstance(payload, dict) else []

    return jsonify({
        'status': 'success',
        'agent_id': agent_id,
        'symbol': trader.config.symbol,
        'ticks': ticks,
    }), 200
