"""Flask blueprint encapsulating live trading routes."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
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
_CANDLE_CACHE: dict[str, dict[str, Any]] = {}
_CANDLE_CACHE_META: dict[str, dict[str, Any]] = {}
_TICK_CACHE: dict[str, dict[str, Any]] = {}


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


def _parse_bar_timestamp(value: Any) -> Optional[str]:
    """Translate bridge bar timestamps into ISO-8601 UTC strings."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            numeric = float(value)
            # IBKR can emit milliseconds; normalise to seconds range
            if numeric > 1e12:
                numeric /= 1000.0
            moment = datetime.fromtimestamp(numeric, tz=timezone.utc)
            return moment.isoformat()
        except (ValueError, OSError):
            return None

    text = str(value).strip()
    if not text:
        return None

    normalised = " ".join(text.split())
    patterns = (
        "%Y%m%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y%m%d",
        "%Y-%m-%d",
    )

    for pattern in patterns:
        try:
            moment = datetime.strptime(normalised, pattern)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            else:
                moment = moment.astimezone(timezone.utc)
            return moment.isoformat()
        except ValueError:
            continue

    try:
        cleaned = normalised.replace('Z', '+00:00')
        moment = datetime.fromisoformat(cleaned)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(timezone.utc)
        return moment.isoformat()
    except ValueError:
        return None


def _normalise_bar(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract OHLC values from bridge payload entries."""

    def _to_float(raw: Any) -> Optional[float]:
        if raw in (None, "", "nan", "NaN"):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _to_int(raw: Any) -> Optional[int]:
        value = _to_float(raw)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    time_value = (
        payload.get('time')
        or payload.get('Time')
        or payload.get('timestamp')
        or payload.get('Timestamp')
        or payload.get('date')
        or payload.get('Date')
    )
    iso_time = _parse_bar_timestamp(time_value)
    if iso_time is None:
        return None

    open_px = _to_float(payload.get('open') or payload.get('Open'))
    high_px = _to_float(payload.get('high') or payload.get('High'))
    low_px = _to_float(payload.get('low') or payload.get('Low'))
    close_px = _to_float(payload.get('close') or payload.get('Close'))
    if any(value is None for value in (open_px, high_px, low_px, close_px)):
        return None

    volume = _to_int(payload.get('volume') or payload.get('Volume'))

    return {
        'time': iso_time,
        'open': open_px,
        'high': high_px,
        'low': low_px,
        'close': close_px,
        'volume': volume,
    }


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

        if 'auto_restart' in overrides:
            auto_restart_flag = _coerce_bool(overrides['auto_restart'])
            overrides['auto_restart'] = auto_restart_flag
        else:
            auto_restart_flag = _coerce_bool(data.get('auto_restart', True))
            overrides['auto_restart'] = auto_restart_flag

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

        _AGENT_MANAGER.create_agent(
            config=config,
            start_immediately=start_immediately,
            auto_restart=auto_restart_flag,
        )
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

    timeout_seconds = max(5, min(duration + 5, 180))  # bridge collects ticks for `duration`, so extend read timeout accordingly

    request_started = time.monotonic()
    try:
        response = requests.get(
            url,
            params=params,
            timeout=(2, timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.Timeout:
        cached = _TICK_CACHE.get(agent_id)
        cached_ticks = cached.get('ticks') if cached else []
        cached_stamp = cached.get('fetched_at').isoformat() if cached and cached.get('fetched_at') else None
        _LOGGER.warning(
            "Bridge tick request timed out for agent %s (symbol=%s, duration=%ss). Returning %d cached ticks.",
            agent_id,
            getattr(trader.config, 'symbol', 'UNKNOWN'),
            duration,
            len(cached_ticks),
        )
        return jsonify({
            'status': 'warning',
            'agent_id': agent_id,
            'symbol': trader.config.symbol,
            'ticks': cached_ticks,
            'cached': bool(cached_ticks),
            'cached_at': cached_stamp,
            'message': 'Bridge request timed out',
            'latency_ms': int((time.monotonic() - request_started) * 1000),
        }), 200
    except requests.exceptions.RequestException as exc:
        cached = _TICK_CACHE.get(agent_id)
        cached_ticks = cached.get('ticks') if cached else []
        cached_stamp = cached.get('fetched_at').isoformat() if cached and cached.get('fetched_at') else None
        _LOGGER.debug("Bridge market data request failed: %s", exc)
        if cached_ticks:
            return jsonify({
                'status': 'warning',
                'agent_id': agent_id,
                'symbol': trader.config.symbol,
                'ticks': cached_ticks,
                'cached': True,
                'cached_at': cached_stamp,
                'message': str(exc),
                'latency_ms': int((time.monotonic() - request_started) * 1000),
            }), 200
        return jsonify({'status': 'error', 'error': str(exc)}), 502
    except ValueError:
        return jsonify({'status': 'error', 'error': 'Bridge returned non-JSON payload'}), 502

    ticks = payload if isinstance(payload, list) else payload.get('ticks', []) if isinstance(payload, dict) else []

    elapsed_ms = int((time.monotonic() - request_started) * 1000)
    now_utc = datetime.now(timezone.utc)

    if ticks:
        _TICK_CACHE[agent_id] = {
            'ticks': ticks,
            'fetched_at': now_utc,
            'duration': duration,
            'sec_type': sec_type,
            'exchange': exchange,
        }

    return jsonify({
        'status': 'success',
        'agent_id': agent_id,
        'symbol': trader.config.symbol,
        'ticks': ticks,
        'latency_ms': elapsed_ms,
        'fetched_at': now_utc.isoformat(),
    }), 200


@_live_bp.route('/agents/<agent_id>/candles', methods=['GET'])
def get_agent_candles(agent_id: str):
    if _AGENT_MANAGER is None:
        raise RuntimeError("Live routes not initialised with agent manager")

    try:
        trader = _AGENT_MANAGER.get_agent(agent_id)
    except KeyError as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 404

    duration_days = request.args.get('durationDays') or request.args.get('duration_days')
    try:
        duration_days = int(duration_days) if duration_days is not None else 3
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'error': 'durationDays must be an integer'}), 400
    duration_days = max(1, min(duration_days, 30))

    bar_size = request.args.get('barSize') or request.args.get('bar_size') or '15 mins'
    bar_size = str(bar_size).strip() or '15 mins'

    limit = request.args.get('limit', default=160, type=int) or 160
    limit = max(16, min(limit, 400))

    extras = trader.config.extras if isinstance(trader.config.extras, dict) else {}

    sec_type = (
        request.args.get('secType')
        or request.args.get('sec_type')
        or extras.get('sec_type')
        or extras.get('secType')
        or extras.get('security_type')
        or extras.get('securityType')
        or 'STK'
    )
    if isinstance(sec_type, str):
        sec_type = sec_type.upper()

    exchange = (
        request.args.get('exchange')
        or extras.get('exchange')
        or extras.get('primary_exchange')
        or extras.get('primaryExchange')
        or 'SMART'
    )
    if isinstance(exchange, str):
        exchange = exchange.upper()

    bridge_host = request.args.get('bridge_host') or getattr(trader.config, 'bridge_host', 'localhost') or 'localhost'
    bridge_port = request.args.get('bridge_port') or getattr(trader.config, 'bridge_port', 5080) or 5080

    try:
        bridge_port = int(bridge_port)
    except (TypeError, ValueError):
        return jsonify({'status': 'error', 'error': 'Invalid bridge_port value'}), 400

    url = f"http://{bridge_host}:{bridge_port}/historical"
    params = {
        'symbol': trader.config.symbol,
        'secType': sec_type,
        'exchange': exchange,
        'durationDays': duration_days,
        'barSize': bar_size,
    }

    # Historical endpoint loops over multiple whatToShow combinations; give it ample time.
    read_timeout = max(30, min(duration_days * 40, 240))

    try:
        response = requests.get(
            url,
            params=params,
            timeout=(2, read_timeout),
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.Timeout:
        cached = _CANDLE_CACHE.get(agent_id)
        cached_meta = _CANDLE_CACHE_META.get(agent_id, {})
        cached_bars = cached.get('bars') if cached else []
        cached_stamp = cached_meta.get('fetched_at')
        _LOGGER.warning(
            "Bridge historical request timed out for agent %s (symbol=%s, barSize=%s). Returning %d cached bars.",
            agent_id,
            getattr(trader.config, 'symbol', 'UNKNOWN'),
            bar_size,
            len(cached_bars),
        )
        return jsonify({
            'status': 'warning',
            'agent_id': agent_id,
            'symbol': trader.config.symbol,
            'bar_size': bar_size,
            'duration_days': duration_days,
            'count': len(cached_bars),
            'sec_type': sec_type,
            'exchange': exchange,
            'bars': cached_bars,
            'cached': bool(cached_bars),
            'cached_at': cached_stamp,
            'message': 'Bridge request timed out',
        }), 200
    except requests.exceptions.RequestException as exc:
        cached = _CANDLE_CACHE.get(agent_id)
        cached_meta = _CANDLE_CACHE_META.get(agent_id, {})
        cached_bars = cached.get('bars') if cached else []
        cached_stamp = cached_meta.get('fetched_at')
        _LOGGER.debug("Bridge historical data request failed: %s", exc)
        if cached_bars:
            return jsonify({
                'status': 'warning',
                'agent_id': agent_id,
                'symbol': trader.config.symbol,
                'bar_size': bar_size,
                'duration_days': duration_days,
                'count': len(cached_bars),
                'sec_type': sec_type,
                'exchange': exchange,
                'bars': cached_bars,
                'cached': True,
                'cached_at': cached_stamp,
                'message': str(exc),
            }), 200
        return jsonify({'status': 'error', 'error': str(exc)}), 502
    except ValueError:
        return jsonify({'status': 'error', 'error': 'Bridge returned non-JSON payload'}), 502

    raw_bars = []
    if isinstance(payload, dict):
        raw_bars = payload.get('Bars') or payload.get('bars') or []
    elif isinstance(payload, list):
        raw_bars = payload

    normalised = []
    for item in raw_bars:
        if not isinstance(item, dict):
            continue
        bar = _normalise_bar(item)
        if bar is None:
            continue
        normalised.append(bar)

    if normalised:
        normalised = normalised[-limit:]
        fetched_at = datetime.now(timezone.utc).isoformat()
        _CANDLE_CACHE[agent_id] = {'bars': normalised}
        _CANDLE_CACHE_META[agent_id] = {
            'fetched_at': fetched_at,
            'bar_size': bar_size,
            'duration_days': duration_days,
            'sec_type': sec_type,
            'exchange': exchange,
        }

    return jsonify({
        'status': 'success',
        'agent_id': agent_id,
        'symbol': trader.config.symbol,
        'bar_size': bar_size,
        'duration_days': duration_days,
        'count': len(normalised),
        'sec_type': sec_type,
        'exchange': exchange,
        'bars': normalised,
    }), 200
