"""
Live Trading Engine Module

Provides runtime execution for trained RL agents:
- Loads trained Stable-Baselines3 models and matching normalizers
- Rebuilds production features and generates trading decisions
- Executes orders in paper trading or broker-connected modes
- Tracks live performance for frontend dashboard consumption

Author: YoopRL System
Date: November 8, 2025
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, UTC, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

try:  # Guard import so unit tests can run without full SB3
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.base_class import BaseAlgorithm
except ImportError as exc:  # pragma: no cover - hard failure if missing
    raise ImportError("stable_baselines3 must be installed for live trading") from exc

from database.db_manager import DatabaseManager
from data_download.feature_engineering import FeatureEngineering
from data_download.loader import download_history
from data_download.intraday_features import IntradayFeatureSpec, add_intraday_features
from data_download.intraday_loader import build_intraday_dataset
from utils.state_normalizer import StateNormalizer

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
except ImportError:  # pragma: no cover - fallback for older runtimes
    import pytz

    NY_TZ = pytz.timezone("America/New_York")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

# Acceptable action mapping used by the environments (0 = sell, 1 = hold, 2 = buy)
ACTION_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}


def _infer_benchmark_symbol(symbol: str) -> str:
    mapping = {
        "TQQQ": "QQQ",
        "SQQQ": "QQQ",
        "UPRO": "SPY",
        "SPXL": "SPY",
        "TNA": "IWM",
        "TMF": "TLT",
    }
    return mapping.get((symbol or "").upper(), "SPY")


@dataclass(slots=True)
class LiveTraderConfig:
    """Configuration payload for a single live trader instance."""

    agent_id: str
    agent_type: str
    symbol: str
    model_path: Path
    features_used: List[str]
    normalizer_path: Optional[Path] = None
    features_config: Optional[dict] = None
    initial_capital: float = 10_000.0
    max_position_pct: float = 0.5
    risk_per_trade: float = 0.02
    time_frame: str = "daily"
    bar_size: str = "1 day"
    lookback_days: int = 180
    check_frequency: str = "EOD"
    paper_trading: bool = True
    bridge_host: str = "localhost"
    bridge_port: int = 5080
    allow_premarket: bool = False
    allow_afterhours: bool = False
    data_frequency: str = "daily"
    interval: str = "1d"
    benchmark_symbol: Optional[str] = None
    metadata_path: Optional[Path] = None
    auto_restart: bool = True
    extras: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_metadata(
        cls,
        agent_id: str,
        metadata: Dict[str, object],
        overrides: Optional[Dict[str, object]] = None,
    ) -> "LiveTraderConfig":
        """Build a config object using persisted training metadata."""

        overrides = overrides or {}

        def _resolve_path(raw: object) -> Optional[Path]:
            if raw in (None, ""):
                return None
            path = Path(str(raw))
            if not path.is_absolute():
                path = (PROJECT_ROOT / path).resolve()
            else:
                path = path.resolve()
            return path

        model_path = _resolve_path(metadata.get("model_path"))
        if model_path is None:
            raise KeyError("model_path missing from metadata")

        features_used = list(metadata.get("features_used", []))
        normalizer_path = _resolve_path(metadata.get("normalizer_path"))
        metadata_path = _resolve_path(metadata.get("metadata_path"))

        def _to_bool(value: object) -> bool:
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

        def _coalesce_bool(*values: object) -> bool:
            for candidate in values:
                if candidate is not None:
                    return _to_bool(candidate)
            return False

        base_kwargs = {
            "agent_id": agent_id,
            "agent_type": str(metadata.get("agent_type", "PPO")),
            "symbol": str(metadata.get("symbol")),
            "model_path": model_path,
            "features_used": features_used,
            "normalizer_path": normalizer_path,
            "features_config": metadata.get("features"),
            "initial_capital": float(metadata.get("training_settings", {}).get("initial_capital", 10_000.0)),
            "metadata_path": metadata_path,
        }

        training_settings = dict(metadata.get("training_settings") or {})

        data_frequency = str(training_settings.get("data_frequency", "daily") or "daily").lower()
        interval = str(training_settings.get("interval", "1d") or "1d").lower()
        benchmark_symbol = (
            training_settings.get("benchmark_symbol")
            or metadata.get("benchmark_symbol")
            or _infer_benchmark_symbol(base_kwargs["symbol"])
        )

        base_kwargs.update(
            {
                "data_frequency": data_frequency,
                "interval": interval,
                "benchmark_symbol": benchmark_symbol.upper() if benchmark_symbol else None,
            }
        )

        extras_payload: Dict[str, object] = dict(metadata.get("extras") or {})
        extras_payload.setdefault("training_settings", training_settings)
        extras_payload.setdefault("benchmark_symbol", base_kwargs["benchmark_symbol"])
        extras_payload.setdefault("data_frequency", data_frequency)
        extras_payload.setdefault("interval", interval)
        extras_payload.setdefault("intraday_enabled", bool(training_settings.get("intraday_enabled")))
        if metadata_path and "metadata_path" not in extras_payload:
            extras_payload["metadata_path"] = str(metadata_path)

        auto_restart = _coalesce_bool(
            overrides.get("auto_restart"),
            extras_payload.get("auto_restart"),
            training_settings.get("auto_restart"),
            metadata.get("auto_restart"),
            True,
        )
        base_kwargs["auto_restart"] = auto_restart
        extras_payload["auto_restart"] = auto_restart

        allow_premarket = _coalesce_bool(
            overrides.get("allow_premarket"),
            extras_payload.get("allow_premarket"),
            training_settings.get("allow_premarket"),
            metadata.get("allow_premarket"),
        )
        allow_afterhours = _coalesce_bool(
            overrides.get("allow_afterhours"),
            extras_payload.get("allow_afterhours"),
            training_settings.get("allow_afterhours"),
            metadata.get("allow_afterhours"),
        )

        extras_payload["allow_premarket"] = allow_premarket
        extras_payload["allow_afterhours"] = allow_afterhours

        is_intraday = any(
            value in {"intraday", "15m", "15min"}
            for value in (
                data_frequency,
                interval,
                str(metadata.get("time_frame", "")).lower(),
                str(metadata.get("bar_size", "")).lower(),
            )
        )

        if is_intraday:
            base_kwargs.setdefault("time_frame", "intraday")
            base_kwargs.setdefault("bar_size", "15 min")
            base_kwargs.setdefault("lookback_days", int(training_settings.get("intraday_lookback_days", 10) or 10))
            base_kwargs.setdefault("check_frequency", "15min")
        else:
            base_kwargs.setdefault("time_frame", "daily")
            base_kwargs.setdefault("bar_size", "1 day")
            base_kwargs.setdefault("lookback_days", int(training_settings.get("lookback_days", 180) or 180))
            base_kwargs.setdefault("check_frequency", "EOD")

        base_kwargs["extras"] = extras_payload
        base_kwargs["allow_premarket"] = allow_premarket
        base_kwargs["allow_afterhours"] = allow_afterhours

        base_kwargs.update(overrides)

        base_kwargs["allow_premarket"] = _to_bool(base_kwargs.get("allow_premarket"))
        base_kwargs["allow_afterhours"] = _to_bool(base_kwargs.get("allow_afterhours"))
        base_kwargs["auto_restart"] = _to_bool(base_kwargs.get("auto_restart", True))

        if "extras" in overrides and overrides["extras"] is not None:
            merged_extras = dict(extras_payload)
            merged_extras.update(overrides["extras"])
            base_kwargs["extras"] = merged_extras

        extras_obj = base_kwargs.get("extras")
        if isinstance(extras_obj, dict):
            extras_obj["allow_premarket"] = base_kwargs["allow_premarket"]
            extras_obj["allow_afterhours"] = base_kwargs["allow_afterhours"]
            extras_obj["auto_restart"] = base_kwargs["auto_restart"]
            base_kwargs["extras"] = extras_obj

        return cls(**base_kwargs)

    def as_dict(self) -> Dict[str, object]:
        """Serialized form used by API responses."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "symbol": self.symbol,
            "model_path": str(self.model_path),
            "normalizer_path": str(self.normalizer_path) if self.normalizer_path else None,
            "features_used": list(self.features_used),
            "features_config": self.features_config or {},
            "initial_capital": self.initial_capital,
            "max_position_pct": self.max_position_pct,
            "risk_per_trade": self.risk_per_trade,
            "time_frame": self.time_frame,
            "bar_size": self.bar_size,
            "lookback_days": self.lookback_days,
            "check_frequency": self.check_frequency,
            "paper_trading": self.paper_trading,
            "bridge_host": self.bridge_host,
            "bridge_port": self.bridge_port,
            "allow_premarket": bool(self.allow_premarket),
            "allow_afterhours": bool(self.allow_afterhours),
            "data_frequency": self.data_frequency,
            "interval": self.interval,
            "benchmark_symbol": self.benchmark_symbol,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "auto_restart": bool(self.auto_restart),
            "extras": self.extras,
        }


class LiveTrader:
    """Runtime execution engine for a single RL trading agent."""

    def __init__(self, config: LiveTraderConfig, db_manager: Optional[DatabaseManager] = None):
        self.config = config
        self.logger = logging.getLogger(f"LiveTrader.{config.agent_id}")
        self.db: Optional[DatabaseManager] = db_manager

        # Artifacts populated during startup
        self.model: Optional[BaseAlgorithm] = None
        self.normalizer: Optional[StateNormalizer] = None
        self.feature_engineer: Optional[FeatureEngineering] = None
        self.broker = None  # Lazy import of IBKR adapter when needed
        self.expected_obs_dim: Optional[int] = None
        self._observation_columns: Optional[List[str]] = None

        # Trading state
        self.is_running: bool = False
        self.paper_trading: bool = bool(config.paper_trading)
        self.auto_restart: bool = bool(getattr(config, "auto_restart", True))
        self.allow_premarket: bool = bool(getattr(config, "allow_premarket", False))
        self.allow_afterhours: bool = bool(getattr(config, "allow_afterhours", False))
        self.config.allow_premarket = self.allow_premarket
        self.config.allow_afterhours = self.allow_afterhours
        self.config.auto_restart = self.auto_restart
        self.current_position: int = 0
        self.entry_price: float = 0.0
        self.current_price: float = 0.0
        self.current_capital: float = float(config.initial_capital)
        self.total_pnl: float = 0.0
        self.last_action: Optional[int] = None
        self.last_run_at: Optional[datetime] = None
        self.last_error: Optional[str] = None

        self._trades: List[Dict[str, object]] = []
        self._equity_curve: List[Tuple[datetime, float]] = [(datetime.now(UTC), self.current_capital)]

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Load artifacts, connect to broker (if requested), and mark trader active."""

        self.logger.info("Starting live trader for %s (%s)", self.config.symbol, self.config.agent_type)
        self.last_error = None
        try:
            self._load_model()
            self._load_normalizer()
            if not self._using_intraday():
                self._init_feature_engineer()
            if not self.paper_trading:
                self._connect_broker()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Failed to start trader: %s", exc, exc_info=True)
            self.last_error = str(exc)
            self.is_running = False
            return False

        self.is_running = True
        self.last_run_at = None
        self.logger.info("Live trader %s ready", self.config.agent_id)
        return True

    def stop(self) -> None:
        """Stop the trader and release broker resources."""

        self.logger.info("Stopping trader %s", self.config.agent_id)
        self.is_running = False
        if self.broker:
            try:
                stop = getattr(self.broker, "stop_monitoring", None)
                if callable(stop):
                    stop()
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.debug("Error stopping broker monitor: %s", exc)
        self.broker = None

    # ------------------------------------------------------------------
    # Public execution surface
    # ------------------------------------------------------------------
    def run_single_check(self) -> bool:
        """Execute one trading cycle: fetch data, score action, and (paper) trade."""

        if not self.is_running:
            self.logger.warning("run_single_check called while trader is stopped")
            return False

        try:
            features, price = self._prepare_latest_observation()
            if features is None:
                self._log_invalid_observation(reason="Missing feature window")
                return False

            if not self._validate_observation(features):
                return False

            action = self._predict_action(features)
            self._execute_action(action, price, features)

            self.last_run_at = datetime.now(UTC)
            self.last_action = action
            self.current_price = price
            self._push_equity_snapshot(price)
            return True
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Live check failed: %s", exc, exc_info=True)
            self._persist_risk_event(
                event_type="EXECUTION_ERROR",
                severity="CRITICAL",
                description=str(exc),
            )
            return False

    def close_position(self) -> bool:
        """Force liquidate any open position (paper simulation for now)."""

        if self.current_position == 0:
            self.logger.info("No open position to close for %s", self.config.symbol)
            return True

        self.logger.info("Force closing %s position", self.config.symbol)
        self._execute_action(0, self.current_price or 0.0)
        return True

    def should_run_now(self, moment: Optional[datetime] = None) -> bool:
        """Determine if the trader is allowed to execute at the provided timestamp."""

        if not self.is_running:
            return False

        if not self._using_intraday():
            return True

        now_utc = moment or datetime.now(UTC)
        local_now = now_utc.astimezone(NY_TZ)

        if local_now.weekday() >= 5:  # Saturday/Sunday
            return False

        time_of_day = local_now.time()

        rth_start = dtime(9, 30)
        rth_end = dtime(16, 0)
        if rth_start <= time_of_day < rth_end:
            return True

        pre_start = dtime(7, 0)
        if self.allow_premarket and pre_start <= time_of_day < rth_start:
            return True

        after_end = dtime(20, 0)
        if self.allow_afterhours and rth_end <= time_of_day <= after_end:
            return True

        return False

    def get_status(self) -> Dict[str, object]:
        """Summarize live state for API/Frontend consumption."""

        portfolio_value = self.current_capital + (self.current_position * self.current_price)
        return {
            "agent_id": self.config.agent_id,
            "symbol": self.config.symbol,
            "agent_type": self.config.agent_type,
            "is_running": self.is_running,
            "paper_trading": self.paper_trading,
            "allow_premarket": bool(self.allow_premarket),
            "allow_afterhours": bool(self.allow_afterhours),
            "auto_restart": bool(self.auto_restart),
            "current_position": self.current_position,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "portfolio_value": portfolio_value,
            "initial_capital": self.config.initial_capital,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": (self.total_pnl / self.config.initial_capital * 100.0) if self.config.initial_capital else 0.0,
            "last_action": ACTION_NAMES.get(self.last_action, "UNKNOWN"),
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_error": self.last_error,
            "trades": list(self._trades[-20:]),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        agent_type = self.config.agent_type.upper()
        path = self.config.model_path
        if not path.exists():
            alt = path.with_suffix("")
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")

        self.logger.debug("Loading %s model from %s", agent_type, path)
        if agent_type == "PPO":
            self.model = PPO.load(str(path))
        elif agent_type == "SAC":
            self.model = SAC.load(str(path))
        else:
            raise ValueError(f"Unsupported agent_type '{agent_type}' for live trading")

        shape = getattr(self.model.observation_space, "shape", None)
        if shape and len(shape) > 0:
            self.expected_obs_dim = int(shape[0])

    def _load_normalizer(self) -> None:
        if not self.config.normalizer_path:
            self.logger.warning("No normalizer_path provided, skipping normalization")
            return

        normalizer = StateNormalizer(method="zscore")
        normalizer.load_params(self.config.normalizer_path)
        self.normalizer = normalizer
        self.logger.debug("Loaded normalizer from %s", self.config.normalizer_path)

    def _init_feature_engineer(self) -> None:
        feature_config = self.config.features_config or {}
        self.feature_engineer = FeatureEngineering(
            symbol=self.config.symbol,
            enable_multi_asset=bool(feature_config.get("multi_asset", {}).get("enabled")) if isinstance(feature_config.get("multi_asset"), dict) else False,
            multi_asset_symbols=feature_config.get("multi_asset", {}).get("symbols") if isinstance(feature_config.get("multi_asset"), dict) else None,
            enable_sentiment=bool(feature_config.get("sentiment", {}).get("enabled")) if isinstance(feature_config.get("sentiment"), dict) else bool(feature_config.get("sentiment")),
            feature_config=feature_config,
        )

    def _connect_broker(self) -> None:
        try:
            from IBKR_Bridge.python_adapter.interreact_bridge_adapter import InterReactBridgeAdapter
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning(
                "IBKR bridge unavailable (%s). Reverting to paper trading mode.",
                exc,
            )
            self.paper_trading = True
            return

        adapter = InterReactBridgeAdapter(
            host=self.config.bridge_host,
            port=self.config.bridge_port,
            auto_start_timer=False,
        )
        if not adapter.is_connected():
            self.logger.warning("Bridge connection failed. Paper trading mode enabled.")
            self.paper_trading = True
            return

        self.broker = adapter
        self.logger.info("Connected to IBKR bridge at %s:%s", self.config.bridge_host, self.config.bridge_port)

    def _prepare_latest_observation(self) -> Tuple[Optional[np.ndarray], float]:
        frame: Optional[pd.DataFrame]
        if self._using_intraday():
            frame = self._build_intraday_frame()
        else:
            frame = self._build_daily_frame()

        if frame is None or frame.empty:
            self.logger.error("Unable to construct live feature frame for %s", self.config.symbol)
            return None, 0.0

        latest_row = frame.iloc[-1].copy()
        price = self._resolve_last_price(latest_row)

        position_value = self.current_position * price
        total_value = self.current_capital + position_value

        if "position_context" in latest_row.index:
            latest_row["position_context"] = position_value / total_value if total_value > 0 else 0.0

        feature_order = list(self.config.features_used)
        observation_columns = self._resolve_observation_columns(frame)
        extra_columns = [col for col in observation_columns if col not in feature_order]

        feature_series = latest_row.reindex(feature_order, fill_value=0.0)
        feature_values = feature_series.astype(np.float32, copy=False).to_numpy()

        if self.normalizer is not None and feature_values.size:
            try:
                feature_values = (
                    self.normalizer.transform(feature_values.reshape(1, -1))
                    .astype(np.float32)
                    .flatten()
                )
            except Exception as exc:  # pylint: disable=broad-except
                self.logger.debug("Normalizer transform failed: %s", exc)
                feature_values = feature_series.astype(np.float32, copy=False).to_numpy()

        extra_values = np.array([], dtype=np.float32)
        if extra_columns:
            extras = latest_row.reindex(extra_columns, fill_value=0.0).astype(np.float32, copy=False)
            extra_values = extras.to_numpy()

        portfolio_state = np.array(
            [
                self.current_capital / (self.config.initial_capital or 1.0),
                position_value / (self.config.initial_capital or 1.0),
                self.current_position / 1000.0,
                total_value / (self.config.initial_capital or 1.0),
            ],
            dtype=np.float32,
        )

        observation = np.concatenate((portfolio_state, feature_values, extra_values)).astype(np.float32)

        return observation, price

    def _validate_observation(self, features: Optional[np.ndarray]) -> bool:
        """Ensure observation shape matches the policy expectation."""

        if features is None:
            self._log_invalid_observation(reason="Features is None")
            return False

        if isinstance(features, (list, tuple)):
            features = np.asarray(features, dtype=float)

        if not isinstance(features, np.ndarray):
            self._log_invalid_observation(reason=f"Unexpected feature type {type(features)}")
            return False

        if features.ndim > 1:
            try:
                features = features.reshape(-1)
            except Exception:  # pragma: no cover - reshape failure
                self._log_invalid_observation(reason=f"Cannot reshape observation with shape {features.shape}")
                return False

        expected = self.expected_obs_dim
        if expected is None:
            self.expected_obs_dim = int(features.shape[0])
            return True

        if int(features.shape[0]) == int(expected):
            return True

        self._log_invalid_observation(
            reason=f"Unexpected observation shape {features.shape}; expected ({expected},)"
        )
        return False

    def _log_invalid_observation(self, reason: str) -> None:
        self.logger.error("Invalid observation: %s", reason)
        self.last_error = reason
        self._persist_risk_event(
            event_type="INVALID_OBSERVATION",
            severity="CRITICAL",
            description=reason,
        )

    def _fetch_daily_window(self) -> Optional[pd.DataFrame]:
        period = self._resolve_period(self.config.lookback_days)
        self.logger.debug("Downloading %s data for %s (period=%s)", self.config.time_frame, self.config.symbol, period)
        try:
            df = download_history(self.config.symbol, period=period)
            return df.tail(self.config.lookback_days or 180)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Failed to download data: %s", exc)
            return None

    def _build_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not self.feature_engineer:
            raise RuntimeError("Feature engineer not initialised")
        try:
            processed = self.feature_engineer.process(df.copy())
            # Ensure we keep only expected feature columns for determinism
            missing = [f for f in self.config.features_used if f not in processed.columns]
            if missing:
                self.logger.warning("Missing features in live set: %s", ", ".join(missing))
                for feature in missing:
                    processed[feature] = 0.0
            return processed
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Feature engineering failed: %s", exc)
            return None

    def _build_daily_frame(self) -> Optional[pd.DataFrame]:
        window = self._fetch_daily_window()
        if window is None or window.empty:
            return None
        return self._build_features(window)

    def _build_intraday_frame(self) -> Optional[pd.DataFrame]:
        extras = self.config.extras if isinstance(self.config.extras, dict) else {}
        benchmark = (
            self.config.benchmark_symbol
            or extras.get("benchmark_symbol")
            or _infer_benchmark_symbol(self.config.symbol)
        )

        interval = (self.config.interval or "15m").lower()
        lookback_days = int(max(3, min(self.config.lookback_days or 10, 30)))
        today = date.today()
        start_date = today - timedelta(days=lookback_days)

        try:
            dataset = build_intraday_dataset(
                (self.config.symbol.upper(), benchmark.upper()),
                interval=interval,
                start=start_date,
                end=today,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Failed to build intraday dataset for %s: %s", self.config.symbol, exc)
            return None

        if dataset.empty:
            return None

        dataset = dataset.sort_index()
        max_rows = max(1, lookback_days * 32)  # ~2 bars per hour across sessions
        if len(dataset) > max_rows:
            dataset = dataset.iloc[-max_rows:]

        spec = IntradayFeatureSpec(
            primary_symbol=self.config.symbol.upper(),
            benchmark_symbol=benchmark.upper(),
        )

        dataset = add_intraday_features(dataset, spec)
        dataset = dataset.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        primary_prefix = self.config.symbol.lower()
        primary_close_col = f"{primary_prefix}_close"
        if primary_close_col in dataset.columns:
            base_close = dataset[primary_close_col]
            dataset["close"] = base_close
            dataset["price"] = base_close

        return dataset

    def _resolve_observation_columns(self, frame: pd.DataFrame) -> List[str]:
        if self._observation_columns:
            cached = [col for col in self._observation_columns if col in frame.columns]
            if cached and len(cached) == len(self._observation_columns):
                return cached

        base_order = list(self.config.features_used)
        extras: List[str] = []

        for column in frame.columns:
            if column in base_order:
                continue
            if column.lower() == 'session_date':
                continue
            try:
                if not is_numeric_dtype(frame[column]):
                    continue
            except Exception:  # pragma: no cover - dtype inference fallback
                continue
            extras.append(column)

        ordered = base_order + [col for col in extras if col not in base_order]
        self._observation_columns = ordered
        return ordered

    def _predict_action(self, observation: np.ndarray) -> int:
        if not self.model:
            raise RuntimeError("Model not loaded")
        action, _ = self.model.predict(observation.reshape(1, -1), deterministic=True)
        return int(action)

    def _execute_action(self, action: int, price: float, features: Optional[np.ndarray] = None) -> None:
        price = max(price, 0.0)
        quantity = self._determine_position_size(price)

        executed_quantity = 0
        if action == 2:
            executed_quantity = self._handle_buy(quantity, price)
        elif action == 0:
            executed_quantity = self._handle_sell(price)
        else:
            self.logger.debug("Holding position (%s)", self.config.symbol)

        log_quantity = executed_quantity if action in (0, 2) else 0
        self._persist_action_event(action, price, log_quantity, features)

    def _determine_position_size(self, price: float) -> int:
        if price <= 0:
            return 0
        max_shares = int((self.current_capital * self.config.max_position_pct) / price)
        risk_shares = int((self.current_capital * self.config.risk_per_trade) / price)
        shares = max(0, min(max_shares, risk_shares))
        return shares or (1 if self.current_capital >= price else 0)

    def _handle_buy(self, shares: int, price: float) -> int:
        if shares <= 0:
            self.logger.info("Buy signal ignored (no capital available)")
            return 0
        if self.current_position > 0:
            self.logger.debug("Already in position, skipping buy")
            return 0

        cost = shares * price
        self.current_capital -= cost
        self.current_position = shares
        self.entry_price = price
        self.total_pnl -= self.config.risk_per_trade * self.current_capital  # Track reserved risk budget

        self._log_trade("BUY", shares, price)
        if not self.paper_trading and self.broker:
            self._submit_order("BUY", shares)

        return shares

    def _handle_sell(self, price: float) -> int:
        if self.current_position <= 0:
            self.logger.debug("Sell signal with no position")
            return 0

        shares = self.current_position
        proceeds = shares * price
        pnl = (price - self.entry_price) * shares
        self.current_capital += proceeds
        self.total_pnl += pnl
        self._log_trade("SELL", shares, price, pnl)

        if not self.paper_trading and self.broker:
            self._submit_order("SELL", shares)

        self.current_position = 0
        self.entry_price = 0.0
        return shares

    def _submit_order(self, action: str, quantity: int) -> None:
        if not self.broker:
            return
        try:
            result = self.broker.place_order(
                symbol=self.config.symbol,
                action=action,
                quantity=quantity,
                order_type="MKT",
            )
            self.logger.info("Broker order response: %s", result)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Broker order failed: %s", exc)

    def _log_trade(self, action: str, quantity: int, price: float, pnl: float = 0.0) -> None:
        trade = {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": action,
            "quantity": quantity,
            "price": price,
            "pnl": pnl,
            "paper": self.paper_trading or not bool(self.broker),
        }
        self._trades.append(trade)
        self.logger.info("%s %s x%d @ %.2f (PnL %.2f)", action, self.config.symbol, quantity, price, pnl)

    def _push_equity_snapshot(self, price: float) -> None:
        portfolio_value = self.current_capital + (self.current_position * price)
        self._equity_curve.append((datetime.now(UTC), portfolio_value))

    def _persist_action_event(
        self,
        action: int,
        price: float,
        quantity: int,
        state_features: Optional[np.ndarray] = None,
    ) -> None:
        if self.db is None:
            return

        try:
            action_name = ACTION_NAMES.get(action, str(action))
            state_payload = self._build_state_payload(state_features)
            self.db.log_agent_action(
                agent_name=self.config.agent_id,
                symbol=self.config.symbol,
                action=action_name,
                quantity=quantity,
                price=price,
                reward=None,
                rationale=None,
                confidence=None,
                state=state_payload,
            )
        except Exception:  # pragma: no cover - logging should not interrupt trading loop
            self.logger.debug(
                "Failed to persist action event for %s", self.config.agent_id, exc_info=True
            )

    def _persist_risk_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        *,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        if self.db is None:
            return

        try:
            self.db.log_risk_event(
                event_type=event_type,
                severity=severity,
                description=description,
                agent_name=self.config.agent_id,
                symbol=self.config.symbol,
                value=value,
                threshold=threshold,
            )
        except Exception:  # pragma: no cover
            self.logger.debug(
                "Failed to persist risk event for %s", self.config.agent_id, exc_info=True
            )

    def _build_state_payload(self, features: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if features is None:
            return None

        try:
            values = features.tolist()
        except AttributeError:
            values = list(features)

        names = list(self.config.features_used)
        if len(names) == len(values):
            return {name: float(values[idx]) for idx, name in enumerate(names)}

        return {
            "values": [float(val) for val in values],
            "feature_count": len(values),
        }

    def _resolve_last_price(self, latest_row: pd.Series) -> float:
        symbol_lower = self.config.symbol.lower()
        candidates = [
            f"{symbol_lower}_close",
            f"{symbol_lower}_price",
            "Close",
            "price",
            "close",
        ]

        for field in candidates:
            if field in latest_row.index and pd.notna(latest_row[field]):
                try:
                    return float(latest_row[field])
                except (TypeError, ValueError):
                    continue

        numeric_candidates = pd.to_numeric(latest_row, errors="coerce")
        numeric_candidates = numeric_candidates.dropna()
        if not numeric_candidates.empty:
            return float(numeric_candidates.iloc[-1])
        return 0.0

    def _using_intraday(self) -> bool:
        flags = {
            str(self.config.data_frequency or "").lower(),
            str(self.config.interval or "").lower(),
            str(self.config.time_frame or "").lower(),
            str(self.config.bar_size or "").lower(),
        }
        return any(flag in {"intraday", "15m", "15min", "15 min"} for flag in flags)

    @staticmethod
    def _resolve_period(days: int) -> str:
        if days <= 30:
            return "3mo"
        if days <= 180:
            return "1y"
        if days <= 365:
            return "2y"
        if days <= 730:
            return "5y"
        return "10y"
