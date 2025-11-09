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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # Guard import so unit tests can run without full SB3
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.base_class import BaseAlgorithm
except ImportError as exc:  # pragma: no cover - hard failure if missing
    raise ImportError("stable_baselines3 must be installed for live trading") from exc

from data_download.feature_engineering import FeatureEngineering
from data_download.loader import download_history
from utils.state_normalizer import StateNormalizer

logger = logging.getLogger(__name__)

# Acceptable action mapping used by the environments (0 = sell, 1 = hold, 2 = buy)
ACTION_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}


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
    metadata_path: Optional[Path] = None
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
        model_path = Path(str(metadata["model_path"]))
        features_used = list(metadata.get("features_used", []))
        normalizer_path = metadata.get("normalizer_path")

        base_kwargs = {
            "agent_id": agent_id,
            "agent_type": str(metadata.get("agent_type", "PPO")),
            "symbol": str(metadata.get("symbol")),
            "model_path": model_path,
            "features_used": features_used,
            "normalizer_path": Path(normalizer_path) if normalizer_path else None,
            "features_config": metadata.get("features"),
            "initial_capital": float(metadata.get("training_settings", {}).get("initial_capital", 10_000.0)),
            "metadata_path": Path(metadata.get("metadata_path", "")) if metadata.get("metadata_path") else None,
        }
        base_kwargs.update(overrides)
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
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "extras": self.extras,
        }


class LiveTrader:
    """Runtime execution engine for a single RL trading agent."""

    def __init__(self, config: LiveTraderConfig):
        self.config = config
        self.logger = logging.getLogger(f"LiveTrader.{config.agent_id}")

        self.model: Optional[BaseAlgorithm] = None
        self.normalizer: Optional[StateNormalizer] = None
        self.feature_engineer: Optional[FeatureEngineering] = None
        self.broker = None  # Lazy import of IBKR adapter when needed

        # Trading state
        self.is_running: bool = False
        self.paper_trading: bool = bool(config.paper_trading)
        self.current_position: int = 0
        self.entry_price: float = 0.0
        self.current_price: float = 0.0
        self.current_capital: float = float(config.initial_capital)
        self.total_pnl: float = 0.0
        self.last_action: Optional[int] = None
        self.last_run_at: Optional[datetime] = None

        self._trades: List[Dict[str, object]] = []
        self._equity_curve: List[Tuple[datetime, float]] = [(datetime.utcnow(), self.current_capital)]

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Load artifacts, connect to broker (if requested), and mark trader active."""

        self.logger.info("Starting live trader for %s (%s)", self.config.symbol, self.config.agent_type)
        try:
            self._load_model()
            self._load_normalizer()
            self._init_feature_engineer()
            if not self.paper_trading:
                self._connect_broker()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Failed to start trader: %s", exc, exc_info=True)
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
                return False

            action = self._predict_action(features)
            self._execute_action(action, price)

            self.last_run_at = datetime.utcnow()
            self.last_action = action
            self.current_price = price
            self._push_equity_snapshot(price)
            return True
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error("Live check failed: %s", exc, exc_info=True)
            return False

    def close_position(self) -> bool:
        """Force liquidate any open position (paper simulation for now)."""

        if self.current_position == 0:
            self.logger.info("No open position to close for %s", self.config.symbol)
            return True

        self.logger.info("Force closing %s position", self.config.symbol)
        self._execute_action(0, self.current_price or 0.0)
        return True

    def get_status(self) -> Dict[str, object]:
        """Summarize live state for API/Frontend consumption."""

        portfolio_value = self.current_capital + (self.current_position * self.current_price)
        return {
            "agent_id": self.config.agent_id,
            "symbol": self.config.symbol,
            "agent_type": self.config.agent_type,
            "is_running": self.is_running,
            "paper_trading": self.paper_trading,
            "current_position": self.current_position,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "portfolio_value": portfolio_value,
            "initial_capital": self.config.initial_capital,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": (self.total_pnl / self.config.initial_capital * 100.0) if self.config.initial_capital else 0.0,
            "last_action": ACTION_NAMES.get(self.last_action, "UNKNOWN"),
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
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
        df = self._download_window()
        if df is None or df.empty:
            self.logger.error("No market data available for %s", self.config.symbol)
            return None, 0.0

        processed = self._build_features(df)
        if processed is None or processed.empty:
            self.logger.error("Feature engineering returned empty dataset for %s", self.config.symbol)
            return None, 0.0

        latest_row = processed.iloc[-1]
        price = float(latest_row.get("Close", latest_row.get("price", 0.0)))

        features = latest_row.reindex(self.config.features_used).values.astype(np.float32)
        if self.normalizer is not None:
            features = self.normalizer.transform(features.reshape(1, -1)).astype(np.float32).flatten()

        return features, price

    def _download_window(self) -> Optional[pd.DataFrame]:
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

    def _predict_action(self, observation: np.ndarray) -> int:
        if not self.model:
            raise RuntimeError("Model not loaded")
        action, _ = self.model.predict(observation.reshape(1, -1), deterministic=True)
        return int(action)

    def _execute_action(self, action: int, price: float) -> None:
        price = max(price, 0.0)
        quantity = self._determine_position_size(price)

        if action == 2:
            self._handle_buy(quantity, price)
        elif action == 0:
            self._handle_sell(price)
        else:
            self.logger.debug("Holding position (%s)", self.config.symbol)

    def _determine_position_size(self, price: float) -> int:
        if price <= 0:
            return 0
        max_shares = int((self.current_capital * self.config.max_position_pct) / price)
        risk_shares = int((self.current_capital * self.config.risk_per_trade) / price)
        shares = max(0, min(max_shares, risk_shares))
        return shares or (1 if self.current_capital >= price else 0)

    def _handle_buy(self, shares: int, price: float) -> None:
        if shares <= 0:
            self.logger.info("Buy signal ignored (no capital available)")
            return
        if self.current_position > 0:
            self.logger.debug("Already in position, skipping buy")
            return

        cost = shares * price
        self.current_capital -= cost
        self.current_position = shares
        self.entry_price = price
        self.total_pnl -= self.config.risk_per_trade * self.current_capital  # Track reserved risk budget

        self._log_trade("BUY", shares, price)
        if not self.paper_trading and self.broker:
            self._submit_order("BUY", shares)

    def _handle_sell(self, price: float) -> None:
        if self.current_position <= 0:
            self.logger.debug("Sell signal with no position")
            return

        proceeds = self.current_position * price
        pnl = (price - self.entry_price) * self.current_position
        self.current_capital += proceeds
        self.total_pnl += pnl
        self._log_trade("SELL", self.current_position, price, pnl)

        if not self.paper_trading and self.broker:
            self._submit_order("SELL", self.current_position)

        self.current_position = 0
        self.entry_price = 0.0

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
            "timestamp": datetime.utcnow().isoformat(),
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
        self._equity_curve.append((datetime.utcnow(), portfolio_value))

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
