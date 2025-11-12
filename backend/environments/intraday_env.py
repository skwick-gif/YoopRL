"""Intraday trading environment tailored for SAC + DSR experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from environments.base_env import BaseTradingEnv


@dataclass
class IntradaySessionSampler:
    """Sampling policy for intraday sessions."""

    shuffle: bool = True
    sequential: bool = False

    def next_index(self, total: int, previous: Optional[int]) -> int:
        if total <= 0:
            raise ValueError("No sessions available for sampling")
        if total == 1:
            return 0
        if self.shuffle:
            choices = list(range(total))
            if previous is not None and previous < total:
                choices.pop(previous)
            return random.choice(choices)
        if self.sequential:
            return 0 if previous is None else (previous + 1) % total
        return random.randrange(total)


class IntradayEquityEnv(BaseTradingEnv):
    """Equity trading environment operating on 15-minute sessions."""

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        commission: Union[float, Dict[str, float]] = 0.5,
        max_position_size: float = 1.0,
        normalize_obs: bool = True,
        history_config: Optional[dict] = None,
        sampler: Optional[IntradaySessionSampler] = None,
        slippage_config: Optional[dict] = None,
        forced_exit_minutes: Optional[float] = 375.0,
        forced_exit_tolerance: float = 1.0,
        forced_exit_column: Optional[str] = None
    ):
        if 'session_date' not in df.columns:
            raise ValueError("IntradayEquityEnv requires 'session_date' column in dataframe")

        self.master_df = df.copy()
        self.session_dates: List[pd.Timestamp] = sorted(df['session_date'].dropna().unique())
        if not self.session_dates:
            raise ValueError("IntradayEquityEnv requires at least one session")

        timing_candidates = {'minutes_from_open', 'time_fraction', 'timestamp'}
        if forced_exit_column:
            timing_candidates.add(forced_exit_column)
        has_timing_signal = any(col in self.master_df.columns for col in timing_candidates)
        if not has_timing_signal:
            raise ValueError(
                "IntradayEquityEnv requires intraday timing metadata (minutes_from_open, "
                "time_fraction, or timestamp) to enforce forced exits"
            )

        self.sampler = sampler or IntradaySessionSampler(shuffle=True)
        self.active_session_idx: Optional[int] = None
        self._prev_total_value = initial_capital
        self._last_cost_ratio = 0.0
        self._last_forced_exit_reason: Optional[str] = None

        self.forced_exit_column = forced_exit_column
        self._session_minutes = 390.0
        self.forced_exit_minutes = float(forced_exit_minutes) if forced_exit_minutes is not None else None
        if self.forced_exit_minutes is not None:
            self.forced_exit_minutes = max(0.0, self.forced_exit_minutes)
        self.forced_exit_tolerance = max(0.0, float(forced_exit_tolerance))
        self.forced_exit_fraction = (
            min(1.0, (self.forced_exit_minutes or self._session_minutes) / self._session_minutes)
            if self.forced_exit_minutes is not None
            else None
        )

        initial_session = self._prepare_session_dataframe(0)

        super().__init__(
            df=initial_session,
            initial_capital=initial_capital,
            commission=commission,
            max_position_size=max_position_size,
            normalize_obs=normalize_obs,
            history_config=history_config,
            slippage_config=slippage_config
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        next_idx = self.sampler.next_index(len(self.session_dates), self.active_session_idx)
        self.active_session_idx = next_idx
        self.df = self._prepare_session_dataframe(next_idx)
        self.n_steps = len(self.df)
        self._prev_total_value = self.initial_capital
        self._last_cost_ratio = 0.0
        self._last_forced_exit_reason = None
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def step(self, action: int):  # type: ignore[override]
        current_row = self.df.iloc[self.current_step] if self.current_step < len(self.df) else None
        forced_exit = self._should_force_exit(action, current_row)
        adjusted_action = 2 if forced_exit else action
        prev_total_value = self._prev_total_value
        obs, reward, terminated, truncated, info = super().step(adjusted_action)

        row_idx = max(self.current_step - 1, 0)
        row = self.df.iloc[row_idx]

        info = dict(info)
        info['session_date'] = str(self.session_dates[self.active_session_idx])
        info['forced_exit'] = bool(forced_exit)
        info['forced_exit_reason'] = self._last_forced_exit_reason
        info['original_action'] = int(action)
        info['time_fraction'] = float(row.get('time_fraction', 0.0))
        info['bar_index'] = int(row.get('bar_index', row_idx))
        info['minutes_from_open'] = float(row.get('minutes_from_open', 0.0))
        info['prev_total_value'] = float(prev_total_value)
        info['transaction_cost_ratio'] = float(self._last_cost_ratio)

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action: int, current_price: float) -> float:
        prev = self._prev_total_value if self._prev_total_value != 0 else 1e-8
        transaction_costs = getattr(self, '_last_transaction_cost', 0.0)
        gross_total = self.total_value + transaction_costs
        gross_return = (gross_total - self._prev_total_value) / prev
        cost_ratio = transaction_costs / prev
        reward = gross_return - cost_ratio
        self._last_cost_ratio = float(cost_ratio)
        self._prev_total_value = self.total_value
        return float(reward)

    def _is_last_step(self) -> bool:
        return self.current_step >= (self.n_steps - 1)

    def _prepare_session_dataframe(self, session_idx: int) -> pd.DataFrame:
        session_date = self.session_dates[session_idx]
        mask = self.master_df['session_date'] == session_date
        session_df = self.master_df.loc[mask].copy()
        session_df = session_df.drop(columns=['session_date'], errors='ignore')
        session_df.reset_index(drop=True, inplace=True)
        return session_df

    def _should_force_exit(self, action: int, row: Optional[pd.Series]) -> bool:
        self._last_forced_exit_reason = None

        if action == 2 or self.holdings <= 0:
            return False

        if row is None:
            return False

        # Explicit override column
        if self.forced_exit_column and self.forced_exit_column in row.index:
            value = row.get(self.forced_exit_column)
            if self._value_triggers_force_exit(value):
                self._last_forced_exit_reason = self.forced_exit_column
                return True

        if 'minutes_from_open' in row.index and self._value_triggers_force_exit(row.get('minutes_from_open')):
            self._last_forced_exit_reason = 'minutes_from_open'
            return True

        if 'timestamp' in row.index and self._timestamp_triggers_force_exit(row.get('timestamp')):
            self._last_forced_exit_reason = 'timestamp'
            return True

        if 'time_fraction' in row.index and self._fraction_triggers_force_exit(row.get('time_fraction')):
            self._last_forced_exit_reason = 'time_fraction'
            return True

        if 'is_session_end' in row.index and bool(row.get('is_session_end')):
            self._last_forced_exit_reason = 'is_session_end'
            return True

        return False

    def _value_triggers_force_exit(self, raw_value: Optional[float]) -> bool:
        if self.forced_exit_minutes is None:
            return False
        if raw_value is None:
            return False

        try:
            minutes = float(raw_value)
        except (TypeError, ValueError):
            return False

        threshold = self.forced_exit_minutes - self.forced_exit_tolerance
        return minutes >= threshold

    def _fraction_triggers_force_exit(self, raw_value: Optional[float]) -> bool:
        if self.forced_exit_fraction is None or raw_value is None:
            return False

        try:
            fraction = float(raw_value)
        except (TypeError, ValueError):
            return False

        if np.isnan(fraction):
            return False

        threshold = self.forced_exit_fraction - 1e-4
        return fraction >= threshold

    def _timestamp_triggers_force_exit(self, raw_value: Optional[pd.Timestamp]) -> bool:
        if self.forced_exit_minutes is None or raw_value is None:
            return False

        timestamp = raw_value
        if not isinstance(timestamp, pd.Timestamp):
            try:
                timestamp = pd.Timestamp(timestamp)
            except (TypeError, ValueError):
                return False

        market_open_minutes = 9 * 60 + 30
        minutes_from_open = timestamp.hour * 60 + timestamp.minute - market_open_minutes
        return self._value_triggers_force_exit(minutes_from_open)
