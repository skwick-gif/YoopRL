"""Intraday trading environment tailored for SAC + DSR experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

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
        commission: float = 0.5,
        max_position_size: float = 1.0,
        normalize_obs: bool = True,
        history_config: Optional[dict] = None,
        sampler: Optional[IntradaySessionSampler] = None
    ):
        if 'session_date' not in df.columns:
            raise ValueError("IntradayEquityEnv requires 'session_date' column in dataframe")

        self.master_df = df.copy()
        self.session_dates: List[pd.Timestamp] = sorted(df['session_date'].dropna().unique())
        if not self.session_dates:
            raise ValueError("IntradayEquityEnv requires at least one session")

        self.sampler = sampler or IntradaySessionSampler(shuffle=True)
        self.active_session_idx: Optional[int] = None
        self._prev_total_value = initial_capital

        initial_session = self._prepare_session_dataframe(0)

        super().__init__(
            df=initial_session,
            initial_capital=initial_capital,
            commission=commission,
            max_position_size=max_position_size,
            normalize_obs=normalize_obs,
            history_config=history_config
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        next_idx = self.sampler.next_index(len(self.session_dates), self.active_session_idx)
        self.active_session_idx = next_idx
        self.df = self._prepare_session_dataframe(next_idx)
        self.n_steps = len(self.df)
        self._prev_total_value = self.initial_capital
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def step(self, action: int):  # type: ignore[override]
        forced_exit = self._is_last_step() and self.holdings > 0 and action != 2
        adjusted_action = 2 if forced_exit else action
        obs, reward, terminated, truncated, info = super().step(adjusted_action)

        row_idx = max(self.current_step - 1, 0)
        row = self.df.iloc[row_idx]

        info = dict(info)
        info['session_date'] = str(self.session_dates[self.active_session_idx])
        info['forced_exit'] = bool(forced_exit)
        info['time_fraction'] = float(row.get('time_fraction', 0.0))
        info['bar_index'] = int(row.get('bar_index', row_idx))
        info['minutes_from_open'] = float(row.get('minutes_from_open', 0.0))

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action: int, current_price: float) -> float:
        prev = self._prev_total_value if self._prev_total_value != 0 else 1e-8
        reward = (self.total_value - self._prev_total_value) / prev
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
