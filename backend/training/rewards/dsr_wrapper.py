"""Differential Sharpe Ratio reward wrapper for intraday SAC experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np


@dataclass
class DSRConfig:
    """Configuration for Differential Sharpe Ratio reward shaping."""

    decay: float = 0.94
    epsilon: float = 1e-9
    warmup_steps: int = 200
    clip_value: Optional[float] = None
    entry_bonus: float = 0.0
    entry_min_position: float = 0.0

    def validate(self) -> None:
        if not 0.0 < self.decay < 1.0:
            raise ValueError("decay must be within (0, 1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.clip_value is not None and self.clip_value <= 0:
            raise ValueError("clip_value must be positive when provided")
        if self.entry_bonus < 0:
            raise ValueError("entry_bonus must be >= 0")
        if self.entry_min_position < 0:
            raise ValueError("entry_min_position must be >= 0")


class DSRRewardWrapper(gym.Wrapper):
    """Gym wrapper that replaces environment rewards with Differential Sharpe Ratio."""

    def __init__(self, env: gym.Env, config: Optional[DSRConfig] = None):
        super().__init__(env)
        self.config = config or DSRConfig()
        self.config.validate()

        self._mean = 0.0
        self._second = 0.0
        self._prev_mean = 0.0
        self._steps = 0
        self._prev_value = None
        self._prev_holdings = 0.0

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        self._mean = 0.0
        self._second = 0.0
        self._prev_mean = 0.0
        self._steps = 0
        self._prev_value = None
        self._prev_holdings = 0.0
        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        total_value = self._extract_total_value(info)
        if total_value is None:
            raise RuntimeError("DSRRewardWrapper requires 'total_value' in info or env.total_value")

        if self._prev_value is None:
            self._prev_value = float(total_value)

        prev_value = self._prev_value if self._prev_value != 0 else 1e-8
        net_return = float(total_value - self._prev_value) / prev_value

        self._steps += 1

        if self._steps <= self.config.warmup_steps:
            reward = net_return
        else:
            self._prev_mean = self._mean
            self._mean += (1.0 - self.config.decay) * (net_return - self._mean)
            self._second += (1.0 - self.config.decay) * (net_return ** 2 - self._second)

            variance = max(self._second - self._mean ** 2, self.config.epsilon)
            numerator = self._mean - self._prev_mean
            denominator = math.sqrt(variance)
            reward = numerator / denominator if denominator > 0 else 0.0

        if self.config.clip_value is not None:
            reward = float(np.clip(reward, -self.config.clip_value, self.config.clip_value))

        entry_bonus = self._compute_entry_bonus(info)
        if entry_bonus:
            reward += entry_bonus
            info = dict(info)
            info['entry_bonus'] = entry_bonus

        self._prev_value = float(total_value)
        self._prev_holdings = float(info.get('holdings', self._prev_holdings))

        info = dict(info)
        info['dsr_reward'] = reward
        info['net_return'] = net_return
        info['base_reward'] = base_reward

        return obs, reward, terminated, truncated, info

    def _extract_total_value(self, info: Dict) -> Optional[float]:
        if isinstance(info, dict) and 'total_value' in info:
            return float(info['total_value'])
        return getattr(self.env, 'total_value', None)

    def _compute_entry_bonus(self, info: Dict[str, Any]) -> float:
        bonus = float(self.config.entry_bonus)
        if bonus <= 0.0:
            return 0.0

        current_holdings = float(info.get('holdings', 0.0))
        if current_holdings <= self.config.entry_min_position:
            return 0.0

        if self._prev_holdings > self.config.entry_min_position:
            return 0.0

        trade_shares = float(info.get('trade_shares', 0.0))
        if trade_shares <= 0.0:
            return 0.0

        return bonus
