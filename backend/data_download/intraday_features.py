"""Feature engineering for intraday SAC + DSR workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class IntradayFeatureSpec:
    primary_symbol: str
    benchmark_symbol: str
    ema_period: int = 50
    short_ema_period: int = 10
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


def add_intraday_features(df: pd.DataFrame, spec: IntradayFeatureSpec) -> pd.DataFrame:
    data = df.copy()
    primary_prefix = spec.primary_symbol.lower()
    benchmark_prefix = spec.benchmark_symbol.lower()

    primary_close = data[f"{primary_prefix}_close"].astype(float)
    primary_high = data[f"{primary_prefix}_high"].astype(float)
    primary_low = data[f"{primary_prefix}_low"].astype(float)
    benchmark_close = data[f"{benchmark_prefix}_close"].astype(float)
    benchmark_high = data[f"{benchmark_prefix}_high"].astype(float)
    benchmark_low = data[f"{benchmark_prefix}_low"].astype(float)

    data['primary_return'] = primary_close.pct_change().fillna(0.0)
    data['benchmark_return'] = benchmark_close.pct_change().fillna(0.0)

    benchmark_ema = _ema(benchmark_close, spec.ema_period)
    benchmark_atr = _atr(benchmark_high, benchmark_low, benchmark_close, spec.atr_period)
    data['base_trend_context'] = (benchmark_close - benchmark_ema) / (benchmark_atr + 1e-8)

    macd_line, signal_line, histogram = _macd(benchmark_close, spec.macd_fast, spec.macd_slow, spec.macd_signal)
    data['base_momentum'] = histogram

    plus_di, minus_di, adx = _adx(benchmark_high, benchmark_low, benchmark_close, spec.atr_period)
    data['base_trend_strength'] = adx
    data['base_plus_di'] = plus_di
    data['base_minus_di'] = minus_di

    data['base_extremes'] = _rsi(benchmark_close, spec.atr_period)

    primary_atr = _atr(primary_high, primary_low, primary_close, spec.atr_period)
    data['leveraged_volatility'] = primary_atr / (primary_close + 1e-8)

    primary_ema_short = _ema(primary_close, spec.short_ema_period)
    data['leveraged_momentum_short'] = (primary_close - primary_ema_short) / (primary_atr + 1e-8)

    data['time_context'] = data.get('time_fraction', 0.0)
    data['position_context'] = 0.0

    metadata_cols = ['bar_index', 'time_fraction', 'minutes_from_open', 'is_session_end']
    for col in metadata_cols:
        if col not in data.columns:
            data[col] = 0.0

    _stabilize_high_variance_features(data, primary_prefix, benchmark_prefix)

    return data.fillna(0.0)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return _ema(true_range, period)


def _macd(series: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high_diff = high.diff()
    low_diff = low.diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    atr = _atr(high, low, close, period)
    plus_di = 100 * _ema(pd.Series(plus_dm, index=high.index), period) / (atr + 1e-8)
    minus_di = 100 * _ema(pd.Series(minus_dm, index=high.index), period) / (atr + 1e-8)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)) * 100
    adx = _ema(dx, period)
    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx.fillna(0.0)


def _stabilize_high_variance_features(
    data: pd.DataFrame,
    primary_prefix: str,
    benchmark_prefix: str,
) -> None:
    """Robustly scale noisy magnitude features before env normalization."""

    targets = [
        (f"{primary_prefix}_volume", True),
        (f"{benchmark_prefix}_volume", True),
        ("leveraged_volatility", False),
    ]

    for column, apply_log in targets:
        if column not in data.columns:
            continue
        data[column] = _robust_rescale_feature(data[column], log_transform=apply_log)


def _robust_rescale_feature(
    series: pd.Series,
    *,
    log_transform: bool,
    clip_sigma: float = 4.0,
) -> pd.Series:
    """Center, scale, then clip a feature to dampen extreme swings."""

    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    arr = values.to_numpy(copy=True)
    if log_transform:
        arr = np.log1p(np.clip(arr, 0.0, None))

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = 1.0

    normalized = (arr - median) / scale
    clipped = np.clip(normalized, -clip_sigma, clip_sigma)
    rescaled = clipped / clip_sigma
    return pd.Series(rescaled, index=series.index)
