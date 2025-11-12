import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
for candidate in (BACKEND_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from environments.intraday_env import IntradayEquityEnv, IntradaySessionSampler


def _build_intraday_frame(prices, minutes):
    session_date = pd.Timestamp("2024-01-02")
    data = {
        "session_date": [session_date] * len(prices),
        "close": list(map(float, prices)),
        "time_fraction": [float(m) / 390.0 for m in minutes],
        "minutes_from_open": list(map(float, minutes)),
        "bar_index": list(range(len(prices))),
        "feature_a": np.linspace(0.0, 1.0, len(prices)),
    }
    return pd.DataFrame(data)


def _make_env(prices, minutes, *, commission=0.0, slippage_bps=10.0, max_position_size=0.101):
    df = _build_intraday_frame(prices, minutes)
    sampler = IntradaySessionSampler(shuffle=False, sequential=True)
    slippage_config = {"buy_bps": slippage_bps, "sell_bps": slippage_bps}
    env = IntradayEquityEnv(
        df=df,
        initial_capital=1_000.0,
        commission=commission,
    max_position_size=max_position_size,
        normalize_obs=False,
        history_config=None,
        sampler=sampler,
        slippage_config=slippage_config,
    )
    env.reset()
    return env


def test_slippage_applied_on_buy():
    env = _make_env(prices=[100.0, 100.0], minutes=[360.0, 375.0])

    _, reward, _, _, info = env.step(1)

    expected_slippage = 0.1  # 10 bps on $100 notional
    assert env.holdings == 1
    assert info["slippage"] == pytest.approx(expected_slippage, rel=1e-6)
    assert info["transaction_cost"] == pytest.approx(expected_slippage, rel=1e-6)
    assert info["trade_shares"] == 1
    assert reward == pytest.approx(-expected_slippage / info["prev_total_value"], rel=1e-6)


def test_forced_exit_at_1545_minutes_from_open():
    env = _make_env(prices=[100.0, 101.0, 101.0], minutes=[360.0, 375.0, 390.0])

    env.step(1)
    assert env.current_step == 1

    _, reward, terminated, _, info = env.step(0)

    assert info["forced_exit"] is True
    assert info["original_action"] == 0
    assert info["action"] == 2
    assert info["forced_exit_reason"] in {"minutes_from_open", "time_fraction"}
    assert info["minutes_from_open"] == pytest.approx(375.0, rel=1e-6)
    assert env.holdings == 0
    assert terminated is True


def test_reward_nets_transaction_costs():
    env = _make_env(prices=[100.0, 100.0], minutes=[360.0, 375.0])

    _, reward1, _, _, info1 = env.step(1)
    cost_ratio1 = info1["transaction_cost"] / info1["prev_total_value"]
    gross_return1 = (
        (info1["total_value"] + info1["transaction_cost"] - info1["prev_total_value"]) /
        info1["prev_total_value"]
    )
    assert reward1 == pytest.approx(gross_return1 - cost_ratio1, rel=1e-6)
    assert info1["transaction_cost_ratio"] == pytest.approx(cost_ratio1, rel=1e-6)

    _, reward2, terminated2, _, info2 = env.step(0)
    cost_ratio2 = info2["transaction_cost"] / info2["prev_total_value"]
    gross_return2 = (
        (info2["total_value"] + info2["transaction_cost"] - info2["prev_total_value"]) /
        info2["prev_total_value"]
    )
    assert reward2 == pytest.approx(gross_return2 - cost_ratio2, rel=1e-6)
    assert info2["transaction_cost_ratio"] == pytest.approx(cost_ratio2, rel=1e-6)
    assert terminated2 is True
