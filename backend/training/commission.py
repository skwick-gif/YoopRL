"""Utilities for normalizing trading commission configurations."""

from __future__ import annotations

from typing import Any, Dict, Union

IBKR_DEFAULT_COMMISSION: Dict[str, float] = {
    "type": "ibkr_tiered_us_equities",
    "per_share": 0.01,
    "min_fee": 2.5,
    "max_pct": 0.01,
}

DEFAULT_SLIPPAGE_CONFIG: Dict[str, float] = {
    "buy_bps": 0.0,
    "sell_bps": 0.0,
    "buy_per_share": 0.0,
    "sell_per_share": 0.0,
}


def _extract(settings: Union[Dict[str, Any], Any], key: str, default: Any) -> Any:
    if isinstance(settings, dict):
        return settings.get(key, default)
    return getattr(settings, key, default)


def resolve_commission_config(settings: Union[Dict[str, Any], Any, None]) -> Dict[str, float]:
    """Return a normalized commission configuration for environment use."""

    merged = dict(IBKR_DEFAULT_COMMISSION)

    if settings is None:
        commission_value = None
    else:
        commission_value = _extract(settings, "commission", None)

    if isinstance(commission_value, dict):
        key_aliases = {
            "per_share": ("per_share", "perShare", "value", "per_trade", "perTrade"),
            "min_fee": ("min_fee", "minFee", "flat_fee", "flatFee"),
            "max_pct": ("max_pct", "maxPct"),
        }

        for target_key, candidates in key_aliases.items():
            for candidate in candidates:
                if candidate in commission_value and commission_value[candidate] is not None:
                    try:
                        merged[target_key] = float(commission_value[candidate])
                    except (TypeError, ValueError):
                        pass
                    break

        raw_type = commission_value.get("type") or commission_value.get("model")
        if raw_type:
            merged["type"] = str(raw_type)

        if merged["per_share"] == IBKR_DEFAULT_COMMISSION["per_share"]:
            fallback = _extract(settings, "commission_per_share", None)
            if fallback is not None:
                try:
                    merged["per_share"] = float(fallback)
                except (TypeError, ValueError):
                    pass

        if merged["min_fee"] == IBKR_DEFAULT_COMMISSION["min_fee"]:
            fallback = _extract(settings, "commission_min_fee", merged["min_fee"])
            try:
                merged["min_fee"] = float(fallback)
            except (TypeError, ValueError):
                merged["min_fee"] = IBKR_DEFAULT_COMMISSION["min_fee"]

        if merged["max_pct"] == IBKR_DEFAULT_COMMISSION["max_pct"]:
            fallback = _extract(settings, "commission_max_pct", merged["max_pct"])
            try:
                merged["max_pct"] = float(fallback)
            except (TypeError, ValueError):
                merged["max_pct"] = IBKR_DEFAULT_COMMISSION["max_pct"]
    else:
        if commission_value is None:
            commission_value = _extract(settings, "commission_per_share", None)

        min_fee = _extract(settings, "commission_min_fee", merged["min_fee"])
        max_pct = _extract(settings, "commission_max_pct", merged["max_pct"])
        model_type = _extract(settings, "commission_model", merged["type"])

        if commission_value is not None:
            merged["per_share"] = float(commission_value)
        merged["min_fee"] = float(min_fee)
        merged["max_pct"] = float(max_pct)
        merged["type"] = str(model_type) if model_type else merged["type"]

    # Safety clamps
    if merged["per_share"] <= 0:
        merged["per_share"] = IBKR_DEFAULT_COMMISSION["per_share"]
    if merged["min_fee"] <= 0:
        merged["min_fee"] = IBKR_DEFAULT_COMMISSION["min_fee"]
    if merged["max_pct"] <= 0:
        merged["max_pct"] = IBKR_DEFAULT_COMMISSION["max_pct"]

    return merged


def _apply_slippage_dict(target: Dict[str, float], payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return

    if payload.get("enabled") is False:
        target.update(DEFAULT_SLIPPAGE_CONFIG)
        return

    shared_bps_keys = (
        "bps",
        "basis_points",
        "basisPoints",
        "slippage_bps",
    )

    for key in shared_bps_keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            numeric = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
        target["buy_bps"] = numeric
        target["sell_bps"] = numeric
        break

    buy_bps_candidates = (
        "buy_bps",
        "buyBasisPoints",
        "long_bps",
        "buy",
    )
    sell_bps_candidates = (
        "sell_bps",
        "sellBasisPoints",
        "short_bps",
        "sell",
    )

    for key in buy_bps_candidates:
        value = payload.get(key)
        if value is None:
            continue
        try:
            target["buy_bps"] = max(0.0, float(value))
        except (TypeError, ValueError):
            pass
        break

    for key in sell_bps_candidates:
        value = payload.get(key)
        if value is None:
            continue
        try:
            target["sell_bps"] = max(0.0, float(value))
        except (TypeError, ValueError):
            pass
        break

    shared_per_share_keys = (
        "per_share",
        "perShare",
        "slippage_per_share",
    )
    for key in shared_per_share_keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            numeric = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
        target["buy_per_share"] = numeric
        target["sell_per_share"] = numeric
        break

    buy_per_share_candidates = (
        "buy_per_share",
        "buyPerShare",
        "long_per_share",
    )
    sell_per_share_candidates = (
        "sell_per_share",
        "sellPerShare",
        "short_per_share",
    )

    for key in buy_per_share_candidates:
        value = payload.get(key)
        if value is None:
            continue
        try:
            target["buy_per_share"] = max(0.0, float(value))
        except (TypeError, ValueError):
            pass
        break

    for key in sell_per_share_candidates:
        value = payload.get(key)
        if value is None:
            continue
        try:
            target["sell_per_share"] = max(0.0, float(value))
        except (TypeError, ValueError):
            pass
        break


def resolve_slippage_config(settings: Union[Dict[str, Any], Any, None]) -> Dict[str, float]:
    """Normalize slippage inputs for consumption by trading environments."""

    merged = dict(DEFAULT_SLIPPAGE_CONFIG)

    slippage_payload = None
    if settings is not None:
        slippage_payload = _extract(settings, "slippage", None)
        if slippage_payload is None:
            slippage_payload = _extract(settings, "slippage_config", None)

    if isinstance(slippage_payload, (int, float)):
        value = max(0.0, float(slippage_payload))
        merged["buy_bps"] = value
        merged["sell_bps"] = value
    elif isinstance(slippage_payload, dict):
        _apply_slippage_dict(merged, slippage_payload)
    else:
        if settings is not None:
            if slippage_payload is not None and hasattr(slippage_payload, "__dict__"):
                _apply_slippage_dict(merged, vars(slippage_payload))

            overrides: Dict[str, Any] = {}
            for key in (
                "slippage_bps",
                "slippage_buy_bps",
                "slippage_sell_bps",
                "slippage_per_share",
                "slippage_buy_per_share",
                "slippage_sell_per_share",
            ):
                value = _extract(settings, key, None)
                if value is not None:
                    overrides[key] = value

            if overrides:
                _apply_slippage_dict(merged, overrides)

    merged["buy_bps"] = max(0.0, float(merged.get("buy_bps", 0.0)))
    merged["sell_bps"] = max(0.0, float(merged.get("sell_bps", 0.0)))
    merged["buy_per_share"] = max(0.0, float(merged.get("buy_per_share", 0.0)))
    merged["sell_per_share"] = max(0.0, float(merged.get("sell_per_share", 0.0)))

    return merged

__all__ = [
    "IBKR_DEFAULT_COMMISSION",
    "DEFAULT_SLIPPAGE_CONFIG",
    "resolve_commission_config",
    "resolve_slippage_config",
]
