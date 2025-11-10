"""Utilities for normalizing trading commission configurations."""

from __future__ import annotations

from typing import Any, Dict, Union

IBKR_DEFAULT_COMMISSION: Dict[str, float] = {
    "type": "ibkr_tiered_us_equities",
    "per_share": 0.01,
    "min_fee": 2.5,
    "max_pct": 0.01,
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


__all__ = ["IBKR_DEFAULT_COMMISSION", "resolve_commission_config"]
