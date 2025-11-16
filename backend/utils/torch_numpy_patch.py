"""Compatibility helpers for PyTorch <-> NumPy conversions.

Stable-Baselines3 relies on ``torch.as_tensor``/``torch.tensor`` to convert
NumPy observations into tensors. Recent NumPy 2.x releases changed their C API
in a way that breaks precompiled PyTorch wheels (and other dependencies) that
were built against NumPy 1.x. When that happens ``torch.as_tensor`` raises
``RuntimeError: Could not infer dtype of numpy.float32`` as soon as SB3 tries to
process the first observation, which aborts training immediately.

To make the intraday pipeline usable without downgrading NumPy manually we
install a lightweight runtime patch that detects the failure and falls back to a
Python list conversion. This keeps the training loop functional (at the cost of
an extra copy) and surfaces a clear warning so users can later pin ``numpy<2``
for better performance when desired.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch

_NUMPY_DTYPE_ERROR = "Could not infer dtype of numpy.float32"
_PATCHED = False


def ensure_torch_numpy_compat() -> None:
    """Enable a safe conversion path when PyTorch cannot parse NumPy arrays."""

    global _PATCHED
    if _PATCHED:
        return

    if _torch_handles_numpy_arrays():
        return

    _PATCHED = True
    _install_tensor_patches()
    print(
        "[WARN] NumPy 2.x compatibility workaround enabled for PyTorch. "
        "Consider installing numpy<2 for zero-copy conversions."
    )


def _torch_handles_numpy_arrays() -> bool:
    """Return True when the current torch build can convert NumPy arrays."""

    try:
        torch.as_tensor(np.zeros(1, dtype=np.float32))
        return True
    except RuntimeError as exc:
        message = str(exc)
        if _NUMPY_DTYPE_ERROR in message:
            return False
        raise


def _install_tensor_patches() -> None:
    """Patch tensor constructors to gracefully handle NumPy 2.x objects."""

    original_as_tensor = torch.as_tensor
    original_tensor = torch.tensor
    original_from_numpy = torch.from_numpy

    def _convert_numpy_payload(data: Any) -> Tuple[Any, Any]:
        if isinstance(data, np.generic):
            return data.item(), _map_torch_dtype(data.dtype)
        if isinstance(data, np.ndarray):
            dtype = _map_torch_dtype(data.dtype)
            return data.tolist(), dtype
        return data, None

    def _safe_as_tensor(data: Any, *args: Any, **kwargs: Any):
        try:
            return original_as_tensor(data, *args, **kwargs)
        except RuntimeError as exc:
            if _NUMPY_DTYPE_ERROR not in str(exc):
                raise
            converted, dtype = _convert_numpy_payload(data)
            if dtype is not None and "dtype" not in kwargs:
                kwargs["dtype"] = dtype
            return original_as_tensor(converted, *args, **kwargs)

    def _safe_tensor(data: Any, *args: Any, **kwargs: Any):
        try:
            return original_tensor(data, *args, **kwargs)
        except RuntimeError as exc:
            if _NUMPY_DTYPE_ERROR not in str(exc):
                raise
            converted, dtype = _convert_numpy_payload(data)
            if dtype is not None and "dtype" not in kwargs:
                kwargs["dtype"] = dtype
            return original_tensor(converted, *args, **kwargs)

    def _safe_from_numpy(array: np.ndarray, *args: Any, **kwargs: Any):
        converted, dtype = _convert_numpy_payload(array)
        if converted is array:
            return original_from_numpy(array, *args, **kwargs)
        if dtype is not None and "dtype" not in kwargs:
            kwargs["dtype"] = dtype
        return original_tensor(converted, *args, **kwargs)

    torch.as_tensor = _safe_as_tensor  # type: ignore[assignment]
    torch.tensor = _safe_tensor  # type: ignore[assignment]
    torch.from_numpy = _safe_from_numpy  # type: ignore[assignment]


def _map_torch_dtype(np_dtype: np.dtype) -> torch.dtype:
    if np.issubdtype(np_dtype, np.floating):
        if np_dtype == np.float16:
            return torch.float16
        if np_dtype == np.float64:
            return torch.float64
        return torch.float32
    if np.issubdtype(np_dtype, np.integer):
        if np_dtype == np.uint8:
            return torch.uint8
        return torch.int64
    if np.issubdtype(np_dtype, np.bool_):
        return torch.bool
    return torch.float32
