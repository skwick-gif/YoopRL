"""
Live data helper utilities for real-time market data snapshots.
Used for live trading dashboards and agent monitoring.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import logging

try:
    import yfinance as yf
except Exception:
    yf = None

logger = logging.getLogger(__name__)


def bar_interval_seconds(interval: str) -> int:
    """
    Convert bar interval string to seconds.
    
    Args:
        interval: Interval string ("1m", "5m", "15m")
        
    Returns:
        Number of seconds for the interval
    """
    mapping = {"1m": 60, "5m": 300, "15m": 900}
    return mapping.get(interval, 60)


def next_fetch_time(base: Optional[datetime], interval_seconds: int) -> datetime:
    """
    Calculate the next fetch time aligned to interval boundaries.
    
    Args:
        base: Base datetime (default: now)
        interval_seconds: Interval in seconds
        
    Returns:
        Next aligned datetime
    """
    reference = base or datetime.utcnow()
    reference = reference.replace(microsecond=0)
    
    epoch = datetime(1970, 1, 1)
    elapsed = (reference - epoch).total_seconds()
    remainder = elapsed % interval_seconds
    
    if remainder == 0:
        return reference
    
    delta = interval_seconds - remainder
    return (reference + timedelta(seconds=delta)).replace(microsecond=0)


def snapshot_path(symbol: str, data_dir: Optional[Path] = None) -> Path:
    """
    Get path for storing market snapshots.
    
    Args:
        symbol: Stock ticker symbol
        data_dir: Base directory for snapshots (default: data/snapshots)
        
    Returns:
        Path to snapshot JSONL file
    """
    if data_dir is None:
        data_dir = Path("data/snapshots")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    safe = symbol.upper().strip()
    return data_dir / f"{safe}.jsonl"


def snapshot_fetch(symbol: str, bar_interval: str = "1m") -> Optional[Dict[str, Any]]:
    """
    Fetch latest market snapshot from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol
        bar_interval: Bar interval ("1m", "5m", "15m")
        
    Returns:
        Snapshot dictionary with OHLCV data or None if failed
    """
    if yf is None:
        logger.warning("yfinance not installed")
        return None
    
    try:
        sym = (symbol or "").strip().upper()
        if not sym:
            return None
        
        interval = bar_interval or "1m"
        
        # Determine period based on interval
        if interval == "1m":
            period = "1d"
        elif interval in ("5m", "15m"):
            period = "5d"
        else:
            period = "5d"
        
        # Download recent bars
        df = yf.Ticker(sym).history(period=period, interval=interval)
        
        if df is None or df.empty:
            return None
        
        # Get last complete bar (2nd to last)
        last = df.tail(2)
        if len(last) >= 2:
            row = last.iloc[-2]
        else:
            row = last.iloc[-1]
        
        # Extract timestamp
        ts_raw = row.name
        ts = ts_raw.to_pydatetime() if hasattr(ts_raw, "to_pydatetime") else pd.Timestamp(ts_raw).to_pydatetime()
        
        # Extract close price
        close_val = row.get("Close")
        if pd.isna(close_val):
            close_val = row.get("Adj Close", close_val)
        if pd.isna(close_val):
            return None
        
        close = float(close_val)
        
        # Build snapshot
        out = {
            "timestamp": ts.isoformat(),
            "symbol": sym,
            "close": close,
            "interval": interval,
        }
        
        # Add OHLV if available
        for col in ("Open", "High", "Low", "Volume"):
            val = row.get(col)
            if pd.isna(val):
                continue
            out[col.lower()] = float(val)
        
        return out
    
    except Exception as e:
        logger.warning(f"Failed to fetch snapshot for {symbol}: {e}")
        return None


def save_snapshot(snapshot: Dict[str, Any], data_dir: Optional[Path] = None) -> None:
    """
    Save snapshot to JSONL file.
    
    Args:
        snapshot: Snapshot dictionary
        data_dir: Base directory for snapshots
    """
    try:
        symbol = snapshot.get("symbol", "UNKNOWN")
        path = snapshot_path(symbol, data_dir)
        
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(snapshot) + "\n")
        
        logger.debug(f"Saved snapshot for {symbol} at {snapshot.get('timestamp')}")
    
    except Exception as e:
        logger.warning(f"Failed to save snapshot: {e}")


def load_snapshots(symbol: str, limit: int = 400, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load recent snapshots from JSONL file.
    
    Args:
        symbol: Stock ticker symbol
        limit: Maximum number of snapshots to return
        data_dir: Base directory for snapshots
        
    Returns:
        List of snapshot dictionaries
    """
    path = snapshot_path(symbol, data_dir)
    
    if not path.exists():
        return []
    
    snapshots = []
    
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    snap = json.loads(line)
                    snapshots.append(snap)
                except Exception:
                    continue
        
        return snapshots[-limit:]
    
    except Exception as e:
        logger.warning(f"Failed to load snapshots for {symbol}: {e}")
        return []


__all__ = [
    "bar_interval_seconds",
    "next_fetch_time",
    "snapshot_path",
    "snapshot_fetch",
    "save_snapshot",
    "load_snapshots",
]
