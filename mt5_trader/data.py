"""Loading/saving bar data and generating synthetic data for offline testing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["time", "open", "high", "low", "close"]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load bars from a CSV with at least time, open, high, low, close columns.

    Also accepts MT5 'Export bars' files (tab-separated <DATE> <TIME> <OPEN> ... headers).
    """
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    if "time" not in df.columns and "date" in df.columns:
        df["time"] = df["date"]
    elif "date" in df.columns and "time" in df.columns:
        df["time"] = df["date"].astype(str) + " " + df["time"].astype(str)
    if "tickvol" in df.columns and "tick_volume" not in df.columns:
        df["tick_volume"] = df["tickvol"]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0
    return df.sort_values("time").reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def synthetic_bars(n: int = 5000, start_price: float = 1.10, seed: int = 0,
                   freq: str = "15min") -> pd.DataFrame:
    """Random-walk-with-regimes bars. Only for smoke-testing the pipeline;
    a strategy that 'works' here proves nothing about real markets."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.00004, n // 250 + 1), 250)[:n]
    rets = drift + rng.normal(0, 0.0008, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, 0.0004, n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    times = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "time": times, "open": open_, "high": high, "low": low, "close": close,
        "tick_volume": rng.integers(100, 1000, n),
    })
