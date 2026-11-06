"""Technical indicators and ML features computed from OHLC bars.

Input DataFrame columns: time, open, high, low, close, tick_volume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "dist_fast", "dist_slow", "dist_trend", "ema_slope",
    "rsi", "atr_pct", "macd_hist", "range_pct", "body_pct",
    "vol_z", "hour_sin", "hour_cos",
]


def add_indicators(df: pd.DataFrame, s) -> pd.DataFrame:
    """Add indicator + feature columns. `s` is a StrategyConfig."""
    out = df.copy()
    c = out["close"]
    out["ema_fast"] = ema(c, s.fast_ema)
    out["ema_slow"] = ema(c, s.slow_ema)
    out["ema_trend"] = ema(c, s.trend_ema)
    out["rsi"] = rsi(c, s.rsi_period)
    out["atr"] = atr(out, s.atr_period)
    _, _, out["macd_hist"] = macd(c)

    # Scale-free features for the ML model
    for n in (1, 3, 6, 12):
        out[f"ret_{n}"] = c.pct_change(n)
    out["dist_fast"] = (c - out["ema_fast"]) / out["atr"]
    out["dist_slow"] = (c - out["ema_slow"]) / out["atr"]
    out["dist_trend"] = (c - out["ema_trend"]) / out["atr"]
    out["ema_slope"] = out["ema_slow"].diff(5) / out["atr"]
    out["atr_pct"] = out["atr"] / c
    out["macd_hist"] = out["macd_hist"] / out["atr"]
    out["range_pct"] = (out["high"] - out["low"]) / c
    out["body_pct"] = (c - out["open"]) / c
    vol = out["tick_volume"].astype(float) if "tick_volume" in out else pd.Series(0.0, index=out.index)
    out["vol_z"] = ((vol - vol.rolling(50).mean()) / vol.rolling(50).std()).fillna(0.0)
    hours = pd.to_datetime(out["time"]).dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    return out
