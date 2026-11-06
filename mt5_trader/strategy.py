"""Signal generation: trend-following rules, optionally confirmed by the ML model."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from .model import SignalModel


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"


@dataclass
class Signal:
    side: Side
    price: float
    atr: float
    reason: str
    ml_prob_up: float | None = None


def rule_signal(row: pd.Series, prev: pd.Series) -> tuple[Side, str]:
    """EMA pullback in the direction of the long-term trend.

    BUY  when: price above trend EMA, fast EMA above slow EMA, and the fast
               EMA is crossed back upward after a pullback, RSI not overbought.
    SELL is the mirror image.
    """
    uptrend = row.close > row.ema_trend and row.ema_fast > row.ema_slow
    downtrend = row.close < row.ema_trend and row.ema_fast < row.ema_slow

    crossed_up = prev.close <= prev.ema_fast and row.close > row.ema_fast
    crossed_down = prev.close >= prev.ema_fast and row.close < row.ema_fast

    if uptrend and crossed_up and 45 < row.rsi < 70:
        return Side.BUY, f"uptrend pullback, RSI {row.rsi:.0f}"
    if downtrend and crossed_down and 30 < row.rsi < 55:
        return Side.SELL, f"downtrend pullback, RSI {row.rsi:.0f}"
    return Side.NONE, "no setup"


def generate_signal(feats: pd.DataFrame, strategy_cfg, model: SignalModel | None) -> Signal:
    """Signal for the last *closed* bar of `feats` (output of add_indicators)."""
    row, prev = feats.iloc[-1], feats.iloc[-2]
    side, reason = rule_signal(row, prev)
    sig = Signal(side, float(row.close), float(row.atr), reason)

    if side is Side.NONE or not strategy_cfg.use_ml_filter or model is None or not model.ready:
        return sig

    p_up = model.prob_up(feats.iloc[[-1]])
    sig.ml_prob_up = p_up
    conf = p_up if side is Side.BUY else 1 - p_up
    if conf < strategy_cfg.ml_min_proba:
        sig.reason += f" | rejected by ML (conf {conf:.2f})"
        sig.side = Side.NONE
    else:
        sig.reason += f" | ML conf {conf:.2f}"
    return sig
