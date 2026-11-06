"""Signal generation.

Pipeline for each closed bar:
  1. rule setup  (EMA pullback, or liquidity sweep in the trend direction)
  2. ML filter   (optional) - the model must agree with the direction
  3. Monte Carlo (optional) - simulate N future paths for BUY and SELL; the
                              setup's side must have the better, positive EV
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .model import SignalModel
from .montecarlo import MCResult, bar_components, simulate


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
    mc: MCResult | None = None


def _ema_pullback(row, prev) -> tuple[Side, str]:
    """Pullback to the fast EMA in the direction of the long-term trend."""
    uptrend = row.close > row.ema_trend and row.ema_fast > row.ema_slow
    downtrend = row.close < row.ema_trend and row.ema_fast < row.ema_slow
    crossed_up = prev.close <= prev.ema_fast and row.close > row.ema_fast
    crossed_down = prev.close >= prev.ema_fast and row.close < row.ema_fast
    if uptrend and crossed_up and 45 < row.rsi < 70:
        return Side.BUY, f"uptrend pullback, RSI {row.rsi:.0f}"
    if downtrend and crossed_down and 30 < row.rsi < 55:
        return Side.SELL, f"downtrend pullback, RSI {row.rsi:.0f}"
    return Side.NONE, "no setup"


def _sweep_trend(row, prev) -> tuple[Side, str]:
    """Liquidity sweep in the direction of the trend.

    BUY:  trend up (price > EMA200 and EMA50 > EMA200) and the bar swept the
          last swing low (stop hunt) then closed back above it.
    SELL: trend down and the bar swept the last swing high then closed back below.
    """
    uptrend = row.close > row.ema_trend and row.ema_slow > row.ema_trend
    downtrend = row.close < row.ema_trend and row.ema_slow < row.ema_trend
    if uptrend and row.sweep_bull:
        return Side.BUY, f"uptrend + sell-side liquidity sweep below {row.swing_low:.2f}"
    if downtrend and row.sweep_bear:
        return Side.SELL, f"downtrend + buy-side liquidity sweep above {row.swing_high:.2f}"
    return Side.NONE, "no setup"


RULES = {"ema_pullback": _ema_pullback, "sweep_trend": _sweep_trend}


def rule_signal(row, prev, mode: str = "ema_pullback") -> tuple[Side, str]:
    return RULES[mode](row, prev)


def run_montecarlo(feats: pd.DataFrame, i: int, cfg, spread: float,
                   comps: np.ndarray | None = None, n_sims: int | None = None) -> MCResult:
    """Monte Carlo from the close of bar `i`, using only bars <= i."""
    mc, r = cfg.montecarlo, cfg.risk
    if comps is None:
        comps = bar_components(feats)
    hist = comps[max(1, i - mc.lookback_bars + 1): i + 1]
    atr = float(feats["atr"].iloc[i])
    return simulate(hist, float(feats["close"].iloc[i]), spread,
                    sl_dist=atr * r.sl_atr_mult, tp_dist=atr * r.tp_atr_mult,
                    n_sims=n_sims or mc.n_sims, horizon=mc.horizon_bars,
                    block_len=mc.block_len, chunk=mc.chunk, seed=mc.seed)


def generate_signal(feats: pd.DataFrame, cfg, model: SignalModel | None = None,
                    spread: float = 0.0, i: int | None = None,
                    proba: float | None = None, comps: np.ndarray | None = None,
                    mc_sims: int | None = None) -> Signal:
    """Signal for closed bar `i` (default: last row of `feats`).

    `proba` (precomputed P(up)) and `comps` let the backtester avoid recomputation.
    """
    s = cfg.strategy
    i = len(feats) - 1 if i is None else i
    row, prev = feats.iloc[i], feats.iloc[i - 1]
    side, reason = rule_signal(row, prev, s.mode)
    sig = Signal(side, float(row.close), float(row.atr), reason)
    if side is Side.NONE:
        return sig

    # --- ML filter
    if s.use_ml_filter and (proba is not None or (model is not None and model.ready)):
        p_up = proba if proba is not None else model.prob_up(feats.iloc[[i]])
        sig.ml_prob_up = p_up
        conf = p_up if side is Side.BUY else 1 - p_up
        if np.isnan(conf) or conf < s.ml_min_proba:
            sig.reason += f" | rejected by ML (conf {conf:.2f})"
            sig.side = Side.NONE
            return sig
        sig.reason += f" | ML conf {conf:.2f}"

    # --- Monte Carlo filter
    if cfg.montecarlo.enabled:
        res = run_montecarlo(feats, i, cfg, spread, comps, mc_sims)
        sig.mc = res
        best = res.best(cfg.montecarlo.min_edge_r)
        if best != side.value:
            sig.reason += f" | rejected by {res} -> {best}"
            sig.side = Side.NONE
        else:
            sig.reason += f" | {res}"
    return sig
