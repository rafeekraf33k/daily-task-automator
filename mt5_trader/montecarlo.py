"""Monte Carlo trade evaluation.

Before a trade, we simulate `n_sims` (default 1,000,000) possible future price
paths and check, for BOTH a buy and a sell with the same stop-loss/take-profit
distances and the real spread, which one does better on average.

Paths are built by block-bootstrapping recent bars: random runs of
`block_len` consecutive historical bars (their close-to-close return and
their high/low excursions) are stitched together. This keeps realistic
volatility, wicks and short-term momentum of the current market.

What it can and can't tell you: it measures whether the SL/TP placement and
the recent behaviour of the market (drift, volatility, wick size) favour a
side after costs. It does not know about news or future regime changes, so it
is a filter, not a crystal ball.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MCResult:
    n_sims: int
    buy_ev: float        # expected result per trade, in R (1R = amount risked)
    sell_ev: float
    buy_tp_rate: float   # share of paths where TP is hit before SL
    sell_tp_rate: float
    buy_sl_rate: float
    sell_sl_rate: float

    def best(self, min_edge_r: float) -> str:
        if max(self.buy_ev, self.sell_ev) < min_edge_r:
            return "NONE"
        return "BUY" if self.buy_ev >= self.sell_ev else "SELL"

    def __str__(self) -> str:
        return (f"MC {self.n_sims:,} paths: BUY EV {self.buy_ev:+.3f}R "
                f"(TP {self.buy_tp_rate:.1%}/SL {self.buy_sl_rate:.1%}), "
                f"SELL EV {self.sell_ev:+.3f}R (TP {self.sell_tp_rate:.1%}/SL {self.sell_sl_rate:.1%})")


def bar_components(df: pd.DataFrame) -> np.ndarray:
    """(n, 3) array of log(close/prev_close), log(high/prev_close), log(low/prev_close).
    Row 0 is NaN (no previous close)."""
    prev = df["close"].shift(1)
    comps = np.column_stack([
        np.log(df["close"] / prev), np.log(df["high"] / prev), np.log(df["low"] / prev),
    ])
    return comps


def _eval_long(C, H, L, entry, sl_dist, tp_dist):
    """Long trade on BID paths. Returns (sum_R, n_tp, n_sl). SL wins ties."""
    n, hz = C.shape
    sl_hit = L <= entry - sl_dist
    tp_hit = H >= entry + tp_dist
    si = np.where(sl_hit.any(1), sl_hit.argmax(1), hz)
    ti = np.where(tp_hit.any(1), tp_hit.argmax(1), hz)
    sl_first = (si < hz) & (si <= ti)
    tp_first = (ti < hz) & (ti < si)
    r = np.where(sl_first, -1.0,
                 np.where(tp_first, tp_dist / sl_dist, (C[:, -1] - entry) / sl_dist))
    return float(r.sum()), int(tp_first.sum()), int(sl_first.sum())


def simulate(comps: np.ndarray, price: float, spread: float, sl_dist: float, tp_dist: float,
             n_sims: int = 1_000_000, horizon: int = 48, block_len: int = 4,
             chunk: int = 50_000, seed: int | None = None) -> MCResult:
    """Evaluate a buy and a sell from `price` (current BID).

    Buy enters at ask (= bid + spread) and exits at bid; sell enters at bid and
    exits at ask. A sell is evaluated as a buy on the mirrored price series
    (bid' = -ask), so both sides use exactly the same logic and the same paths.
    """
    comps = comps[~np.isnan(comps).any(axis=1)]
    if len(comps) < block_len * 10:
        raise ValueError("Not enough history for Monte Carlo")
    if sl_dist <= 0:
        raise ValueError("sl_dist must be > 0")
    rng = np.random.default_rng(seed)
    nb = -(-horizon // block_len)
    offsets = np.arange(block_len)
    tot = {"b": [0.0, 0, 0], "s": [0.0, 0, 0]}

    done = 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        starts = rng.integers(0, len(comps) - block_len + 1, size=(m, nb))
        idx = (starts[:, :, None] + offsets).reshape(m, -1)[:, :horizon]
        rc, rh, rl = comps[idx, 0], comps[idx, 1], comps[idx, 2]
        log_c = np.cumsum(rc, axis=1)
        log_prev = log_c - rc
        C = price * np.exp(log_c)
        H = price * np.exp(log_prev + rh)
        L = price * np.exp(log_prev + rl)

        for key, res in (
            ("b", _eval_long(C, H, L, price + spread, sl_dist, tp_dist)),
            # mirrored: bid' = -(bid + spread); high' = -(low + spread); low' = -(high + spread)
            ("s", _eval_long(-(C + spread), -(L + spread), -(H + spread), -price, sl_dist, tp_dist)),
        ):
            for j in range(3):
                tot[key][j] += res[j]
        done += m

    b, s = tot["b"], tot["s"]
    return MCResult(n_sims, b[0] / n_sims, s[0] / n_sims, b[1] / n_sims, s[1] / n_sims,
                    b[2] / n_sims, s[2] / n_sims)
