"""Bar-by-bar backtester that mirrors the live logic.

Conventions (deliberately conservative):
- signals are computed on a closed bar, entry happens at the NEXT bar's open
- the spread is paid on every trade
- if both SL and TP fall inside the same bar, the SL is assumed to hit first
- results are measured in R (multiples of the amount risked) and compounded
  at `risk_per_trade_pct` per trade, so they are independent of contract size
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import FEATURE_COLUMNS, add_indicators
from .model import SignalModel
from .risk import stop_levels
from .strategy import Side, rule_signal


@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    exit_time: pd.Timestamp | None = None
    exit: float | None = None
    r: float | None = None


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    start_equity: float
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        return "\n".join(f"  {k:<18} {v}" for k, v in self.stats.items())


def _stats(trades: list[Trade], equity: pd.Series, start: float) -> dict:
    rs = np.array([t.r for t in trades if t.r is not None])
    if len(rs) == 0:
        return {"trades": 0}
    wins, losses = rs[rs > 0], rs[rs <= 0]
    peak = equity.cummax()
    max_dd = ((peak - equity) / peak).max() * 100
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
    return {
        "trades": len(rs),
        "win_rate": f"{len(wins) / len(rs) * 100:.1f}%",
        "avg_R": f"{rs.mean():.3f}",
        "total_R": f"{rs.sum():.2f}",
        "profit_factor": f"{pf:.2f}",
        "return": f"{(equity.iloc[-1] / start - 1) * 100:.2f}%",
        "max_drawdown": f"{max_dd:.2f}%",
        "final_equity": f"{equity.iloc[-1]:.2f}",
    }


def run_backtest(bars: pd.DataFrame, cfg, model: SignalModel | None = None,
                 spread: float = 0.0, start_equity: float = 10_000.0) -> BacktestResult:
    s, r = cfg.strategy, cfg.risk
    feats = add_indicators(bars, s).reset_index(drop=True)

    use_ml = s.use_ml_filter and model is not None and model.ready
    if use_ml:
        valid = feats[FEATURE_COLUMNS].notna().all(axis=1)
        proba = np.full(len(feats), np.nan)
        proba[valid.to_numpy()] = model.clf.predict_proba(feats.loc[valid, FEATURE_COLUMNS])[:, 1]

    o, h, l = feats["open"].to_numpy(), feats["high"].to_numpy(), feats["low"].to_numpy()
    times = pd.to_datetime(feats["time"])
    hours = times.dt.hour.to_numpy()
    start_h, end_h = cfg.trading_hours_utc

    equity = start_equity
    curve = np.full(len(feats), np.nan)
    trades: list[Trade] = []
    open_t: Trade | None = None
    warmup = s.trend_ema

    rows = list(feats.itertuples(index=False))
    for i in range(warmup, len(feats) - 1):
        # 1) manage open trade on bar i
        if open_t is not None:
            hit = None
            # Buys exit at the bid, sells exit at the ask (bid + spread).
            # A gap through the stop fills at the open, not at the stop price.
            if open_t.side == "BUY":
                if l[i] <= open_t.sl:
                    hit = min(o[i], open_t.sl)
                elif h[i] >= open_t.tp:
                    hit = max(o[i], open_t.tp)
            else:
                if h[i] + spread >= open_t.sl:
                    hit = max(o[i] + spread, open_t.sl)
                elif l[i] + spread <= open_t.tp:
                    hit = min(o[i] + spread, open_t.tp)
            if hit is not None:
                risk_dist = abs(open_t.entry - open_t.sl)
                move = (hit - open_t.entry) if open_t.side == "BUY" else (open_t.entry - hit)
                open_t.exit, open_t.exit_time = hit, times[i]
                open_t.r = move / risk_dist
                equity *= 1 + open_t.r * r.risk_per_trade_pct / 100
                trades.append(open_t)
                open_t = None
        curve[i] = equity

        # 2) look for a new entry on the close of bar i, fill at open of i+1
        if open_t is not None or not (start_h <= hours[i] < end_h):
            continue
        side, _ = rule_signal(rows[i], rows[i - 1])
        if side is Side.NONE:
            continue
        if use_ml:
            p = proba[i]
            conf = p if side is Side.BUY else 1 - p
            if np.isnan(conf) or conf < s.ml_min_proba:
                continue
        entry = o[i + 1] + (spread if side is Side.BUY else 0.0)
        sl, tp = stop_levels(side.value, entry, rows[i].atr, r)
        open_t = Trade(side.value, times[i + 1], entry, sl, tp)

    curve[-1] = equity
    eq = pd.Series(curve, index=times).ffill().dropna()
    return BacktestResult(trades, eq, start_equity, _stats(trades, eq, start_equity))
