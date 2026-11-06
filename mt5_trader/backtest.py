"""Bar-by-bar backtester that mirrors the live logic.

Conventions (deliberately conservative):
- signals are computed on a closed bar, entry happens at the NEXT bar's open
- buys enter at ask (bid + spread) and exit at bid; sells the opposite
- if a stop and a target fall inside the same bar, the stop is assumed first
- a gap through a stop fills at the bar's open, not at the stop price

Single-trade mode measures results in R and compounds `risk_per_trade_pct`.
Basket mode uses real money: P&L = price move * lots * contract_size, and the
equity curve includes FLOATING losses (essential to see basket risk).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .basket import Basket, next_lot
from .indicators import FEATURE_COLUMNS, add_indicators
from .model import SignalModel
from .montecarlo import bar_components
from .risk import stop_levels
from .strategy import Side, generate_signal


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
    layers: int = 1
    pnl: float | None = None
    reason: str = ""


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    start_equity: float
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        return "\n".join(f"  {k:<20} {v}" for k, v in self.stats.items())


def _dd(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((peak - equity) / peak).max() * 100)


def _stats(trades: list[Trade], equity: pd.Series, start: float) -> dict:
    rs = np.array([t.r for t in trades if t.r is not None])
    if len(rs) == 0:
        return {"trades": 0}
    wins, losses = rs[rs > 0], rs[rs <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else float("inf")
    return {
        "trades": len(rs),
        "win_rate": f"{len(wins) / len(rs) * 100:.1f}%",
        "avg_R": f"{rs.mean():.3f}",
        "total_R": f"{rs.sum():.2f}",
        "profit_factor": f"{pf:.2f}",
        "return": f"{(equity.iloc[-1] / start - 1) * 100:.2f}%",
        "max_drawdown": f"{_dd(equity):.2f}%",
        "final_equity": f"{equity.iloc[-1]:.2f}",
    }


class _Signals:
    """Shared signal evaluation for both backtest modes."""

    def __init__(self, feats, cfg, model, spread, mc_sims):
        self.feats, self.cfg, self.spread, self.mc_sims = feats, cfg, spread, mc_sims
        s = cfg.strategy
        self.proba = None
        if s.use_ml_filter and model is not None and model.ready:
            valid = feats[FEATURE_COLUMNS].notna().all(axis=1)
            self.proba = np.full(len(feats), np.nan)
            self.proba[valid.to_numpy()] = model.clf.predict_proba(feats.loc[valid, FEATURE_COLUMNS])[:, 1]
        self.comps = bar_components(feats) if cfg.montecarlo.enabled else None
        self.hours = pd.to_datetime(feats["time"]).dt.hour.to_numpy()
        self.mc_runs = 0

    def at(self, i: int):
        start_h, end_h = self.cfg.trading_hours_utc
        if not (start_h <= self.hours[i] < end_h):
            return None
        p = None if self.proba is None else float(self.proba[i])
        sig = generate_signal(self.feats, self.cfg, None, self.spread, i=i, proba=p,
                              comps=self.comps, mc_sims=self.mc_sims)
        if sig.mc is not None:
            self.mc_runs += 1
        return sig if sig.side is not Side.NONE else None


def run_backtest(bars: pd.DataFrame, cfg, model: SignalModel | None = None,
                 spread: float | None = None, start_equity: float = 10_000.0,
                 mc_sims: int | None = None) -> BacktestResult:
    spread = cfg.assumed_spread if spread is None else spread
    s, r = cfg.strategy, cfg.risk
    feats = add_indicators(bars, s).reset_index(drop=True)
    signals = _Signals(feats, cfg, model, spread, mc_sims)

    o, h, l = feats["open"].to_numpy(), feats["high"].to_numpy(), feats["low"].to_numpy()
    atr = feats["atr"].to_numpy()
    times = pd.to_datetime(feats["time"])

    equity = start_equity
    curve = np.full(len(feats), np.nan)
    trades: list[Trade] = []
    open_t: Trade | None = None
    warmup = max(s.trend_ema, cfg.montecarlo.horizon_bars if cfg.montecarlo.enabled else 0)

    for i in range(warmup, len(feats) - 1):
        if open_t is not None:
            hit = None
            # Buys exit at the bid, sells exit at the ask (bid + spread).
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

        if open_t is not None:
            continue
        sig = signals.at(i)
        if sig is None:
            continue
        entry = o[i + 1] + (spread if sig.side is Side.BUY else 0.0)
        sl, tp = stop_levels(sig.side.value, entry, atr[i], r)
        open_t = Trade(sig.side.value, times[i + 1], entry, sl, tp, reason=sig.reason)

    curve[-1] = equity
    eq = pd.Series(curve, index=times).ffill().dropna()
    stats = _stats(trades, eq, start_equity)
    if signals.mc_runs:
        stats["monte_carlo_runs"] = signals.mc_runs
    return BacktestResult(trades, eq, start_equity, stats)


def run_basket_backtest(bars: pd.DataFrame, cfg, model: SignalModel | None = None,
                        spread: float | None = None, start_equity: float = 10_000.0,
                        contract_size: float = 100.0, mc_sims: int | None = None) -> BacktestResult:
    """Backtest of the recovery basket.

    Sell baskets are simulated as buy baskets on the mirrored price series
    (bid' = -ask), so one code path handles both directions exactly.
    """
    spread = cfg.assumed_spread if spread is None else spread
    s, bc = cfg.strategy, cfg.basket
    feats = add_indicators(bars, s).reset_index(drop=True)
    signals = _Signals(feats, cfg, model, spread, mc_sims)

    o, h, l, c = (feats[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = feats["atr"].to_numpy()
    times = pd.to_datetime(feats["time"])
    vpp = contract_size  # account money per 1.0 price move per 1.0 lot

    def bar(i, side):
        """BID bar in 'buy space'. Entries at bid + spread, exits at bid."""
        if side == "BUY":
            return o[i], h[i], l[i], c[i]
        return -(o[i] + spread), -(l[i] + spread), -(h[i] + spread), -(c[i] + spread)

    balance = start_equity
    curve = np.full(len(feats), np.nan)
    trades: list[Trade] = []
    basket: Basket | None = None     # always stored in buy space
    real_side = None
    b_atr = b_balance = 0.0
    b_start = None
    warmup = max(s.trend_ema, cfg.montecarlo.horizon_bars if cfg.montecarlo.enabled else 0)
    hard_stops = 0
    layer_hist: list[int] = []

    def close(i, px, reason):
        nonlocal balance, basket, hard_stops
        pnl = basket.pnl(px, vpp)
        balance += pnl
        sgn = 1 if real_side == "BUY" else -1
        trades.append(Trade(real_side, b_start, sgn * basket.legs[0].entry, 0, 0,
                            times[i], sgn * px, r=pnl / b_balance * 100, layers=len(basket.legs),
                            pnl=pnl, reason=reason))
        layer_hist.append(len(basket.legs))
        if reason == "hard stop":
            hard_stops += 1
        basket = None

    for i in range(warmup, len(feats) - 1):
        if basket is not None:
            bo, bh, bl, bclose = bar(i, real_side)
            step, tp_dist = b_atr * bc.step_atr, b_atr * bc.tp_atr
            max_loss = b_balance * bc.max_basket_loss_pct / 100
            stop = basket.stop(max_loss, vpp)
            if bl <= stop:
                close(i, min(bo, stop), "hard stop")
            else:
                added = False
                while len(basket.legs) < bc.max_layers:
                    lvl = basket.next_level(step)
                    if bl + spread > lvl:
                        break
                    fill = min(bo + spread, lvl)
                    basket = basket.with_leg(fill, next_lot(basket, bc))
                    added = True
                    stop = basket.stop(max_loss, vpp)
                if added and bl <= stop:
                    close(i, stop, "hard stop")
                elif not added and bh >= basket.target(tp_dist):
                    close(i, max(bo, basket.target(tp_dist)), "take profit")
                else:
                    curve[i] = balance + basket.pnl(bl, vpp)   # worst point of the bar
                    continue
        curve[i] = balance

        sig = signals.at(i)
        if sig is None:
            continue
        real_side = sig.side.value
        bo, _, _, _ = bar(i + 1, real_side)
        basket = Basket("BUY")
        basket = basket.with_leg(bo + spread, next_lot(basket, bc))
        b_atr, b_balance, b_start = atr[i], balance, times[i + 1]

    if basket is not None:
        close(len(feats) - 1, bar(len(feats) - 1, real_side)[3], "end of data")
    curve[-1] = balance
    eq = pd.Series(curve, index=times).ffill().dropna()

    pnls = np.array([t.pnl for t in trades])
    stats: dict = {"baskets": len(trades)}
    if len(pnls):
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
        stats.update({
            "win_rate": f"{len(wins) / len(pnls) * 100:.1f}%",
            "avg_win": f"{wins.mean():.2f}" if len(wins) else "-",
            "avg_loss": f"{losses.mean():.2f}" if len(losses) else "-",
            "largest_loss": f"{pnls.min():.2f}",
            "profit_factor": f"{pf:.2f}",
            "hard_stops_hit": hard_stops,
            "avg_layers": f"{np.mean(layer_hist):.2f}",
            "max_layers_used": int(max(layer_hist)),
            "return": f"{(balance / start_equity - 1) * 100:.2f}%",
            "max_drawdown(float)": f"{_dd(eq):.2f}%",
            "final_balance": f"{balance:.2f}",
        })
    if signals.mc_runs:
        stats["monte_carlo_runs"] = signals.mc_runs
    return BacktestResult(trades, eq, start_equity, stats)
