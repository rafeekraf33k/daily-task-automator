"""Position sizing and account-level safety checks."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SymbolSpec:
    """The subset of MT5 symbol_info the risk engine needs."""
    point: float
    tick_size: float
    tick_value: float        # account-currency value of one tick for 1.0 lot
    volume_min: float
    volume_max: float
    volume_step: float
    digits: int


def stop_levels(side: str, price: float, atr: float, risk_cfg) -> tuple[float, float]:
    sl_dist = atr * risk_cfg.sl_atr_mult
    tp_dist = atr * risk_cfg.tp_atr_mult
    if side == "BUY":
        return price - sl_dist, price + tp_dist
    return price + sl_dist, price - tp_dist


def lot_size(equity: float, risk_pct: float, entry: float, sl: float, spec: SymbolSpec) -> float:
    """Lots such that hitting `sl` loses ~risk_pct% of equity. Returns 0 if too small."""
    risk_money = equity * risk_pct / 100.0
    ticks = abs(entry - sl) / spec.tick_size
    if ticks <= 0 or spec.tick_value <= 0:
        return 0.0
    loss_per_lot = ticks * spec.tick_value
    raw = risk_money / loss_per_lot
    steps = math.floor(raw / spec.volume_step + 1e-9)
    lots = round(steps * spec.volume_step, 8)
    if lots < spec.volume_min:
        return 0.0  # can't size the trade within the risk budget -> skip it
    return min(lots, spec.volume_max)


class RiskGuard:
    """Blocks new trades when daily loss or exposure limits are hit."""

    def __init__(self, risk_cfg):
        self.cfg = risk_cfg
        self.day = None
        self.day_start_equity = None

    def start_day_if_needed(self, today, equity: float) -> None:
        if today != self.day:
            self.day = today
            self.day_start_equity = equity

    def daily_loss_pct(self, equity: float) -> float:
        if not self.day_start_equity:
            return 0.0
        return max(0.0, (self.day_start_equity - equity) / self.day_start_equity * 100)

    def can_open(self, equity: float, open_positions: int, spread_points: float) -> tuple[bool, str]:
        if self.daily_loss_pct(equity) >= self.cfg.max_daily_loss_pct:
            return False, f"daily loss limit hit ({self.daily_loss_pct(equity):.2f}%)"
        if open_positions >= self.cfg.max_open_positions:
            return False, f"max open positions ({open_positions})"
        if spread_points > self.cfg.max_spread_points:
            return False, f"spread too wide ({spread_points:.0f} pts)"
        return True, "ok"
