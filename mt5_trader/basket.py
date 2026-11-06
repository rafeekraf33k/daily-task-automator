"""Recovery basket ("averaging") position management.

How it works:
  1. A signal opens the first position.
  2. If price moves AGAINST the basket by `step_atr * ATR` from the last
     entry, another position is opened in the same direction (up to
     `max_layers`). Each new lot = base_lot * lot_multiplier ** layer.
  3. All positions share one take-profit: the average entry price plus
     `tp_atr * ATR`. Because the average improves with each layer, a smaller
     bounce is enough to close the whole basket in profit.
  4. All positions also share one HARD STOP, placed where the whole basket
     would lose `max_basket_loss_pct` of the balance. Both TP and stop are set
     on the broker's server, so they work even if the bot or PC goes down.

WARNING: averaging wins often and loses rarely but big. In a strong trend
against you the hard stop will be hit, and that one loss can wipe out many
wins. Never run it without the hard stop and keep lot_multiplier at 1.0
unless you fully understand martingale risk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Leg:
    entry: float
    lots: float


@dataclass
class Basket:
    side: str                      # "BUY" or "SELL"
    legs: list[Leg] = field(default_factory=list)

    @property
    def sign(self) -> int:
        return 1 if self.side == "BUY" else -1

    @property
    def lots(self) -> float:
        return sum(l.lots for l in self.legs)

    @property
    def avg(self) -> float:
        return sum(l.entry * l.lots for l in self.legs) / self.lots

    @property
    def last_entry(self) -> float:
        """The entry furthest against us (lowest buy / highest sell)."""
        entries = [l.entry for l in self.legs]
        return min(entries) if self.side == "BUY" else max(entries)

    def next_level(self, step: float) -> float:
        """Entry-side price at which the next layer is added."""
        return self.last_entry - self.sign * step

    def target(self, tp_dist: float) -> float:
        """Exit-side price at which the whole basket is closed in profit."""
        return self.avg + self.sign * tp_dist

    def stop(self, max_loss_money: float, value_per_price: float) -> float:
        """Exit-side price at which the basket has lost `max_loss_money`."""
        return self.avg - self.sign * max_loss_money / (value_per_price * self.lots)

    def pnl(self, exit_price: float, value_per_price: float) -> float:
        return sum(self.sign * (exit_price - l.entry) * l.lots for l in self.legs) * value_per_price

    def with_leg(self, entry: float, lots: float) -> "Basket":
        return Basket(self.side, self.legs + [Leg(entry, lots)])


def round_lot(lots: float, step: float, vmin: float, vmax: float) -> float:
    lots = math.floor(lots / step + 1e-9) * step
    return round(min(max(lots, vmin), vmax), 8)


def next_lot(basket: Basket, cfg, step: float = 0.01, vmin: float = 0.01, vmax: float = 100.0) -> float:
    raw = cfg.base_lot * cfg.lot_multiplier ** len(basket.legs)
    return round_lot(raw, step, vmin, vmax)
