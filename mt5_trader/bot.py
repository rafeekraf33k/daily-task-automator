"""Live trading loop (single-trade mode or recovery-basket mode)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from .basket import Basket, Leg, next_lot, round_lot
from .indicators import add_indicators
from .model import SignalModel
from .risk import RiskGuard, lot_size, stop_levels
from .strategy import Side, generate_signal

log = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, cfg, client):
        self.cfg = cfg
        self.client = client
        self.guard = RiskGuard(cfg.risk)
        self.last_bar_time: dict[str, object] = {}
        self.model = None
        mp = Path(cfg.strategy.model_path)
        if cfg.strategy.use_ml_filter:
            if mp.exists():
                self.model = SignalModel.load(mp)
                log.info("Loaded ML model from %s", mp)
            else:
                log.warning("ML filter enabled but %s not found - run `train` first. "
                            "Trading without the ML filter.", mp)

    def in_trading_hours(self) -> bool:
        start, end = self.cfg.trading_hours_utc
        return start <= self.client.server_time().hour < end

    def step(self) -> None:
        equity = self.client.equity()
        self.guard.start_day_if_needed(self.client.server_time().date(), equity)
        for symbol in self.cfg.symbols:
            try:
                self._process_symbol(symbol, equity)
            except Exception:  # keep running for other symbols
                log.exception("Error processing %s", symbol)

    def _open_count(self) -> int:
        pos = self.client.positions()
        if self.cfg.basket.enabled:
            return len({p.symbol for p in pos})  # one basket per symbol
        return len(pos)

    def _process_symbol(self, symbol: str, equity: float) -> None:
        s = self.cfg.strategy
        bars = self.client.bars(symbol, self.cfg.timeframe, self.cfg.history_bars)
        feats = add_indicators(bars, s)
        positions = self.client.positions(symbol)

        # Position management runs on every poll, not just on new bars
        if self.cfg.basket.enabled:
            if positions:
                self._manage_basket(symbol, positions, feats)
        else:
            self._trail(symbol, feats)

        bar_time = feats["time"].iloc[-1]
        if self.last_bar_time.get(symbol) == bar_time:
            return  # already evaluated this closed bar
        self.last_bar_time[symbol] = bar_time
        if positions:
            log.info("%s: position/basket already open, not looking for new entries", symbol)
            return

        spread = max(self.client.spread_price(symbol), self.cfg.assumed_spread)
        sig = generate_signal(feats, self.cfg, self.model, spread=spread)
        log.info("%s %s bar %s -> %s (%s)", symbol, self.cfg.timeframe, bar_time, sig.side.value, sig.reason)
        if sig.side is Side.NONE:
            return
        if not self.in_trading_hours():
            log.info("%s: outside trading hours, skipping", symbol)
            return
        ok, why = self.guard.can_open(equity, self._open_count(), self.client.spread_points(symbol),
                                      self.client.spread_price(symbol))
        if not ok:
            log.info("%s: blocked by risk guard: %s", symbol, why)
            return

        tick = self.client.tick(symbol)
        entry = tick.ask if sig.side is Side.BUY else tick.bid
        spec = self.client.spec(symbol)
        if self.cfg.basket.enabled:
            lots = round_lot(self.cfg.basket.base_lot, spec.volume_step, spec.volume_min, spec.volume_max)
            basket = Basket(sig.side.value, [Leg(entry, lots)])
            tp, sl = self._basket_levels(basket, sig.atr, spec)
        else:
            sl, tp = stop_levels(sig.side.value, entry, sig.atr, self.cfg.risk)
            lots = lot_size(equity, self.cfg.risk.risk_per_trade_pct, entry, sl, spec)
            if lots <= 0:
                log.info("%s: position size below broker minimum for this risk, skipping", symbol)
                return

        summary = f"{sig.side.value} {lots} {symbol} @ {entry} SL {sl:.3f} TP {tp:.3f}"
        if not self.cfg.live:
            log.info("[DRY RUN] would place: %s", summary)
            return
        res = self.client.market_order(symbol, sig.side.value, lots, sl, tp, comment="mt5_trader")
        log.info("Placed: %s (order %s)", summary, res.order)

    # ---- basket -----------------------------------------------------------
    def _basket_levels(self, basket: Basket, atr: float, spec) -> tuple[float, float]:
        """(take-profit, hard-stop) prices shared by every position of the basket."""
        bc = self.cfg.basket
        vpp = spec.tick_value / spec.tick_size
        max_loss = self.client.balance() * bc.max_basket_loss_pct / 100
        return basket.target(atr * bc.tp_atr), basket.stop(max_loss, vpp)

    def _manage_basket(self, symbol: str, positions, feats) -> None:
        bc = self.cfg.basket
        side = "BUY" if positions[0].type == 0 else "SELL"
        basket = Basket(side, [Leg(p.price_open, p.volume) for p in positions])
        spec = self.client.spec(symbol)
        atr = float(feats["atr"].iloc[-1])
        tick = self.client.tick(symbol)
        entry_px = tick.ask if side == "BUY" else tick.bid
        exit_px = tick.bid if side == "BUY" else tick.ask
        sign = basket.sign

        tp, stop = self._basket_levels(basket, atr, spec)
        # Safety net in case the broker-side SL/TP were not set or lagged
        if sign * (exit_px - stop) <= 0 or sign * (exit_px - tp) >= 0:
            why = "hard stop" if sign * (exit_px - stop) <= 0 else "take profit"
            log.info("%s basket %s at %s (%d legs, %.2f lots) - closing all",
                     symbol, why, exit_px, len(basket.legs), basket.lots)
            for p in positions:
                self.client.close_position(p, comment=f"basket {why}")
            return

        # Price moved against us by one step -> add a layer
        if len(basket.legs) < bc.max_layers and sign * (entry_px - basket.next_level(atr * bc.step_atr)) <= 0:
            lots = next_lot(basket, bc, spec.volume_step, spec.volume_min, spec.volume_max)
            bigger = basket.with_leg(entry_px, lots)
            new_tp, new_stop = self._basket_levels(bigger, atr, spec)
            if sign * (exit_px - new_stop) > 0:
                log.info("%s: adding basket layer %d: %s %s @ %s", symbol, len(bigger.legs), side, lots, entry_px)
                self.client.market_order(symbol, side, lots, new_stop, new_tp, comment="mt5_trader layer")
                basket, tp, stop = bigger, new_tp, new_stop
                positions = self.client.positions(symbol)
            else:
                log.info("%s: not adding layer - it would put the basket at its hard stop", symbol)

        # Keep every leg's broker-side SL/TP in sync with the basket levels
        tol = 0.1 * atr
        for p in positions:
            if abs(p.tp - tp) > tol or abs(p.sl - stop) > tol:
                self.client.modify_sltp(p, stop, tp)

    # ---- single-trade trailing stop ----------------------------------------
    def _trail(self, symbol: str, feats) -> None:
        mult = self.cfg.risk.trailing_atr_mult
        if mult <= 0 or not self.cfg.live:
            return
        atr = float(feats["atr"].iloc[-1])
        tick = self.client.tick(symbol)
        for p in self.client.positions(symbol):
            if p.type == 0:  # buy
                new_sl = tick.bid - atr * mult
                if new_sl > p.price_open and (p.sl == 0 or new_sl > p.sl):
                    self.client.modify_sl(p, new_sl)
            else:
                new_sl = tick.ask + atr * mult
                if new_sl < p.price_open and (p.sl == 0 or new_sl < p.sl):
                    self.client.modify_sl(p, new_sl)

    def run_forever(self) -> None:
        mode = "LIVE" if self.cfg.live else "DRY RUN"
        kind = "basket" if self.cfg.basket.enabled else "single-trade"
        log.info("Starting bot in %s %s mode for %s on %s (strategy %s, Monte Carlo %s)",
                 mode, kind, self.cfg.symbols, self.cfg.timeframe, self.cfg.strategy.mode,
                 f"{self.cfg.montecarlo.n_sims:,} paths" if self.cfg.montecarlo.enabled else "off")
        while True:
            try:
                self.step()
            except Exception:
                log.exception("Loop error")
            time.sleep(self.cfg.poll_seconds)
