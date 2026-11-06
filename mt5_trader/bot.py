"""Live trading loop."""
from __future__ import annotations

import logging
import time
from pathlib import Path

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
                            "Trading on rule signals only.", mp)

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

    def _process_symbol(self, symbol: str, equity: float) -> None:
        s = self.cfg.strategy
        bars = self.client.bars(symbol, self.cfg.timeframe, self.cfg.history_bars)
        feats = add_indicators(bars, s)
        self._trail(symbol, feats)

        bar_time = feats["time"].iloc[-1]
        if self.last_bar_time.get(symbol) == bar_time:
            return  # already evaluated this closed bar
        self.last_bar_time[symbol] = bar_time

        sig = generate_signal(feats, s, self.model)
        log.info("%s %s bar %s -> %s (%s)", symbol, self.cfg.timeframe, bar_time, sig.side.value, sig.reason)
        if sig.side is Side.NONE:
            return
        if self.client.positions(symbol):
            log.info("%s: already have a position, skipping", symbol)
            return
        if not self.in_trading_hours():
            log.info("%s: outside trading hours, skipping", symbol)
            return

        ok, why = self.guard.can_open(equity, len(self.client.positions()), self.client.spread_points(symbol))
        if not ok:
            log.info("%s: blocked by risk guard: %s", symbol, why)
            return

        tick = self.client.tick(symbol)
        entry = tick.ask if sig.side is Side.BUY else tick.bid
        sl, tp = stop_levels(sig.side.value, entry, sig.atr, self.cfg.risk)
        lots = lot_size(equity, self.cfg.risk.risk_per_trade_pct, entry, sl, self.client.spec(symbol))
        if lots <= 0:
            log.info("%s: position size below broker minimum for this risk, skipping", symbol)
            return

        summary = f"{sig.side.value} {lots} {symbol} @ {entry} SL {sl:.5f} TP {tp:.5f}"
        if not self.cfg.live:
            log.info("[DRY RUN] would place: %s", summary)
            return
        res = self.client.market_order(symbol, sig.side.value, lots, sl, tp, comment="mt5_trader")
        log.info("Placed: %s (order %s)", summary, res.order)

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
        log.info("Starting bot in %s mode for %s on %s", mode, self.cfg.symbols, self.cfg.timeframe)
        while True:
            try:
                self.step()
            except Exception:
                log.exception("Loop error")
            time.sleep(self.cfg.poll_seconds)
