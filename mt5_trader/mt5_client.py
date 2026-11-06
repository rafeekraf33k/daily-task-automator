"""Thin wrapper around the official `MetaTrader5` Python package.

The MetaTrader5 package only works on Windows, alongside an installed and
logged-in MT5 terminal. It is imported lazily so the rest of the project
(backtesting, training on CSV, tests) runs on any OS.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from .risk import SymbolSpec

log = logging.getLogger(__name__)


def _mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on platform
        raise RuntimeError(
            "The MetaTrader5 package is not installed. It requires Windows: "
            "pip install MetaTrader5"
        ) from exc
    return mt5


class MT5Client:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mt5 = _mt5()

    # ---- connection -------------------------------------------------------
    def connect(self) -> None:
        kwargs = {}
        if self.cfg.terminal_path:
            kwargs["path"] = self.cfg.terminal_path
        if self.cfg.login:
            kwargs.update(login=self.cfg.login, password=self.cfg.password, server=self.cfg.server)
        if not self.mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize failed: {self.mt5.last_error()}")
        acc = self.mt5.account_info()
        if acc is None:
            raise RuntimeError(f"Not logged in to a trading account: {self.mt5.last_error()}")
        mode = {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(acc.trade_mode, str(acc.trade_mode))
        log.info("Connected: account %s (%s) on %s, balance %.2f %s",
                 acc.login, mode, acc.server, acc.balance, acc.currency)
        self.account_mode = mode
        # One MT5 login = one account. e.g. XAUUSDm exists on Exness Standard
        # accounts and XAUUSDc on Cent accounts, so skip what this account lacks.
        available = []
        for sym in self.cfg.symbols:
            if self.mt5.symbol_select(sym, True):
                available.append(sym)
            else:
                log.warning("Symbol %s not available on this account - skipping", sym)
        if not available:
            raise RuntimeError(f"None of {self.cfg.symbols} are available on account {acc.login}")
        self.cfg.symbols = available

    def shutdown(self) -> None:
        self.mt5.shutdown()

    # ---- market data ------------------------------------------------------
    def timeframe(self, name: str):
        tf = getattr(self.mt5, f"TIMEFRAME_{name.upper()}", None)
        if tf is None:
            raise ValueError(f"Unknown timeframe {name}")
        return tf

    def bars(self, symbol: str, timeframe: str, count: int, include_open: bool = False) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos(symbol, self.timeframe(timeframe), 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates for {symbol}: {self.mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        if not include_open:
            df = df.iloc[:-1]  # last bar is still forming
        return df.reset_index(drop=True)

    def spec(self, symbol: str) -> SymbolSpec:
        i = self.mt5.symbol_info(symbol)
        return SymbolSpec(
            point=i.point, tick_size=i.trade_tick_size, tick_value=i.trade_tick_value,
            volume_min=i.volume_min, volume_max=i.volume_max, volume_step=i.volume_step,
            digits=i.digits,
        )

    def tick(self, symbol: str):
        return self.mt5.symbol_info_tick(symbol)

    def spread_points(self, symbol: str) -> float:
        t, i = self.tick(symbol), self.mt5.symbol_info(symbol)
        return (t.ask - t.bid) / i.point

    def spread_price(self, symbol: str) -> float:
        t = self.tick(symbol)
        return t.ask - t.bid

    # ---- account ----------------------------------------------------------
    def equity(self) -> float:
        return float(self.mt5.account_info().equity)

    def balance(self) -> float:
        return float(self.mt5.account_info().balance)

    def positions(self, symbol: str | None = None) -> list:
        pos = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        return [p for p in (pos or []) if p.magic == self.cfg.magic_number]

    def server_time(self) -> datetime:
        return datetime.now(timezone.utc)

    # ---- orders -----------------------------------------------------------
    def _filling(self, symbol: str):
        mode = self.mt5.symbol_info(symbol).filling_mode
        if mode & 1:
            return self.mt5.ORDER_FILLING_FOK
        if mode & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def market_order(self, symbol: str, side: str, volume: float, sl: float, tp: float, comment: str = ""):
        mt5 = self.mt5
        t = self.tick(symbol)
        digits = mt5.symbol_info(symbol).digits
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": t.ask if side == "BUY" else t.bid,
            "sl": round(sl, digits),
            "tp": round(tp, digits),
            "deviation": self.cfg.deviation_points,
            "magic": self.cfg.magic_number,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(symbol),
        }
        res = mt5.order_send(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"order_send failed: {getattr(res, 'retcode', None)} "
                               f"{getattr(res, 'comment', mt5.last_error())}")
        return res

    def close_position(self, position, comment: str = "mt5_trader close") -> None:
        mt5 = self.mt5
        t = self.tick(position.symbol)
        is_buy = position.type == mt5.POSITION_TYPE_BUY
        res = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": t.bid if is_buy else t.ask,
            "deviation": self.cfg.deviation_points,
            "magic": self.cfg.magic_number,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(position.symbol),
        })
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"close failed for {position.ticket}: {getattr(res, 'comment', mt5.last_error())}")

    def modify_sltp(self, position, sl: float, tp: float) -> None:
        mt5 = self.mt5
        digits = mt5.symbol_info(position.symbol).digits
        res = mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": position.ticket,
            "sl": round(sl, digits),
            "tp": round(tp, digits),
            "magic": self.cfg.magic_number,
        })
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            log.warning("SL/TP modify failed for %s: %s", position.ticket, getattr(res, "comment", None))

    def modify_sl(self, position, new_sl: float) -> None:
        mt5 = self.mt5
        digits = mt5.symbol_info(position.symbol).digits
        res = mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": position.ticket,
            "sl": round(new_sl, digits),
            "tp": position.tp,
            "magic": self.cfg.magic_number,
        })
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            log.warning("SL modify failed for %s: %s", position.ticket, getattr(res, "comment", None))
