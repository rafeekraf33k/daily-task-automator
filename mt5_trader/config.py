"""Configuration loading.

Settings come from a YAML file (see config.example.yaml). Broker credentials
are read from environment variables so they never end up in the repository.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 1.0      # % of equity risked per trade (stop-loss distance)
    max_daily_loss_pct: float = 3.0      # stop trading for the day after this drawdown
    max_open_positions: int = 3          # across all symbols managed by the bot
    max_spread_points: int = 30          # skip entries when spread is wider than this
    max_spread_price: float | None = None  # same, in price units (e.g. 0.5 for gold); overrides points
    sl_atr_mult: float = 1.5             # stop loss = ATR * mult
    tp_atr_mult: float = 3.0             # take profit = ATR * mult (2:1 reward/risk by default)
    trailing_atr_mult: float = 0.0       # 0 disables trailing stop


@dataclass
class StrategyConfig:
    mode: str = "ema_pullback"           # "ema_pullback" or "sweep_trend" (liquidity sweep + trend)
    swing_len: int = 5                   # bars on each side that define a swing high/low
    sweep_min_atr: float = 0.1           # wick must pierce the swing level by >= this * ATR
    fast_ema: int = 20
    slow_ema: int = 50
    trend_ema: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    use_ml_filter: bool = True           # require the ML model to agree with the rule signal
    ml_min_proba: float = 0.55           # minimum model confidence to accept a trade
    ml_horizon: int = 12                 # bars ahead the model predicts
    model_path: str = "models/model.joblib"


@dataclass
class MonteCarloConfig:
    """Before each trade, simulate many possible future price paths and only
    trade if the chosen direction has a positive expected result."""
    enabled: bool = False
    n_sims: int = 1_000_000
    horizon_bars: int = 48               # how far ahead each simulated path runs
    lookback_bars: int = 3000            # recent history the paths are resampled from
    block_len: int = 4                   # consecutive bars resampled together (keeps momentum)
    min_edge_r: float = 0.05             # minimum expected R (after spread) to accept a side
    chunk: int = 50_000                  # paths per batch (memory control)
    seed: int | None = None


@dataclass
class BasketConfig:
    """Recovery basket: add positions when price moves against the basket,
    close everything together once the basket is in profit."""
    enabled: bool = False
    base_lot: float = 0.01
    lot_multiplier: float = 1.0          # 1.0 = equal lots (averaging); >1 = martingale (dangerous)
    max_layers: int = 5                  # max positions in one basket, including the first
    step_atr: float = 1.5                # add a layer each time price moves this * ATR against us
    tp_atr: float = 1.0                  # close all when price is this * ATR past the average entry
    max_basket_loss_pct: float = 10.0    # hard stop: close all if the basket loses this % of balance

    def __post_init__(self):
        if not 1.0 <= self.lot_multiplier <= 2.0:
            raise ValueError("basket.lot_multiplier must be between 1.0 and 2.0")
        if self.max_layers < 1 or self.max_basket_loss_pct <= 0:
            raise ValueError("basket.max_layers must be >= 1 and max_basket_loss_pct > 0")


@dataclass
class Config:
    symbols: list[str] = field(default_factory=lambda: ["EURUSD"])
    timeframe: str = "M15"
    history_bars: int = 1000
    poll_seconds: int = 30
    magic_number: int = 20261106
    deviation_points: int = 20
    live: bool = False                   # False = dry run: signals are logged, no orders sent
    trading_hours_utc: tuple[int, int] = (7, 20)   # only open trades between these hours
    assumed_spread: float = 0.0          # spread (price units) used by backtests and Monte Carlo
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    montecarlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    basket: BasketConfig = field(default_factory=BasketConfig)

    # Credentials (from env vars, never from the YAML file)
    login: int | None = None
    password: str | None = None
    server: str | None = None
    terminal_path: str | None = None


def _build(cls, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None = None) -> Config:
    data: dict = {}
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    risk = _build(RiskConfig, data.pop("risk", {}) or {})
    strategy = _build(StrategyConfig, data.pop("strategy", {}) or {})
    montecarlo = _build(MonteCarloConfig, data.pop("montecarlo", {}) or {})
    basket = _build(BasketConfig, data.pop("basket", {}) or {})
    if strategy.mode not in ("ema_pullback", "sweep_trend"):
        raise ValueError(f"Unknown strategy.mode {strategy.mode!r}")
    if "trading_hours_utc" in data:
        data["trading_hours_utc"] = tuple(data["trading_hours_utc"])
    cfg = _build(Config, {**data, "risk": risk, "strategy": strategy,
                          "montecarlo": montecarlo, "basket": basket})

    login = os.getenv("MT5_LOGIN")
    cfg.login = int(login) if login else None
    cfg.password = os.getenv("MT5_PASSWORD")
    cfg.server = os.getenv("MT5_SERVER")
    cfg.terminal_path = os.getenv("MT5_TERMINAL_PATH")
    return cfg
