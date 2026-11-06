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
    sl_atr_mult: float = 1.5             # stop loss = ATR * mult
    tp_atr_mult: float = 3.0             # take profit = ATR * mult (2:1 reward/risk by default)
    trailing_atr_mult: float = 0.0       # 0 disables trailing stop


@dataclass
class StrategyConfig:
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
class Config:
    symbols: list[str] = field(default_factory=lambda: ["EURUSD"])
    timeframe: str = "M15"
    history_bars: int = 1000
    poll_seconds: int = 30
    magic_number: int = 20261106
    deviation_points: int = 20
    live: bool = False                   # False = dry run: signals are logged, no orders sent
    trading_hours_utc: tuple[int, int] = (7, 20)   # only open trades between these hours
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

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
    if "trading_hours_utc" in data:
        data["trading_hours_utc"] = tuple(data["trading_hours_utc"])
    cfg = _build(Config, {**data, "risk": risk, "strategy": strategy})

    login = os.getenv("MT5_LOGIN")
    cfg.login = int(login) if login else None
    cfg.password = os.getenv("MT5_PASSWORD")
    cfg.server = os.getenv("MT5_SERVER")
    cfg.terminal_path = os.getenv("MT5_TERMINAL_PATH")
    return cfg
