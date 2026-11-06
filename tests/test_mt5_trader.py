from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from mt5_trader.backtest import run_backtest
from mt5_trader.bot import TradingBot
from mt5_trader.config import Config, RiskConfig, StrategyConfig, load_config
from mt5_trader.data import synthetic_bars
from mt5_trader.indicators import FEATURE_COLUMNS, add_indicators, rsi
from mt5_trader.model import SignalModel, make_labels
from mt5_trader.risk import RiskGuard, SymbolSpec, lot_size, stop_levels

EURUSD = SymbolSpec(point=0.00001, tick_size=0.00001, tick_value=1.0,
                    volume_min=0.01, volume_max=100, volume_step=0.01, digits=5)


def test_lot_size_risks_requested_amount():
    # 10k equity, 1% = $100. 20 pip stop = 200 ticks * $1 = $200/lot -> 0.5 lots
    assert lot_size(10_000, 1.0, 1.10000, 1.09800, EURUSD) == pytest.approx(0.5)


def test_lot_size_rounds_down_and_respects_minimum():
    assert lot_size(10_000, 1.0, 1.1, 1.0970, EURUSD) == pytest.approx(0.33)
    assert lot_size(50, 0.1, 1.1, 1.0, EURUSD) == 0.0


def test_stop_levels():
    r = RiskConfig(sl_atr_mult=1.5, tp_atr_mult=3.0)
    assert stop_levels("BUY", 1.1, 0.001, r) == pytest.approx((1.0985, 1.103))
    assert stop_levels("SELL", 1.1, 0.001, r) == pytest.approx((1.1015, 1.097))


def test_risk_guard():
    g = RiskGuard(RiskConfig(max_daily_loss_pct=3, max_open_positions=2, max_spread_points=20))
    g.start_day_if_needed("d1", 10_000)
    assert g.can_open(10_000, 0, 10)[0]
    assert not g.can_open(9_600, 0, 10)[0]          # -4% today
    assert not g.can_open(10_000, 2, 10)[0]         # too many positions
    assert not g.can_open(10_000, 0, 25)[0]         # spread too wide
    g.start_day_if_needed("d2", 9_600)              # new day resets the baseline
    assert g.can_open(9_600, 0, 10)[0]


def test_indicators_have_no_lookahead():
    bars = synthetic_bars(600)
    s = StrategyConfig()
    full = add_indicators(bars, s)
    partial = add_indicators(bars.iloc[:400], s)
    pd.testing.assert_frame_equal(full.iloc[:400][FEATURE_COLUMNS], partial[FEATURE_COLUMNS])


def test_rsi_bounds():
    r = rsi(synthetic_bars(500)["close"])
    assert r.between(0, 100).all()


def test_labels_last_rows_nan():
    lab = make_labels(synthetic_bars(100), 5)
    assert lab.iloc[-5:].isna().all() and lab.iloc[:-5].notna().all()


def test_train_and_backtest(tmp_path):
    cfg = Config()
    bars = synthetic_bars(6000)
    model = SignalModel()
    report = model.train(bars.iloc[:4000], cfg.strategy)
    assert 0 <= report.accuracy <= 1
    model.save(tmp_path / "m.joblib")
    loaded = SignalModel.load(tmp_path / "m.joblib")
    res = run_backtest(bars.iloc[4000:].reset_index(drop=True), cfg, loaded, spread=0.0001)
    assert res.equity_curve.iloc[-1] > 0
    for t in res.trades:
        assert t.r >= -1.5  # loss bounded near 1R (gaps can exceed slightly)


def test_load_config(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("symbols: [GBPUSD, XAUUSD]\nrisk:\n  risk_per_trade_pct: 0.5\n")
    monkeypatch.setenv("MT5_LOGIN", "123")
    cfg = load_config(p)
    assert cfg.symbols == ["GBPUSD", "XAUUSD"]
    assert cfg.risk.risk_per_trade_pct == 0.5
    assert cfg.login == 123
    p.write_text("riskk: {}\n")
    with pytest.raises(ValueError):
        load_config(p)


class FakeClient:
    def __init__(self, bars):
        self._bars = bars
        self.orders = []

    def bars(self, symbol, tf, n): return self._bars
    def equity(self): return 10_000.0
    def server_time(self): return datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    def positions(self, symbol=None): return []
    def spread_points(self, symbol): return 10
    def tick(self, symbol):
        c = float(self._bars["close"].iloc[-1])
        return SimpleNamespace(bid=c, ask=c + 0.0001)
    def spec(self, symbol): return EURUSD
    def market_order(self, *a, **k):
        self.orders.append(a)
        return SimpleNamespace(order=1)


def _bars_ending_in_signal():
    cfg = StrategyConfig()
    bars = synthetic_bars(5000)
    feats = add_indicators(bars, cfg)
    from mt5_trader.strategy import Side, rule_signal
    for i in range(len(feats) - 1, cfg.trend_ema, -1):
        if rule_signal(feats.iloc[i], feats.iloc[i - 1])[0] is not Side.NONE:
            return bars.iloc[: i + 1].reset_index(drop=True)
    pytest.skip("no signal in synthetic data")


@pytest.mark.parametrize("live", [False, True])
def test_bot_only_sends_orders_when_live(live, tmp_path):
    cfg = Config(live=live)
    cfg.strategy.use_ml_filter = False
    client = FakeClient(_bars_ending_in_signal())
    bot = TradingBot(cfg, client)
    bot.step()
    assert len(client.orders) == (1 if live else 0)
    bot.step()  # same bar again -> no duplicate order
    assert len(client.orders) == (1 if live else 0)
