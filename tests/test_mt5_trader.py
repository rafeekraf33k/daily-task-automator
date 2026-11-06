from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
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
    def spread_price(self, symbol): return 0.0001
    def balance(self): return 10_000.0
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


# ---------------------------------------------------------------- sweeps
def test_sweep_columns_have_no_lookahead():
    bars = synthetic_bars(800)
    s = StrategyConfig()
    full = add_indicators(bars, s)
    part = add_indicators(bars.iloc[:500], s)
    cols = ["swing_low", "swing_high", "sweep_bull", "sweep_bear"]
    pd.testing.assert_frame_equal(full.iloc[:500][cols], part[cols])


def test_sweep_detected_on_stop_hunt():
    # V-shaped market: swing low 99.0 at bar 20, then bar 40 wicks to 98.5 and closes above 99
    n = 60
    lows = np.array([99.0 + abs(i - 20) * 0.05 for i in range(40)] + [98.5] +
                    [99.6 + j * 0.05 for j in range(n - 41)])
    close = lows + 1.0
    bars = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "open": close, "high": lows + 1.5, "low": lows, "close": close, "tick_volume": 100,
    })
    s = StrategyConfig(swing_len=3, sweep_min_atr=0.0)
    f = add_indicators(bars, s)
    assert f["swing_low"].iloc[39] == 99.0
    assert bool(f["sweep_bull"].iloc[40])
    assert not f["sweep_bull"].drop(index=40).any()


# ----------------------------------------------------------- monte carlo
from mt5_trader.montecarlo import bar_components, simulate  # noqa: E402


def test_montecarlo_symmetric_without_drift():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.001, 3000)
    comps = np.column_stack([r, np.abs(r) + 0.0005, -np.abs(r) - 0.0005])
    comps = np.vstack([comps, -comps])  # perfectly symmetric up/down history
    res = simulate(comps, 2000.0, 0.0, 4.0, 8.0, n_sims=200_000, seed=1)
    assert abs(res.buy_ev - res.sell_ev) < 0.05
    with_spread = simulate(comps, 2000.0, 0.3, 4.0, 8.0, n_sims=200_000, seed=1)
    assert with_spread.buy_ev < res.buy_ev and with_spread.sell_ev < res.sell_ev


def test_montecarlo_follows_drift():
    up = bar_components(synthetic_bars(3000, start_price=2000))
    up[1:] += 0.0003
    res = simulate(up, 2000.0, 0.3, 4.0, 8.0, n_sims=100_000, seed=2)
    assert res.best(0.05) == "BUY"
    res_dn = simulate(-up[:, [0, 2, 1]], 2000.0, 0.3, 4.0, 8.0, n_sims=100_000, seed=2)
    assert res_dn.best(0.05) == "SELL"


def test_montecarlo_runs_one_million():
    comps = bar_components(synthetic_bars(3000, start_price=2000))
    res = simulate(comps, 2000.0, 0.3, 4.0, 8.0, n_sims=1_000_000, seed=3)
    assert res.n_sims == 1_000_000
    assert 0 <= res.buy_tp_rate + res.buy_sl_rate <= 1


# ---------------------------------------------------------------- basket
from mt5_trader.backtest import run_basket_backtest  # noqa: E402
from mt5_trader.basket import Basket, Leg, next_lot  # noqa: E402
from mt5_trader.config import BasketConfig, MonteCarloConfig  # noqa: E402


def test_basket_math():
    b = Basket("BUY", [Leg(2000.0, 0.01), Leg(1990.0, 0.01)])
    assert b.avg == pytest.approx(1995.0)
    assert b.next_level(10) == pytest.approx(1980.0)
    assert b.target(5) == pytest.approx(2000.0)
    stop = b.stop(100.0, 100.0)       # $100 max loss, gold contract 100
    assert b.pnl(stop, 100.0) == pytest.approx(-100.0)
    s = Basket("SELL", [Leg(2000.0, 0.02)])
    assert s.pnl(s.stop(50.0, 100.0), 100.0) == pytest.approx(-50.0)
    assert s.next_level(10) == pytest.approx(2010.0)
    assert next_lot(b, BasketConfig(base_lot=0.01, lot_multiplier=2.0)) == pytest.approx(0.04)


def test_basket_config_rejects_extreme_martingale():
    with pytest.raises(ValueError):
        BasketConfig(lot_multiplier=3.0)


def test_basket_backtest_respects_hard_stop():
    cfg = Config(assumed_spread=0.3, trading_hours_utc=(0, 24))
    cfg.strategy = StrategyConfig(mode="sweep_trend", use_ml_filter=False)
    cfg.basket = BasketConfig(enabled=True, max_layers=4, max_basket_loss_pct=5.0)
    res = run_basket_backtest(synthetic_bars(8000, start_price=2000), cfg, contract_size=100.0,
                              start_equity=1000.0)
    assert res.stats["baskets"] > 0
    for t in res.trades:
        assert t.layers <= 4
        assert t.r >= -5.0 * 1.5   # loss per basket bounded by the hard stop (+ gap slack)
    assert {t.side for t in res.trades} <= {"BUY", "SELL"}


def test_backtest_with_montecarlo():
    cfg = Config(assumed_spread=0.3, trading_hours_utc=(0, 24))
    cfg.strategy = StrategyConfig(mode="sweep_trend", use_ml_filter=False)
    cfg.montecarlo = MonteCarloConfig(enabled=True, lookback_bars=1000, seed=0)
    res = run_backtest(synthetic_bars(3000, start_price=2000), cfg, mc_sims=2000)
    assert res.stats.get("monte_carlo_runs", 0) > 0


class FakePos(SimpleNamespace):
    pass


def test_live_basket_adds_layer_and_closes_in_profit():
    cfg = Config(live=True)
    cfg.strategy.use_ml_filter = False
    cfg.basket = BasketConfig(enabled=True, step_atr=1.0, tp_atr=1.0, max_layers=3)
    bars = synthetic_bars(400, start_price=2000)
    client = FakeClient(bars)
    gold = SymbolSpec(point=0.01, tick_size=0.01, tick_value=1.0, volume_min=0.01,
                      volume_max=100, volume_step=0.01, digits=2)
    client.spec = lambda sym: gold
    client.modified, client.closed = [], []
    client.modify_sltp = lambda p, sl, tp: client.modified.append((p.ticket, sl, tp))
    client.close_position = lambda p, comment="": client.closed.append(p.ticket)
    bot = TradingBot(cfg, client)
    feats = add_indicators(bars, cfg.strategy)
    atr = float(feats["atr"].iloc[-1])

    pos = [FakePos(ticket=1, symbol="XAUUSDm", type=0, price_open=2000.0, volume=0.01, sl=0.0, tp=0.0)]
    # price fell 1.5 ATR below the entry -> add a layer
    px = 2000.0 - 1.5 * atr
    client.tick = lambda s: SimpleNamespace(bid=px, ask=px + 0.3)
    client.positions = lambda symbol=None: pos
    bot._manage_basket("XAUUSDm", pos, feats)
    assert len(client.orders) == 1 and client.orders[0][1] == "BUY"
    assert client.modified  # SL/TP synced

    # price rallies past average + tp -> close everything
    pos.append(FakePos(ticket=2, symbol="XAUUSDm", type=0, price_open=px + 0.3, volume=0.01, sl=0, tp=0))
    hi = 2000.0 + 2 * atr
    client.tick = lambda s: SimpleNamespace(bid=hi, ask=hi + 0.3)
    bot._manage_basket("XAUUSDm", pos, feats)
    assert sorted(client.closed) == [1, 2]
