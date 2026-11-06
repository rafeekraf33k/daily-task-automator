# daily-task-automator

## MT5 Trading AI (`mt5_trader`)

An automated trading bot for MetaTrader 5, written in Python. It combines:

1. **A rule-based strategy**: it buys pullbacks in an uptrend and sells pullbacks in a downtrend, using EMA 20/50/200 and RSI.
2. **A machine-learning filter**: a gradient-boosting model predicts the next ~12 bars' direction from 16 features and **only lets through trades it agrees with**.
3. **A risk engine**: ATR-based stop-loss/take-profit, position sizing at a fixed % of equity, a daily loss limit, a cap on open positions, a spread filter, trading hours and an optional trailing stop.

> ⚠️ **Read this first.** No bot, AI or otherwise, can promise profits. Most retail CFD/forex accounts lose money. Run this on a **demo account** for weeks before you consider real money, and never risk money you can't afford to lose. By default the bot runs in **dry-run mode** and sends no orders. It refuses to trade a REAL account unless you pass an explicit extra flag.

### Requirements

- **Windows** (the official `MetaTrader5` Python package only works there), or a Windows VPS
- The MetaTrader 5 terminal installed and logged in to your broker account
- In MT5: *Tools → Options → Expert Advisors → enable "Allow algorithmic trading"*, and press the **Algo Trading** button on the toolbar
- Python 3.10+

Backtesting, training on CSV files and the tests also run on macOS and Linux.

### Setup

```bat
git clone https://github.com/rafeekraf33k/daily-task-automator.git
cd daily-task-automator
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy config.example.yaml config.yaml
```

Optional: set your credentials in environment variables. If you skip this, the bot uses the account the terminal is already logged in to.

```bat
set MT5_LOGIN=12345678
set MT5_PASSWORD=your_password
set MT5_SERVER=YourBroker-Demo
```

### Workflow

```bat
:: 1. Download history from MT5
python -m mt5_trader download --symbol EURUSD --bars 50000

:: 2. Backtest. The ML model trains on the first 50% and is tested on the unseen 50%.
::    --spread is in price units (0.0001 = 1 pip on EURUSD)
python -m mt5_trader backtest --csv data/EURUSD_M15.csv --spread 0.0001
python -m mt5_trader backtest --csv data/EURUSD_M15.csv --spread 0.0001 --no-ml

:: 3. Train the final model on all of the data
python -m mt5_trader train --csv data/EURUSD_M15.csv

:: 4. Dry run: logs what it WOULD trade, places no orders
python -m mt5_trader run

:: 5. Live on a DEMO account
python -m mt5_trader run --live
```

Logs are written to `logs/mt5_trader.log`. Stop the bot with `Ctrl+C`. Positions you already opened keep their SL/TP on the broker's server.

### Gold: XAUUSDm / XAUUSDc setup (`config.gold.yaml`)

A ready-made config for gold that combines everything below: liquidity sweep + trend entries, 1,000,000 Monte Carlo paths before every trade, a $0.30 spread, and the recovery basket.

```bat
python -m mt5_trader -c config.gold.yaml download --symbol XAUUSDm --bars 50000
:: Backtest. Use fewer MC paths here (1M per signal x hundreds of signals = slow)
python -m mt5_trader -c config.gold.yaml backtest --csv data/XAUUSDm_M15.csv --symbol XAUUSDm --equity 1000 --mc-sims 50000
python -m mt5_trader -c config.gold.yaml run          :: dry run, full 1M paths
python -m mt5_trader -c config.gold.yaml run --live   :: demo first!
```

- **One account per terminal.** On Exness, `XAUUSDm` is on Standard accounts and `XAUUSDc` is on **Cent** accounts, so one login won't have both. The bot skips any symbol your account doesn't have. To run both, install MT5 twice (two folders), log each one in to its own account, and start two bots:
  ```bat
  set MT5_TERMINAL_PATH=C:\MT5_Standard\terminal64.exe
  python -m mt5_trader -c config.gold.yaml run --live
  :: second window
  set MT5_TERMINAL_PATH=C:\MT5_Cent\terminal64.exe
  python -m mt5_trader -c config.gold.yaml run --live
  ```
- **Cent accounts** show the balance in US cents (USC). Lot sizing and the basket hard stop use your account currency, so they work as-is. In a backtest, pass `--equity` in the same units.
- Raise *Tools → Options → Charts → Max bars in chart* if `download` returns fewer bars than you asked for.

### Liquidity sweep + trend (`strategy.mode: sweep_trend`)

- **Swing lows/highs** are bars that are the lowest/highest of the `swing_len` bars on each side. They're where stop-losses cluster. A swing is only treated as known `swing_len` bars later, so there's no look-ahead.
- **Buy:** the trend is up (price > EMA200 and EMA50 > EMA200) **and** a bar's wick sweeps below the last swing low (a stop hunt) but closes back above it.
- **Sell:** the mirror image at the last swing high in a downtrend.

### Monte Carlo: "run 1 million possibilities first" (`montecarlo`)

Before each trade the bot simulates `n_sims` (1,000,000) possible paths for the next `horizon_bars`. It builds them by resampling random blocks of real recent bars, with their returns and wicks. For **both** a buy and a sell, with the real spread and the same SL/TP, it counts how often TP is hit before SL and computes the expected result in R. The trade is taken only if the setup's direction has the better expected value **and** that value is at least `min_edge_r`. The result is logged with every signal, for example:

`MC 1,000,000 paths: BUY EV +0.120R (TP 37.4%/SL 62.6%), SELL EV -0.310R (...)`

It runs in about 2 seconds. It's a statistical filter based on recent market behaviour, not a prediction of news or regime changes.

### Recovery basket: add lots on reversals, close all in profit (`basket`)

1. A signal opens the first position (`base_lot`).
2. Each time price moves `step_atr × ATR` **against** the basket, another position is added in the same direction, up to `max_layers`.
3. All positions share one take-profit at **average entry + `tp_atr × ATR`**. A small bounce closes the whole basket in profit.
4. All positions also share one **hard stop**, placed where the basket would lose `max_basket_loss_pct` of the balance. The bot won't add a layer that would put the basket at its stop. TP and stop are set on the broker server, so they still work if your PC goes down.

> ⚠️ **Basket/grid risk.** This style usually wins 85–90% of the time with small profits, then takes a rare loss that can erase many wins. In a backtest on random data it won about 90% of baskets and still lost money overall. **Always backtest it on real data and watch `largest_loss`, `hard_stops_hit` and `max_drawdown(float)`, not just the win rate.** `lot_multiplier > 1` (martingale) makes the rare loss much bigger; it's capped at 2.0 and defaults to 1.0.

### How it decides to trade (on every closed bar)

| Step | Check |
|---|---|
| Trend | Price above EMA200 and EMA20 > EMA50 (buy), or the mirror image (sell) |
| Entry | Price crosses back over EMA20 after a pullback, RSI 45–70 (buy) / 30–55 (sell) |
| AI filter | The model's probability for that direction is ≥ `ml_min_proba` |
| Risk | No open position on that symbol, under the max open positions, daily loss under the limit, spread OK, inside trading hours |
| Order | Market order, SL = 1.5×ATR, TP = 3×ATR, size so that the SL costs 1% of equity |

All of these values can be changed in `config.yaml`.

### Testing without MT5

```bash
python -m pytest tests
python -m mt5_trader backtest --synthetic --spread 0.0001   # random data, pipeline check only
```

### Project layout

```
mt5_trader/
  config.py      settings (YAML) + credentials (env vars)
  indicators.py  EMA, RSI, ATR, MACD + ML features
  model.py       ML signal filter (train/save/load)
  strategy.py    entry rules + ML confirmation
  risk.py        position sizing, SL/TP, daily-loss guard
  montecarlo.py  1M-path Monte Carlo BUY vs SELL evaluation
  basket.py      recovery basket math (layers, shared TP, hard stop)
  backtest.py    bar-by-bar backtester, single-trade and basket modes
  mt5_client.py  MetaTrader5 API wrapper (orders, data, positions)
  bot.py         live loop
  __main__.py    CLI
tests/           unit tests
```
