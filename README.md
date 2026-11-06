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
  backtest.py    bar-by-bar backtester (next-bar fills, spread, SL-first)
  mt5_client.py  MetaTrader5 API wrapper (orders, data, positions)
  bot.py         live loop
  __main__.py    CLI
tests/           unit tests
```
