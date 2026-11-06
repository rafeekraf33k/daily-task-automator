"""Command line interface.

    python -m mt5_trader download --symbol EURUSD --bars 50000
    python -m mt5_trader train    --csv data/EURUSD_M15.csv
    python -m mt5_trader backtest --csv data/EURUSD_M15.csv
    python -m mt5_trader run                 # dry run (no orders)
    python -m mt5_trader run --live          # real orders
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .backtest import run_backtest
from .config import load_config
from .data import load_csv, save_csv, synthetic_bars
from .model import SignalModel


def _setup_logging(verbose: bool) -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("logs/mt5_trader.log")],
    )


def _get_bars(args, cfg):
    if getattr(args, "synthetic", False):
        return synthetic_bars(args.bars or 20000)
    if args.csv:
        return load_csv(args.csv)
    from .mt5_client import MT5Client
    client = MT5Client(cfg)
    client.connect()
    try:
        return client.bars(args.symbol or cfg.symbols[0], cfg.timeframe, args.bars or 50000)
    finally:
        client.shutdown()


def cmd_download(args, cfg):
    bars = _get_bars(args, cfg)
    sym = args.symbol or cfg.symbols[0]
    out = args.out or f"data/{sym}_{cfg.timeframe}.csv"
    save_csv(bars, out)
    print(f"Saved {len(bars)} bars to {out}")


def cmd_train(args, cfg):
    bars = _get_bars(args, cfg)
    model = SignalModel()
    report = model.train(bars, cfg.strategy)
    model.save(cfg.strategy.model_path)
    print(f"Model trained: {report}")
    print(f"Saved to {cfg.strategy.model_path}")
    if report.accuracy <= report.baseline + 0.01:
        print("WARNING: the model is not beating the baseline out-of-sample. "
              "Consider use_ml_filter: false or more/different data.")


def cmd_backtest(args, cfg):
    bars = _get_bars(args, cfg)
    model = None
    if cfg.strategy.use_ml_filter and not args.no_ml:
        # Walk-forward: train on the first part, test only on unseen data
        split = int(len(bars) * args.train_frac)
        model = SignalModel()
        report = model.train(bars.iloc[:split], cfg.strategy)
        print(f"ML model trained on first {args.train_frac:.0%} of data: {report}")
        bars = bars.iloc[split - cfg.strategy.trend_ema:].reset_index(drop=True)
    res = run_backtest(bars, cfg, model, spread=args.spread, start_equity=args.equity)
    print(f"Backtest {bars['time'].iloc[0]} -> {bars['time'].iloc[-1]} ({len(bars)} bars)")
    print(res.summary())
    if args.trades_out:
        import pandas as pd
        save_csv(pd.DataFrame([t.__dict__ for t in res.trades]), args.trades_out)
        print(f"Trades written to {args.trades_out}")


def cmd_run(args, cfg):
    from .bot import TradingBot
    from .mt5_client import MT5Client
    if args.live:
        cfg.live = True
    client = MT5Client(cfg)
    client.connect()
    if cfg.live and client.account_mode == "REAL" and not args.i_understand_the_risk:
        print("Refusing to trade a REAL account without --i-understand-the-risk. "
              "Run on a demo account first.")
        client.shutdown()
        return
    try:
        TradingBot(cfg, client).run_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        client.shutdown()


def main(argv=None):
    p = argparse.ArgumentParser(prog="mt5_trader")
    p.add_argument("-c", "--config", default="config.yaml" if Path("config.yaml").exists() else None)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp):
        sp.add_argument("--csv", help="read bars from CSV instead of MT5")
        sp.add_argument("--synthetic", action="store_true", help="use random synthetic data (testing only)")
        sp.add_argument("--symbol")
        sp.add_argument("--bars", type=int)

    sp = sub.add_parser("download", help="download history from MT5 to CSV")
    data_args(sp)
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("train", help="train the ML filter")
    data_args(sp)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("backtest", help="backtest the strategy")
    data_args(sp)
    sp.add_argument("--spread", type=float, default=0.0, help="spread in price units, e.g. 0.0001")
    sp.add_argument("--equity", type=float, default=10_000.0)
    sp.add_argument("--train-frac", type=float, default=0.5)
    sp.add_argument("--no-ml", action="store_true")
    sp.add_argument("--trades-out")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("run", help="run the trading bot")
    sp.add_argument("--live", action="store_true", help="actually send orders")
    sp.add_argument("--i-understand-the-risk", action="store_true")
    sp.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
