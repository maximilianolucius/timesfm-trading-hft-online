#!/usr/bin/env python3
"""Export historical bars from MetaTrader5 to CSV."""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import pandas as pd

from common.logging_utils import get_logger
from common.mt5_utils import (
    MT5NotAvailableError,
    ensure_symbol,
    fetch_bars,
    initialize,
    shutdown,
    timeframe_to_mt5,
)

LOGGER = get_logger("mt5_export")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MT5 historical data to CSV")
    parser.add_argument("symbol", help="Symbol to export")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--login", type=int, help="Account login")
    parser.add_argument("--password", help="Account password")
    parser.add_argument("--server", help="Server name")
    parser.add_argument("--path", help="Terminal path")
    parser.add_argument("--output", default="data/mt5_export.csv")
    args = parser.parse_args()

    try:
        initialize(
            login=args.login,
            password=args.password,
            server=args.server,
            path=args.path,
        )
    except MT5NotAvailableError as exc:
        raise SystemExit(f"MetaTrader5 Python package not available: {exc}")

    ensure_symbol(args.symbol)
    timeframe = timeframe_to_mt5(args.timeframe)
    rates = fetch_bars(args.symbol, timeframe, args.count)
    shutdown()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"time": "timestamp", "tick_volume": "volume"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info("Exported %s rows to %s", len(df), output_path)


if __name__ == "__main__":
    main()
