#!/usr/bin/env python3
"""Fetch delayed market data from Yahoo Finance for backtesting."""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import pandas as pd
import yfinance as yf

SCHEMA = ["timestamp", "open", "high", "low", "close", "volume"]


def fetch(symbol: str, start: dt.datetime, end: dt.datetime, interval: str) -> pd.DataFrame:
    data = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    data = data.drop(columns=[col for col in data.columns if col not in {"open", "high", "low", "close", "volume"}])
    data = data.reset_index().rename(columns={"Datetime": "timestamp", "Date": "timestamp"})
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data[SCHEMA]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sample data for TimesFM training")
    parser.add_argument("symbol", help="Ticker symbol (e.g., GLD)")
    parser.add_argument("--start", default=(dt.datetime.utcnow() - dt.timedelta(days=365)).strftime("%Y-%m-%d"))
    parser.add_argument("--end", default=dt.datetime.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--interval", default="1h", help="Data interval (1m, 1h, 1d)")
    parser.add_argument("--output", default="data/yahoo_sample.csv", help="Output CSV path")
    args = parser.parse_args()

    start = dt.datetime.fromisoformat(args.start)
    end = dt.datetime.fromisoformat(args.end)
    df = fetch(args.symbol, start, end, args.interval)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
