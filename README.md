# TimesFM Trading FT Online

Production-ready fine-tuning and trading toolkit for Google TimesFM with supervised training, purged walk-forward backtests, a FastAPI inference service, and a MetaTrader5 execution agent.

## Features

- Supervised fine-tuning of Google TimesFM base checkpoints with PyTorch
- Task-based configuration (symbol × timeframe × horizon)
- Purged walk-forward evaluation with RMSE/MAE, hit-rate, Sharpe/Sortino, coverage, and PIT metrics
- Quantile forecasts via native head or bootstrap ensemble fallback
- Typer CLI (`bin/tft`) for prep/train/backtest/serve/trade flows
- FastAPI service (`/predict`, `/signal`) for local inference
- MetaTrader5 execution agent with coverage and calendar kill-switches, plus dry-run mode
- Sample data tooling (Yahoo downloader, MT5 exporter)
- Tests, linting, typing, and Make targets for MLOps hygiene

## Requirements

- Python 3.10+
- PyTorch 2.1+ (CUDA recommended for training, CPU supported)
- TimesFM base checkpoint from the [official Google TimesFM repo](https://github.com/google-research/timesfm)
- (Optional) MetaTrader5 terminal on Windows for live trading

External TimesFM weights may carry non-commercial or research-only licenses — verify before production use.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# Optional extras for TimesFM + MetaTrader5
pip install -e .[quant]
```

Populate `.env` if you plan to run the API/EA:

```bash
cp .env.example .env
# edit MT5 credentials, API host/port, etc.
```

## Configuration

YAML configs live under `configs/`. Each file defines:

- `task`: symbol, timeframe, context length, horizon, feature order
- `data`: CSV paths, frequency, normalization rules
- `training`: optimizer, schedule, checkpoint/log directories
- `backtest`: purged walk-forward settings, costs, ATR window
- `serve`: API bindings and checkpoint(s)
- `trade`: REST endpoint, risk, kill-switch, symbol map
- `mt5`: login settings for live execution (set `enabled: true` only on Windows)

Example configs:

- `configs/xau_5m.yaml` – XAUUSD 5-minute horizon=12 (1 hour ahead)
- `configs/fx_1d.yaml` – EURUSD daily horizon=5 (extend/duplicate for GBPUSD, USDJPY, AUDUSD)
- `configs/rt_xau.yaml` – real-time deployment profile referencing XAUUSD checkpoint

## Data

CSV schema must be:

```
timestamp,open,high,low,close,volume
```

Timestamps must be UTC and gap-free per frequency. Utilities:

```bash
# Yahoo Finance sampler (delayed data, no credentials required)
scripts/fetch_yahoo_example.py GLD --interval 1h --output data/xauusd_m5.csv

# MetaTrader5 exporter (pulls bars from a running MT5 terminal)
scripts/export_mt5_bars.py XAUUSD --timeframe M5 --count 20000 --output data/xauusd_m5.csv

# MySQL exporter (dedupes each timeframe bucket, uses .env DB_* defaults)
python scripts/export_mysql_candles.py XAUUSDm M5 --overwrite --allow-gaps
```

`scripts/export_mysql_candles.py` reads database credentials from `.env` (`DB_HOST`, `DB_USER`, `DB_PASS`, `DB_NAME`, etc.), aligns timestamps to the requested timeframe, keeps the freshest bar per bucket, and writes straight into `data/<symbol>_<timeframe>.csv` unless an `--output` override is supplied. Pass `--start`/`--end` to slice the window or `--allow-gaps` if weekend holes are acceptable. `scripts/export_mt5_bars.py` instead talks to a local MetaTrader5 terminal via `common/mt5_utils.py`, so it’s ideal when you want broker-native history: supply login/server/path (or rely on `.env`), pick a timeframe/count, and it exports the canonical CSV schema.

## Workflow

```bash
# 1. Prep data -> normalized sliding windows + stats
bin/tft prep --config configs/xau_5m.yaml

# 2. Fine-tune TimesFM base for the task (checkpoints in workdir/)
bin/tft train --config configs/xau_5m.yaml

# 3. Purged walk-forward backtest (signals.csv, metrics.json, equity.png)
bin/tft backtest --config configs/xau_5m.yaml

# 4. Serve forecasts via FastAPI (loads fine-tuned checkpoints)
bin/tft serve --config configs/rt_xau.yaml

# 5. Real-time trading agent (dry-run first)
bin/tft trade --config configs/rt_xau.yaml --dry-run
# Remove --dry-run on Windows with MT5 running/logged in
```

### Training Notes

- `base_checkpoint` should point to the official TimesFM release (download and place under `checkpoints/`).
- Mixed precision enabled by default when CUDA is available.
- Quantile forecasts come from native TimesFM quantile head when present; otherwise a bootstrap ensemble is used.

### Backtesting & Evaluation

- Purged walk-forward splits prevent look-ahead leakage.
- Metrics saved under `workdir/backtests/<task>/` alongside equity curve and trade signals.
- Trading rule: go long/short when forecasted return exceeds ±edge after costs; returns are net of `cost_bps`.

### FastAPI Service

Endpoints:

- `GET /healthz` → service status
- `POST /predict` → `{symbol,timeframe,horizon,last_window}` → `{yhat,q10,q50,q90}`
- `POST /signal` → same payload, returns `{side,size,sl,tp,confidence}` (lots computed downstream)

Run locally:

```bash
bin/tft serve --config configs/rt_xau.yaml
# call with
curl -X POST http://127.0.0.1:8000/predict \\
  -H 'Content-Type: application/json' \\
  -d '{"symbol":"XAUUSD","timeframe":"M5","horizon":12,"last_window":[...]]}'
```

### MetaTrader5 EA

- Uses the Python `MetaTrader5` module to connect to a running terminal (Windows recommended).
- Handles symbol activation, risk sizing (risk % of equity), SL/TP placement, and kill-switch logic (calendar + coverage).
- Stores pending forecasts to monitor coverage consistency; if coverage falls below threshold, trading pauses.
- `--dry-run` mode pulls data from CSV and prints intended orders without MT5 access.

## Testing & Quality

```bash
pip install -r requirements.txt
pip install ruff mypy pytest
pytest
ruff check .
ruff format .
mypy .
```

Synthetic tests cover:

- Metric correctness (`tests/test_metrics.py`)
- FastAPI health/predict endpoints (`tests/test_api.py`)
- EA dry-run safety (`tests/test_ea_dryrun.py`)

## Licensing

Project code is licensed under Apache-2.0 (see `LICENSE`). External TimesFM checkpoints and MT5 data may be subject to additional licenses or broker terms — ensure compliance before deployment.

## Windows vs Linux Notes

- Fine-tuning, backtesting, and API serving run on Linux/macOS/Windows.
- MetaTrader5 Python binding requires Windows with the terminal installed and logged in.
- For Linux-based servers, deploy the API/backtests locally and run the EA on a Windows workstation or VPS.

## Repository Layout

```
bin/tft                 # Typer CLI entry point
common/                 # Shared modules (config, data, metrics, quantiles, mt5 helpers, etc.)
jobs/train.py           # Training loop
jobs/backtest.py        # Purged walk-forward backtest
api/app.py             # FastAPI service
rt/ea_mt5.py            # Real-time MetaTrader5 execution agent
scripts/                # Data acquisition utilities
configs/                # YAML task configs
workdir/                # Training artifacts (created at runtime)
```

Happy fine-tuning and safe trading!
