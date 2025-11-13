from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.config import (
    BacktestSettings,
    DataSettings,
    ProjectConfig,
    TaskSettings,
    TradeKillSwitch,
    TradeSettings,
    TrainingSettings,
)
from rt.ea_mt5 import run_trader


class DummyResponse:
    def __init__(self, payload: dict):
        self.status_code = 200
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def build_dryrun_config(tmp_path: Path) -> ProjectConfig:
    data_path = tmp_path / "data"
    data_path.mkdir()
    csv_path = data_path / "dry.csv"
    timestamps = pd.date_range("2024-01-01", periods=600, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.linspace(1900, 1950, len(timestamps)),
            "high": np.linspace(1901, 1951, len(timestamps)),
            "low": np.linspace(1899, 1949, len(timestamps)),
            "close": np.linspace(1900.5, 1950.5, len(timestamps)),
            "volume": np.random.randint(100, 200, len(timestamps)),
        }
    )
    df.to_csv(csv_path, index=False)

    task = TaskSettings(
        name="dry",
        symbol="XAUUSD",
        timeframe="M5",
        context_len=32,
        horizon_len=6,
        target="close",
        features=["open", "high", "low", "close", "volume"],
    )
    data = DataSettings(
        train_csv=str(csv_path),
        frequency="5min",
        timezone="UTC",
        normalize=True,
        expect_no_gaps=True,
    )
    training = TrainingSettings(
        base_checkpoint="fallback",
        device="cpu",
        checkpoint_dir=str(tmp_path / "workdir"),
        log_dir=str(tmp_path / "logs"),
        epochs=1,
        batch_size=4,
        num_workers=0,
    )
    backtest = BacktestSettings()
    trade = TradeSettings(
        api_url="http://127.0.0.1:9999",
        poll_seconds=1,
        risk_percent=1.0,
        atr_lookback=14,
        size_min_lots=0.1,
        magic_number=1,
        symbol=task.symbol,
        timeframe=task.timeframe,
        horizon_len=task.horizon_len,
        cost_bps=5,
        kill_switch=TradeKillSwitch(coverage_min=0.5, lookback=10, calendar_blackout=[]),
        symbol_map_path=None,
        coverage_threshold=0.6,
    )
    return ProjectConfig(
        task=task,
        data=data,
        training=training,
        backtest=backtest,
        trade=trade,
    )


@pytest.mark.parametrize("side", ["FLAT", "BUY"])
def test_dry_run(monkeypatch, tmp_path, side: str) -> None:
    cfg = build_dryrun_config(Path(tmp_path))

    def fake_post(self, url, json, timeout):  # type: ignore[override]
        payload = {
            "side": side,
            "sl": json["last_window"][-1][3] - 1.0,
            "tp": json["last_window"][-1][3] + 2.0,
            "confidence": 0.7,
        }
        return DummyResponse(payload)

    monkeypatch.setattr("requests.Session.post", fake_post)

    # run_trader should exit quickly in dry-run mode without requiring MT5
    run_trader(cfg, dry_run=True)
