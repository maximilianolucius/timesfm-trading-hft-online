from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.app import create_app
from common.config import (
    BacktestSettings,
    DataSettings,
    ProjectConfig,
    ServeModelBinding,
    ServeSettings,
    TaskSettings,
    TradeKillSwitch,
    TradeSettings,
    TrainingSettings,
    WalkforwardSettings,
)
from common.fine_tune import build_model


def build_test_config(tmp_path: Path) -> ProjectConfig:
    workdir = tmp_path / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    task = TaskSettings(
        name="dummy",
        symbol="XAUUSD",
        timeframe="M5",
        context_len=16,
        horizon_len=4,
        target="close",
        features=["open", "high", "low", "close", "volume"],
    )
    data = DataSettings(
        train_csv=str(tmp_path / "dummy.csv"),
        frequency="5min",
        timezone="UTC",
        normalize=True,
        expect_no_gaps=False,
    )
    training = TrainingSettings(
        base_checkpoint="fallback",
        device="cpu",
        checkpoint_dir=str(workdir),
        log_dir=str(tmp_path / "logs"),
        epochs=1,
        batch_size=2,
        num_workers=0,
    )
    walkforward = WalkforwardSettings(n_splits=2, min_train_size=20, test_size=4, gap=0)
    backtest = BacktestSettings(walkforward=walkforward, cost_bps=5, edge_bps=8, output_dir=str(tmp_path / "backtests"))

    checkpoint_path = workdir / "timesfm-ft-test.pt"
    model = build_model(training, task.context_len, task.horizon_len, len(task.features))
    model.save_checkpoint(checkpoint_path)

    stats = {
        "mean": [0.0] * len(task.features),
        "std": [1.0] * len(task.features),
        "features": task.features,
    }
    (workdir / "dummy_stats.json").write_text(json.dumps(stats))

    serve = ServeSettings(
        host="127.0.0.1",
        port=8000,
        models=[
            ServeModelBinding(
                symbol=task.symbol,
                timeframe=task.timeframe,
                horizon_len=task.horizon_len,
                checkpoint=str(checkpoint_path),
                config_path=None,
            )
        ],
    )
    trade = TradeSettings(
        api_url="http://127.0.0.1:8000",
        poll_seconds=60,
        risk_percent=1.0,
        atr_lookback=14,
        size_min_lots=0.1,
        magic_number=1,
        symbol=task.symbol,
        timeframe=task.timeframe,
        horizon_len=task.horizon_len,
        cost_bps=5,
        kill_switch=TradeKillSwitch(coverage_min=0.5, lookback=10, calendar_blackout=[]),
        coverage_threshold=0.5,
    )

    return ProjectConfig(
        task=task,
        data=data,
        training=training,
        backtest=backtest,
        serve=serve,
        trade=trade,
    )


def test_api_predict(tmp_path):
    cfg = build_test_config(Path(tmp_path))

    app = create_app(cfg)
    client = TestClient(app)

    window = np.random.rand(cfg.task.context_len, len(cfg.task.features)).tolist()
    payload = {
        "symbol": cfg.task.symbol,
        "timeframe": cfg.task.timeframe,
        "horizon": cfg.task.horizon_len,
        "last_window": window,
    }

    health = client.get("/healthz")
    assert health.status_code == 200

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["yhat"]) == cfg.task.horizon_len

    signal = client.post("/signal", json=payload)
    assert signal.status_code == 200
    sig_body = signal.json()
    assert {"side", "sl", "tp", "confidence"}.issubset(sig_body.keys())
