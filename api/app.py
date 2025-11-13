"""FastAPI service exposing TimesFM forecasts."""
from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ProjectConfig, load_config
from common.data_io_csv import compute_atr
from common.fine_tune import build_model
from common.logging_utils import get_logger
from common.signal_engine import forecast_to_signal

LOGGER = get_logger("api")


@dataclass
class ModelBundle:
    config: ProjectConfig
    model: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray

    def normalise(self, window: np.ndarray) -> np.ndarray:
        return (window - self.mean) / np.where(self.std == 0, 1.0, self.std)


class PredictRequest(BaseModel):
    symbol: str
    timeframe: str
    horizon: int
    last_window: List[List[float]]


class PredictResponse(BaseModel):
    yhat: List[float]
    q10: Optional[List[float]] = None
    q50: Optional[List[float]] = None
    q90: Optional[List[float]] = None


class SignalResponse(BaseModel):
    side: str
    size: float
    sl: float
    tp: float
    confidence: float
    comment: str


def _load_stats(path: pathlib.Path) -> Dict[str, List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Stats file missing: {path}")
    return json.loads(path.read_text())


def _load_bundle(cfg: ProjectConfig, checkpoint: pathlib.Path) -> ModelBundle:
    stats_path = cfg.resolve_path(cfg.training.checkpoint_dir) / f"{cfg.task.name}_stats.json"
    stats = _load_stats(stats_path)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    model = build_model(
        cfg.training,
        cfg.task.context_len,
        cfg.task.horizon_len,
        feature_dim=len(cfg.task.features),
    )
    model.load_checkpoint(checkpoint)
    model.eval()
    return ModelBundle(config=cfg, model=model, mean=mean, std=std)


def create_app(cfg: ProjectConfig) -> FastAPI:
    if cfg.serve is None:
        raise ValueError("Serve configuration missing")

    app = FastAPI(title="TimesFM FT Service", version="0.1.0")
    bundles: Dict[str, ModelBundle] = {}

    for binding in cfg.serve.models:
        binding_cfg = cfg if binding.config_path is None else load_config(binding.config_path)
        key = _make_key(binding.symbol, binding.timeframe, binding.horizon_len)
        checkpoint = binding_cfg.resolve_path(binding.checkpoint)
        bundles[key] = _load_bundle(binding_cfg, checkpoint)
        LOGGER.info(
            "Model loaded", extra={"key": key, "checkpoint": str(checkpoint)}
        )

    app.state.bundles = bundles

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok", "models": str(list(app.state.bundles.keys()))}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        key = _make_key(request.symbol, request.timeframe, request.horizon)
        bundle = app.state.bundles.get(key)
        if bundle is None:
            raise HTTPException(status_code=404, detail=f"Model not found for key {key}")
        window = np.array(request.last_window, dtype=np.float32)
        if window.shape != (
            bundle.config.task.context_len,
            len(bundle.config.task.features),
        ):
            raise HTTPException(status_code=400, detail="Unexpected window shape")
        norm = bundle.normalise(window)
        tensor = torch.from_numpy(norm).unsqueeze(0).float()
        with torch.inference_mode():
            preds, quant = bundle.model.predict(tensor)
        preds_np = preds.squeeze(0).cpu().numpy().tolist()
        resp = PredictResponse(yhat=preds_np)
        if quant:
            q_sorted = sorted(quant.items())
            resp.q10 = q_sorted[0][1].squeeze(0).cpu().numpy().tolist()
            resp.q50 = q_sorted[len(q_sorted) // 2][1].squeeze(0).cpu().numpy().tolist()
            resp.q90 = q_sorted[-1][1].squeeze(0).cpu().numpy().tolist()
        return resp

    @app.post("/signal", response_model=SignalResponse)
    def signal(request: PredictRequest) -> SignalResponse:
        key = _make_key(request.symbol, request.timeframe, request.horizon)
        bundle = app.state.bundles.get(key)
        if bundle is None:
            raise HTTPException(status_code=404, detail=f"Model not found for key {key}")
        window = np.array(request.last_window, dtype=np.float32)
        if window.shape != (
            bundle.config.task.context_len,
            len(bundle.config.task.features),
        ):
            raise HTTPException(status_code=400, detail="Unexpected window shape")
        df_window = pd.DataFrame(window, columns=bundle.config.task.features)
        atr = compute_atr(df_window, period=bundle.config.trade.atr_lookback if bundle.config.trade else 14).iloc[-1]
        norm = bundle.normalise(window)
        tensor = torch.from_numpy(norm).unsqueeze(0).float()
        with torch.inference_mode():
            preds, quant = bundle.model.predict(tensor)
        preds_np = preds.squeeze(0).cpu().numpy()
        last_price = window[-1, bundle.config.task.features.index(bundle.config.task.target)]
        forecast_return = preds_np[-1] - last_price
        quantiles = None
        if quant:
            q_sorted = sorted(quant.items())
            quantiles = {
                0.1: float(q_sorted[0][1].squeeze(0)[-1]),
                0.5: float(q_sorted[len(q_sorted) // 2][1].squeeze(0)[-1]),
                0.9: float(q_sorted[-1][1].squeeze(0)[-1]),
            }
        signal = forecast_to_signal(
            price=float(last_price),
            forecast=float(forecast_return),
            quantiles=quantiles,
            atr=float(atr) if not np.isnan(atr) else None,
            edge_bps=bundle.config.backtest.edge_bps if bundle.config.backtest else 5.0,
            cost_bps=bundle.config.backtest.cost_bps if bundle.config.backtest else 5.0,
        )
        return SignalResponse(
            side=signal.side,
            size=signal.size,
            sl=signal.sl,
            tp=signal.tp,
            confidence=signal.confidence,
            comment=signal.comment,
        )

    return app


def _make_key(symbol: str, timeframe: str, horizon: int) -> str:
    return f"{symbol}:{timeframe}:{horizon}"


def launch(cfg: ProjectConfig, host: str, port: int) -> None:
    import uvicorn

    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port)


__all__ = ["create_app", "launch"]
