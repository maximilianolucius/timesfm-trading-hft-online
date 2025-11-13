"""MetaTrader5 execution agent for TimesFM forecasts."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ProjectConfig
from common.data_io_csv import load_csv
from common.logging_utils import get_logger
from common.mt5_utils import (
    MT5NotAvailableError,
    account_info,
    ensure_symbol,
    fetch_bars,
    initialize,
    place_order,
    shutdown,
    symbol_info_tick,
    timeframe_to_mt5,
    utc_now,
)
from common.risk_manager import RiskParams, calendar_blocked, compute_lot_size

LOGGER = get_logger("rt")


@dataclass
class PendingForecast:
    target_time: dt.datetime
    q_low: float
    q_high: float


class CoverageMonitor:
    def __init__(self, lookback: int, threshold: float):
        self.lookback = lookback
        self.threshold = threshold
        self.history: Deque[bool] = deque(maxlen=lookback)

    def update(self, covered: bool) -> None:
        self.history.append(covered)

    def blocked(self) -> bool:
        if len(self.history) < min(5, self.lookback):
            return False
        return float(np.mean(self.history)) < self.threshold


def _load_symbol_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def _map_symbol(symbol: str, mapping: Dict[str, str]) -> str:
    return mapping.get(symbol, symbol)


def _bars_to_frame(bars: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    return df.set_index("time")[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)


def _dry_run_iteration(cfg: ProjectConfig, session: requests.Session) -> None:
    df = load_csv(
        cfg.resolve_path(cfg.data.train_csv),
        frequency=cfg.data.frequency,
        timezone=cfg.data.timezone,
        limit=cfg.task.context_len + cfg.task.horizon_len,
        resample=cfg.data.resample,
        expect_no_gaps=False,
    )
    context = df.tail(cfg.task.context_len)
    payload = {
        "symbol": cfg.trade.symbol,
        "timeframe": cfg.trade.timeframe,
        "horizon": cfg.trade.horizon_len,
        "last_window": context[cfg.task.features].to_numpy().tolist(),
    }
    try:
        response = session.post(f"{cfg.trade.api_url}/signal", json=payload, timeout=10)
    except requests.RequestException as exc:
        LOGGER.error("Dry-run signal request error", extra={"error": str(exc)})
        return
    if response.status_code != 200:
        LOGGER.error("Dry-run signal failed", extra={"status": response.status_code, "body": response.text})
        return
    data = response.json()
    LOGGER.info("DRY-RUN", extra={"payload": data})


def run_trader(cfg: ProjectConfig, dry_run: bool = False) -> None:
    if cfg.trade is None:
        raise ValueError("trade block missing in config")

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    if dry_run:
        _dry_run_iteration(cfg, session)
        return

    if cfg.mt5 is None or not cfg.mt5.enabled:
        raise RuntimeError("MT5 credentials must be provided for live trading")

    mapping = _load_symbol_map(cfg.trade.symbol_map_path)
    symbol = cfg.trade.symbol
    mapped_symbol = _map_symbol(symbol, mapping)

    try:
        timeframe_const = timeframe_to_mt5(cfg.trade.timeframe)
        initialize(
            login=cfg.mt5.login,
            password=cfg.mt5.password,
            server=cfg.mt5.server,
            path=cfg.mt5.path,
            timeout=cfg.mt5.timeout,
        )
    except MT5NotAvailableError as exc:
        raise RuntimeError("MetaTrader5 Python package not available") from exc

    ensure_symbol(mapped_symbol)

    horizon_minutes = _timeframe_minutes(cfg.trade.timeframe) * cfg.trade.horizon_len
    horizon_delta = dt.timedelta(minutes=horizon_minutes)

    monitor = CoverageMonitor(
        lookback=cfg.trade.kill_switch.lookback,
        threshold=cfg.trade.kill_switch.coverage_min,
    )
    pending: Deque[PendingForecast] = deque()

    LOGGER.info("Live trader ready", extra={"symbol": mapped_symbol})

    try:
        while True:
            now = utc_now()
            if calendar_blocked(cfg.trade.kill_switch.calendar_blackout, now=now):
                LOGGER.warning("Calendar blackout")
                time.sleep(cfg.trade.poll_seconds)
                continue

            bars = fetch_bars(mapped_symbol, timeframe_const, cfg.task.context_len + cfg.task.horizon_len)
            df = _bars_to_frame(bars)
            context = df.tail(cfg.task.context_len)
            last_price = float(context[cfg.task.target].iloc[-1])

            while pending and pending[0].target_time <= df.index[-1]:
                forecast = pending.popleft()
                if forecast.target_time in df.index:
                    realized_close = float(df.loc[forecast.target_time]["close"])
                    monitor.update(forecast.q_low <= realized_close <= forecast.q_high)
                else:
                    pending.appendleft(forecast)
                    break

            if monitor.blocked():
                LOGGER.warning("Calibration guard active; staying flat")
                time.sleep(cfg.trade.poll_seconds)
                continue

            payload = {
                "symbol": cfg.trade.symbol,
                "timeframe": cfg.trade.timeframe,
                "horizon": cfg.trade.horizon_len,
                "last_window": context[cfg.task.features].to_numpy().tolist(),
            }
            try:
                response = session.post(f"{cfg.trade.api_url}/signal", json=payload, timeout=10)
            except requests.RequestException as exc:
                LOGGER.error("Signal request error", extra={"error": str(exc)})
                time.sleep(cfg.trade.poll_seconds)
                continue
            if response.status_code != 200:
                LOGGER.error("Signal request failed", extra={"status": response.status_code, "body": response.text})
                time.sleep(cfg.trade.poll_seconds)
                continue
            data = response.json()
            side = data["side"]
            if side == "FLAT":
                LOGGER.info("Flat signal", extra={"confidence": data.get('confidence')})
                time.sleep(cfg.trade.poll_seconds)
                continue

            info = account_info()
            equity = info.equity
            stop_price = float(data["sl"])
            tp_price = float(data["tp"])
            params = RiskParams(
                equity=equity,
                risk_percent=cfg.trade.risk_percent,
                atr=0.0,
                entry_price=last_price,
                stop_price=stop_price,
                min_lot=cfg.trade.size_min_lots,
                max_lot=cfg.trade.max_position_lots,
            )
            lots = compute_lot_size(params)
            tick = symbol_info_tick(mapped_symbol)
            price = tick.ask if side == "BUY" else tick.bid
            deviation = int(cfg.trade.cost_bps)
            try:
                place_order(
                    symbol=mapped_symbol,
                    side=side,
                    volume=lots,
                    price=price,
                    sl=stop_price,
                    tp=tp_price,
                    deviation=deviation,
                    magic=cfg.trade.magic_number,
                    comment="TimesFM-FT",
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.error("Order send failed", extra={"error": str(exc)})

            pending.append(
                PendingForecast(
                    target_time=df.index[-1] + horizon_delta,
                    q_low=float(data.get("sl", last_price)),
                    q_high=float(data.get("tp", last_price)),
                )
            )

            time.sleep(cfg.trade.poll_seconds)
    finally:
        shutdown()


def _timeframe_minutes(timeframe: str) -> int:
    tf = timeframe.upper()
    if tf.endswith('MIN') and tf[:-3].isdigit():
        return int(tf[:-3])
    mapping = {
        'M1': 1,
        'M5': 5,
        'M15': 15,
        'M30': 30,
        'H1': 60,
        'H4': 240,
        'D1': 1440,
    }
    if tf in mapping:
        return mapping[tf]
    raise ValueError(f"Unsupported timeframe {timeframe}")


__all__ = ['run_trader']
