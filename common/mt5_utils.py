"""MetaTrader5 helper functions."""
from __future__ import annotations

import datetime as dt
from typing import Iterable, Optional

from .logging_utils import get_logger

LOGGER = get_logger("mt5")


class MT5NotAvailableError(RuntimeError):
    pass


def _lazy_mt5():
    try:
        import MetaTrader5 as mt5  # type: ignore
    except ImportError as exc:
        raise MT5NotAvailableError(
            "MetaTrader5 package is required. Install via pip and run on Windows where MT5 terminal is present."
        ) from exc
    return mt5


def initialize(
    login: Optional[int] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
    path: Optional[str] = None,
    timeout: int = 10,
) -> None:
    mt5 = _lazy_mt5()
    if not mt5.initialize(
        login=login,
        password=password,
        server=server,
        path=path,
        timeout=timeout,
    ):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    LOGGER.info("MetaTrader5 initialized")


def shutdown() -> None:
    mt5 = _lazy_mt5()
    mt5.shutdown()
    LOGGER.info("MetaTrader5 shutdown")


def ensure_symbol(symbol: str) -> None:
    mt5 = _lazy_mt5()
    if mt5.symbol_select(symbol, True):
        return
    raise RuntimeError(f"Failed to enable symbol {symbol}")


def fetch_bars(symbol: str, timeframe: int, count: int) -> Iterable:
    mt5 = _lazy_mt5()
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        raise RuntimeError(f"Failed to fetch rates: {mt5.last_error()}")
    return rates


def place_order(
    symbol: str,
    side: str,
    volume: float,
    price: float,
    sl: float,
    tp: float,
    deviation: int,
    magic: int,
    comment: str = "TimesFM",
) -> int:
    mt5 = _lazy_mt5()
    order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"Order send failed: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"Order rejected: {result.retcode}")
    LOGGER.info("Order placed", extra={"ticket": result.order, "symbol": symbol, "side": side, "volume": volume})
    return result.order


def account_info():
    mt5 = _lazy_mt5()
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"Unable to pull account info: {mt5.last_error()}")
    return info


def symbol_info_tick(symbol: str):
    mt5 = _lazy_mt5()
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Unable to fetch tick for {symbol}: {mt5.last_error()}")
    return tick


def timeframe_to_mt5(timeframe: str) -> int:
    """Map timeframe string to MetaTrader5 constant."""

    mt5 = _lazy_mt5()
    attr = f'TIMEFRAME_{timeframe.upper()}'
    if not hasattr(mt5, attr):
        raise ValueError(f'Unsupported timeframe {timeframe}')
    return getattr(mt5, attr)


def utc_now() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)


__all__ = [
    "initialize",
    "shutdown",
    "ensure_symbol",
    "fetch_bars",
    "place_order",
    "account_info",
    "symbol_info_tick",
    "timeframe_to_mt5",
    "utc_now",
    "MT5NotAvailableError",
]
