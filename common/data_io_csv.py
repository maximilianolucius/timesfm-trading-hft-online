"""CSV data ingestion and preparation utilities."""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """Holds sliding window tensors for model consumption."""

    contexts: np.ndarray
    targets: np.ndarray
    timestamps: np.ndarray

    def __len__(self) -> int:
        return len(self.contexts)


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def load_csv(
    path: pathlib.Path,
    frequency: str,
    timezone: str = "UTC",
    limit: Optional[int] = None,
    resample: Optional[str] = None,
    expect_no_gaps: bool = True,
) -> pd.DataFrame:
    """Load a time-series CSV with strict schema enforcement."""

    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df = _ensure_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if timezone.upper() != "UTC":
        df["timestamp"] = df["timestamp"].dt.tz_convert(timezone)
    df = df.set_index("timestamp").sort_index()
    if resample:
        df = df.resample(resample).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()

    if limit:
        df = df.tail(limit)

    if expect_no_gaps:
        _assert_no_gaps(df, frequency)

    return df


def _assert_no_gaps(df: pd.DataFrame, frequency: str) -> None:
    if df.empty:
        raise ValueError("Dataframe is empty")
    freq = pd.tseries.frequencies.to_offset(frequency)
    expected = pd.date_range(df.index[0], df.index[-1], freq=freq)
    if len(expected) != len(df.index):
        missing = expected.difference(df.index)
        if not missing.empty:
            raise ValueError(f"Gaps detected in dataset: {len(missing)} missing timestamps")


def split_train_validation(
    df: pd.DataFrame,
    validation_split: float,
    shuffle: bool = False,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1")

    n = len(df)
    val_size = max(1, int(math.ceil(n * validation_split)))
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed)
    train = df.iloc[:-val_size]
    val = df.iloc[-val_size:]
    if train.empty:
        raise ValueError("Training split is empty; reduce validation_split")
    return train, val


def zscore_normalise(df: pd.DataFrame, columns: Iterable[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return z-score normalsed dataframe along with mean and std for inverse transform."""

    stats_mean = df[columns].mean()
    stats_std = df[columns].std(ddof=0).replace(0, 1.0)
    normalised = df.copy()
    normalised[columns] = (df[columns] - stats_mean) / stats_std
    return normalised, stats_mean, stats_std


def build_sliding_windows(
    df: pd.DataFrame,
    context_len: int,
    horizon_len: int,
    feature_cols: Iterable[str],
    target_col: str,
    stride: int = 1,
) -> Dataset:
    """Construct sliding windows for supervised forecasting."""

    values = df[feature_cols].to_numpy(dtype=np.float32)
    target = df[target_col].to_numpy(dtype=np.float32)
    timestamps = df.index.to_numpy()

    contexts = []
    targets = []
    ts_out = []

    max_start = len(df) - (context_len + horizon_len)
    if max_start <= 0:
        raise ValueError("Dataframe too short for the requested context/horizon")

    for start in range(0, max_start + 1, stride):
        x_slice = values[start : start + context_len]
        y_slice = target[start + context_len : start + context_len + horizon_len]
        contexts.append(x_slice)
        targets.append(y_slice)
        ts_out.append(timestamps[start + context_len + horizon_len - 1])

    return Dataset(
        contexts=np.stack(contexts, axis=0),
        targets=np.stack(targets, axis=0),
        timestamps=np.array(ts_out),
    )


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - close).abs(),
            (low - close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


__all__ = [
    "Dataset",
    "load_csv",
    "split_train_validation",
    "zscore_normalise",
    "build_sliding_windows",
    "compute_atr",
]
