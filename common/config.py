"""Configuration loading utilities for TimesFM fine-tuning workflows.

The project is intentionally scripts-first, so this module exposes helpers that
convert YAML configuration files and environment variables into strongly typed
pydantic models. Downstream scripts import the models to obtain a validated view
of settings such as data locations, model hyper-parameters, API bindings, and
MetaTrader5 credentials.
"""
from __future__ import annotations

import os
import pathlib
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, PositiveInt, validator

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = ROOT_DIR / ".env"


class PathsSettings(BaseModel):
    """Project-wide path settings."""

    data_dir: str = Field(default="data", description="Directory containing CSV inputs")
    workdir: str = Field(default="workdir", description="Directory for checkpoints and artifacts")
    logs_dir: str = Field(default="logs", description="Directory for log files")

    @validator("data_dir", "workdir", "logs_dir")
    def _expand(cls, value: str) -> str:  # type: ignore[override]
        return os.path.expanduser(value)


class TaskSettings(BaseModel):
    """Identifies the target forecasting task (symbol/timeframe/horizon)."""

    name: str = Field(..., description="Human readable identifier for the task")
    symbol: str = Field(..., description="Trading symbol, e.g. XAUUSD")
    timeframe: str = Field(..., description="Timeframe identifier, e.g. M5 or D1")
    context_len: PositiveInt = Field(..., description="Number of historical bars provided to the model")
    horizon_len: PositiveInt = Field(..., description="Number of steps ahead to forecast")
    target: str = Field(default="close", description="Which column to forecast")
    features: List[str] = Field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"],
        description="Ordered list of feature columns to feed into the model",
    )


class DataSettings(BaseModel):
    """Describes where data lives and how it should be interpreted."""

    train_csv: str = Field(..., description="Path to the training CSV data")
    val_csv: Optional[str] = Field(None, description="Optional validation CSV path")
    test_csv: Optional[str] = Field(None, description="Optional test CSV path")
    frequency: str = Field(..., description="Pandas offset alias representing the bar frequency")
    timezone: str = Field(default="UTC", description="Timezone of timestamps in the CSV")
    normalize: bool = Field(default=True, description="Whether to z-score normalize features")
    limit: Optional[int] = Field(
        default=None, description="Optional maximum number of recent rows to load"
    )
    expect_no_gaps: bool = Field(
        default=True,
        description="If True, raises when gaps are detected in the time series",
    )
    resample: Optional[str] = Field(
        default=None, description="Optional pandas offset alias to resample incoming CSV"
    )


class PrepSettings(BaseModel):
    """Controls train/validation splitting and feature engineering for prep."""

    validation_split: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Fraction of the dataset reserved for validation when val_csv absent",
    )
    shuffle_before_split: bool = Field(
        default=False,
        description="Whether to shuffle before splitting (default False for time-series)",
    )
    dropna: bool = Field(
        default=True, description="Whether to drop rows with missing values after alignment"
    )
    fillna_method: Optional[str] = Field(
        default="ffill",
        description="Optional pandas fillna method (None keeps missing values)",
    )


class TrainingSettings(BaseModel):
    """Hyper-parameters for supervised fine-tuning."""

    base_checkpoint: str = Field(..., description="Path or identifier of the base TimesFM checkpoint")
    finetune_checkpoint: Optional[str] = Field(
        default=None, description="Optional path to resume fine-tuning from"
    )
    timesfm_repo: Optional[str] = Field(
        default="https://github.com/google-research/timesfm",
        description="Reference to the official TimesFM repository for documentation",
    )
    device: str = Field(
        default="cuda",
        description="Torch device (cuda, cuda:0, or cpu) – callers can downshift if unavailable",
    )
    optimizer: str = Field(
        default="adamw",
        regex=r"^(adamw|sgd)$",
        description="Optimizer choice: adamw or sgd",
    )
    learning_rate: float = Field(default=5e-5, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    momentum: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Momentum for SGD (ignored for AdamW)"
    )
    betas: Tuple[float, float] = Field(
        default=(0.9, 0.999), description="Betas for AdamW optimizers"
    )
    epochs: PositiveInt = Field(default=30)
    batch_size: PositiveInt = Field(default=32)
    eval_batch_size: Optional[int] = Field(
        default=None,
        description="Optional override for evaluation batch size (defaults to batch_size)",
    )
    gradient_clip: float = Field(default=1.0, gt=0.0)
    scheduler: Optional[str] = Field(
        default="cosine",
        description="Learning rate schedule: cosine, linear, plateau, or None",
    )
    warmup_steps: int = Field(default=0, ge=0)
    patience: int = Field(default=5, ge=1, description="Early stopping patience on validation loss")
    mixed_precision: bool = Field(
        default=True,
        description="Enable torch.cuda.amp autocast when CUDA is available",
    )
    seed: int = Field(default=42)
    num_workers: int = Field(default=2, ge=0)
    save_best_k: int = Field(default=3, ge=1, description="Keep latest k checkpoints in workdir")
    quantiles: List[float] = Field(
        default_factory=lambda: [0.1, 0.5, 0.9],
        description="Quantiles to request from TimesFM when supported",
    )
    bootstrap_samples: int = Field(
        default=30,
        ge=1,
        description="Bootstrap ensemble size when quantile head unavailable",
    )
    bootstrap_ratio: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="Fraction of training data sampled with replacement for bootstrap members",
    )

    def eval_batch(self) -> int:
        return self.eval_batch_size or self.batch_size


class WalkforwardSettings(BaseModel):
    """Parameters for purged walk-forward evaluation."""

    n_splits: int = Field(default=5, ge=1)
    min_train_size: int = Field(default=1024, ge=32)
    test_size: int = Field(default=64, ge=1)
    step_size: Optional[int] = Field(
        default=None,
        description="Optional custom stride between fold start indices",
    )
    gap: int = Field(default=0, ge=0, description="Number of samples to purge between train and test")
    max_splits: Optional[int] = Field(default=None, ge=1)


class BacktestSettings(BaseModel):
    walkforward: WalkforwardSettings = Field(default_factory=WalkforwardSettings)
    cost_bps: float = Field(default=5.0, ge=0.0)
    edge_bps: float = Field(default=10.0, description="Edge threshold for trading rule")
    risk_per_trade: float = Field(default=1.0, ge=0.0)
    atr_lookback: int = Field(default=14, ge=1)
    output_dir: str = Field(default="workdir/backtests")
    coverage_threshold: float = Field(
        default=0.6,
        description="Minimum acceptable quantile coverage during evaluation",
    )


class ServeModelBinding(BaseModel):
    symbol: str
    timeframe: str
    horizon_len: PositiveInt
    checkpoint: str
    config_path: Optional[str] = Field(
        default=None, description="Optional config override for this binding"
    )
    quantiles: Optional[List[float]] = None


class ServeSettings(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=0, le=65535)
    log_level: str = Field(default="INFO")
    models: List[ServeModelBinding] = Field(default_factory=list)
    reload: bool = Field(default=False)


class TradeKillSwitch(BaseModel):
    coverage_min: float = Field(default=0.55, ge=0.0, le=1.0)
    lookback: int = Field(default=200, ge=1)
    calendar_blackout: List[str] = Field(
        default_factory=list,
        description="List of ISO-8601 datetimes or date prefixes to avoid trading",
    )


class TradeSettings(BaseModel):
    api_url: str = Field(default="http://127.0.0.1:8000")
    poll_seconds: int = Field(default=60, ge=5)
    risk_percent: float = Field(default=1.0, gt=0)
    atr_lookback: int = Field(default=14, ge=1)
    size_min_lots: float = Field(default=0.01, gt=0)
    slippage_factor: float = Field(default=1.0, ge=1.0)
    magic_number: int = Field(default=424242)
    symbol: str = Field(...)
    timeframe: str = Field(...)
    horizon_len: PositiveInt = Field(...)
    cost_bps: float = Field(default=5.0, ge=0.0)
    kill_switch: TradeKillSwitch = Field(default_factory=TradeKillSwitch)
    symbol_map_path: Optional[str] = Field(default=None)
    coverage_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_position_lots: Optional[float] = Field(default=None, gt=0)


class MT5Settings(BaseModel):
    enabled: bool = Field(default=False)
    login: Optional[int] = Field(default=None)
    server: Optional[str] = Field(default=None)
    password: Optional[str] = Field(default=None)
    path: Optional[str] = Field(default=None)
    timeout: int = Field(default=10, ge=1)


class ProjectConfig(BaseModel):
    """Full configuration tree for a task."""

    paths: PathsSettings = Field(default_factory=PathsSettings)
    task: TaskSettings
    data: DataSettings
    prep: PrepSettings = Field(default_factory=PrepSettings)
    training: TrainingSettings
    backtest: Optional[BacktestSettings] = None
    serve: Optional[ServeSettings] = None
    trade: Optional[TradeSettings] = None
    mt5: Optional[MT5Settings] = None

    def resolve_path(self, path: str) -> pathlib.Path:
        """Resolve a path relative to the project root."""

        candidate = pathlib.Path(os.path.expanduser(path))
        if candidate.is_absolute():
            return candidate
        return ROOT_DIR / candidate


def _read_yaml(path: pathlib.Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _merge_env(config: Dict) -> Dict:
    def _expand(node):
        if isinstance(node, dict):
            return {key: _expand(value) for key, value in node.items()}
        if isinstance(node, list):
            return [_expand(value) for value in node]
        if isinstance(node, str):
            return os.path.expandvars(node)
        return node

    return _expand(config)


@lru_cache(maxsize=8)
def load_env(path: pathlib.Path = DEFAULT_ENV_PATH) -> Dict[str, str]:
    """Load optional .env file into the environment and return its key-value map."""

    if not path.exists():
        return {}
    env_map: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            value = value.strip().strip('"')
            env_map[key] = value
            os.environ.setdefault(key, value)
    return env_map


def load_config(path: str, env_path: Optional[str] = None) -> ProjectConfig:
    """Load and validate a YAML configuration file."""

    config_path = pathlib.Path(path)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if env_path is None:
        env_path = str(DEFAULT_ENV_PATH)
    load_env(pathlib.Path(env_path))

    raw = _read_yaml(config_path)
    merged = _merge_env(raw)
    return ProjectConfig(**merged)


__all__ = [
    "ProjectConfig",
    "TaskSettings",
    "DataSettings",
    "TrainingSettings",
    "BacktestSettings",
    "ServeSettings",
    "TradeSettings",
    "MT5Settings",
    "load_config",
    "load_env",
]
