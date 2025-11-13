"""Training job for TimesFM supervised fine-tuning."""
from __future__ import annotations

import io
import json
import pathlib
import sys
from typing import Tuple

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import ProjectConfig
from common.data_io_csv import build_sliding_windows, load_csv, split_train_validation, zscore_normalise
from common.fine_tune import TensorDataset, train
from common.logging_utils import get_logger

LOGGER = get_logger("job.train")


def _load_npz(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["contexts"], data["targets"], data["timestamps"]


def _maybe_prepare(cfg: ProjectConfig) -> Tuple[TensorDataset, TensorDataset]:
    workdir = cfg.resolve_path(cfg.training.checkpoint_dir)
    train_npz = workdir / f"{cfg.task.name}_train.npz"
    val_npz = workdir / f"{cfg.task.name}_val.npz"
    if train_npz.exists() and val_npz.exists():
        train_ctx, train_tgt, _ = _load_npz(train_npz)
        val_ctx, val_tgt, _ = _load_npz(val_npz)
    else:
        df = load_csv(
            cfg.resolve_path(cfg.data.train_csv),
            frequency=cfg.data.frequency,
            timezone=cfg.data.timezone,
            limit=cfg.data.limit,
            resample=cfg.data.resample,
            expect_no_gaps=cfg.data.expect_no_gaps,
        )
        if cfg.data.normalize:
            df_norm, _, _ = zscore_normalise(df, cfg.task.features)
        else:
            df_norm = df
        if cfg.data.val_csv:
            df_val = load_csv(
                cfg.resolve_path(cfg.data.val_csv),
                frequency=cfg.data.frequency,
                timezone=cfg.data.timezone,
                limit=cfg.data.limit,
                resample=cfg.data.resample,
                expect_no_gaps=cfg.data.expect_no_gaps,
            )
        else:
            df_norm, df_val = split_train_validation(
                df_norm,
                validation_split=cfg.prep.validation_split,
                shuffle=cfg.prep.shuffle_before_split,
                seed=cfg.training.seed,
            )
        train_ds_np = build_sliding_windows(
            df_norm,
            cfg.task.context_len,
            cfg.task.horizon_len,
            cfg.task.features,
            cfg.task.target,
        )
        val_ds_np = build_sliding_windows(
            df_val,
            cfg.task.context_len,
            cfg.task.horizon_len,
            cfg.task.features,
            cfg.task.target,
        )
        workdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(train_npz, **train_ds_np.__dict__)
        np.savez_compressed(val_npz, **val_ds_np.__dict__)
        train_ctx, train_tgt = train_ds_np.contexts, train_ds_np.targets
        val_ctx, val_tgt = val_ds_np.contexts, val_ds_np.targets

    train_dataset = TensorDataset(
        contexts=torch.from_numpy(train_ctx).float(),
        targets=torch.from_numpy(train_tgt).float(),
    )
    val_dataset = TensorDataset(
        contexts=torch.from_numpy(val_ctx).float(),
        targets=torch.from_numpy(val_tgt).float(),
    )
    return train_dataset, val_dataset


def run_training(cfg: ProjectConfig) -> pathlib.Path:
    train_dataset, val_dataset = _maybe_prepare(cfg)
    checkpoint_dir = cfg.resolve_path(cfg.training.checkpoint_dir)
    feature_dim = train_dataset.contexts.shape[-1]

    best_path = train(
        cfg.training,
        train_dataset,
        val_dataset,
        context_len=cfg.task.context_len,
        horizon_len=cfg.task.horizon_len,
        feature_dim=feature_dim,
        workdir=checkpoint_dir,
    )

    artifact = {
        "task": cfg.task.dict(),
        "training": cfg.training.dict(),
        "best_checkpoint": str(best_path),
    }
    (checkpoint_dir / f"{cfg.task.name}_artifact.json").write_text(json.dumps(artifact, indent=2))
    LOGGER.info("Training complete", extra={"checkpoint": str(best_path)})
    return best_path


__all__ = ["run_training"]
