"""PyTorch fine-tuning utilities for TimesFM."""
from __future__ import annotations

import json
import math
import pathlib
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import TrainingSettings
from .logging_utils import get_logger
from .timesfm_adapter import TimesFMAdapter

LOGGER = get_logger("trainer")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TensorDataset(Dataset):
    contexts: torch.Tensor
    targets: torch.Tensor

    def __len__(self) -> int:
        return self.contexts.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.contexts[idx], self.targets[idx]


def make_dataloader(
    data: TensorDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def pinball_loss_tensor(targets: torch.Tensor, quantiles: Dict[float, torch.Tensor]) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for q, preds in quantiles.items():
        diff = targets - preds
        q_tensor = torch.tensor(q, device=targets.device, dtype=targets.dtype)
        loss = torch.mean(torch.maximum(q_tensor * diff, (q_tensor - 1.0) * diff))
        losses.append(loss)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=targets.device)


def compute_loss(mu: torch.Tensor, targets: torch.Tensor, quantiles: Optional[Dict[float, torch.Tensor]]) -> torch.Tensor:
    mse = F.mse_loss(mu, targets)
    if quantiles:
        return mse + pinball_loss_tensor(targets, quantiles)
    return mse


def build_model(
    config: TrainingSettings,
    context_len: int,
    horizon_len: int,
    feature_dim: int,
) -> TimesFMAdapter:
    return TimesFMAdapter(
        base_checkpoint=config.base_checkpoint,
        context_len=context_len,
        horizon_len=horizon_len,
        feature_dim=feature_dim,
        device=config.device,
        quantiles=config.quantiles,
    )


def prepare_optimizer(
    model: nn.Module,
    config: TrainingSettings,
) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        momentum = config.momentum or 0.9
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=momentum,
            weight_decay=config.weight_decay,
        )
    betas = config.betas
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=betas,
        weight_decay=config.weight_decay,
    )


def prepare_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingSettings,
    steps_per_epoch: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if config.scheduler is None:
        return None
    total_steps = steps_per_epoch * config.epochs
    if total_steps == 0:
        return None
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if config.scheduler == "linear":
        def lr_lambda(step: int) -> float:
            warmup = config.warmup_steps
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total_steps - warmup))
            return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.patience)
    raise ValueError(f"Unknown scheduler {config.scheduler}")


def train(
    config: TrainingSettings,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    context_len: int,
    horizon_len: int,
    feature_dim: int,
    workdir: pathlib.Path,
) -> pathlib.Path:
    set_seed(config.seed)

    model = build_model(config, context_len, horizon_len, feature_dim)
    optimizer = prepare_optimizer(model, config)

    train_loader = make_dataloader(
        train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    val_loader = make_dataloader(
        val_dataset, config.eval_batch(), shuffle=False, num_workers=config.num_workers
    )

    scheduler = prepare_scheduler(optimizer, config, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision and torch.cuda.is_available())

    best_loss = math.inf
    patience = config.patience
    best_path: Optional[pathlib.Path] = None
    history: List[Dict[str, float]] = []
    global_step = 0

    workdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_contexts, batch_targets in train_loader:
            batch_contexts = batch_contexts.to(model.device)
            batch_targets = batch_targets.to(model.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config.mixed_precision and torch.cuda.is_available()):
                mu, quant = model.forward(batch_contexts)
                loss = compute_loss(mu, batch_targets, quant)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler and not isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step()
            train_loss += float(loss.detach().cpu())
            global_step += 1
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch_contexts, batch_targets in val_loader:
                batch_contexts = batch_contexts.to(model.device)
                batch_targets = batch_targets.to(model.device)
                mu, quant = model.forward(batch_contexts)
                loss = compute_loss(mu, batch_targets, quant)
                val_loss += float(loss.detach().cpu())
        val_loss /= max(1, len(val_loader))

        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        LOGGER.info(
            "Epoch", extra={"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss - 1e-6:
            patience = config.patience
            best_loss = val_loss
            best_path = workdir / f"timesfm-ft-epoch{epoch}.pt"
            model.save_checkpoint(best_path)
        else:
            patience -= 1
            if patience <= 0:
                LOGGER.info("Early stopping triggered", extra={"epoch": epoch})
                break

        # Retain only latest k checkpoints
        checkpoints = sorted(workdir.glob("timesfm-ft-epoch*.pt"), key=lambda p: p.stat().st_mtime)
        for old_ckpt in checkpoints[:-config.save_best_k]:
            old_ckpt.unlink(missing_ok=True)

    if not best_path:
        raise RuntimeError("Training did not produce a checkpoint")

    (workdir / "history.json").write_text(json.dumps(history, indent=2))
    return best_path


__all__ = ["TensorDataset", "train", "build_model"]
