"""Thin wrapper around Google TimesFM with a PyTorch fine-tuning surface."""
from __future__ import annotations

import json
import pathlib
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .logging_utils import get_logger
from .quantiles import ensure_monotonic

LOGGER = get_logger("timesfm")


def _torch_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable, defaulting to CPU")
        return torch.device("cpu")
    return torch.device(device)


class _FallbackTimesFM(nn.Module):
    """Lightweight GRU fallback used for local tests when TimesFM is absent."""

    def __init__(self, feature_dim: int, horizon_len: int, quantiles: Iterable[float]):
        super().__init__()
        self.horizon_len = horizon_len
        self.quantiles = list(quantiles)
        hidden = 256
        self.encoder = nn.GRU(feature_dim, hidden, batch_first=True)
        self.mu_head = nn.Linear(hidden, horizon_len)
        if self.quantiles:
            self.quantile_head = nn.Linear(hidden, horizon_len * len(self.quantiles))
        else:
            self.quantile_head = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict[float, torch.Tensor]]]:
        out, _ = self.encoder(x)
        last = out[:, -1, :]
        mu = self.mu_head(last)
        quantiles = None
        if self.quantile_head is not None:
            q_raw = self.quantile_head(last).view(len(x), len(self.quantiles), self.horizon_len)
            q_sorted = torch.cumsum(F.softplus(q_raw), dim=1)
            quantiles = {
                float(level): q_sorted[:, idx, :]
                for idx, level in enumerate(sorted(self.quantiles))
            }
        return mu, quantiles


class TimesFMAdapter(nn.Module):
    """Abstraction that hides the difference between official TimesFM and fallback."""

    def __init__(
        self,
        base_checkpoint: str,
        context_len: int,
        horizon_len: int,
        feature_dim: int,
        device: str,
        quantiles: Iterable[float],
    ) -> None:
        super().__init__()
        self.device_name = device
        self.device = _torch_device(device)
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.feature_dim = feature_dim
        self.quantiles = list(quantiles)
        self.backend = "fallback"
        self.model = self._load_timesfm(base_checkpoint)
        self.to(self.device)

    def _load_timesfm(self, checkpoint: str) -> nn.Module:
        try:
            from timesfm import TimesFM  # type: ignore

            LOGGER.info("Loading TimesFM from pretrained checkpoint", extra={"checkpoint": checkpoint})
            model = TimesFM.from_pretrained(checkpoint)
            if hasattr(model, "requires_grad_"):
                model.requires_grad_(True)
            self.backend = "timesfm"
            return model
        except Exception as exc:  # pragma: no cover - exercised when package missing
            LOGGER.warning(
                "Falling back to lightweight GRU surrogate (install official TimesFM for production)",
                extra={"error": str(exc)},
            )
            self.backend = "fallback"
            return _FallbackTimesFM(self.feature_dim, self.horizon_len, self.quantiles)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict[float, torch.Tensor]]]:
        if self.backend == "timesfm":  # pragma: no cover - depends on external package
            outputs = self.model(
                x.to(self.device),
                prediction_horizon=self.horizon_len,
                output_quantiles=self.quantiles,
                return_dict=True,
            )
            mean = outputs.get("mean") or outputs.get("pred")
            quant = outputs.get("quantiles")
            if isinstance(quant, dict):
                quant_dict = {float(k): torch.as_tensor(v).to(self.device) for k, v in quant.items()}
            elif quant is None:
                quant_dict = None
            else:
                # assume array shaped (batch, quantiles, horizon)
                quant_levels = outputs.get("quantile_levels", self.quantiles)
                quant_tensor = torch.as_tensor(quant).to(self.device)
                quant_dict = {
                    float(level): quant_tensor[:, idx, :]
                    for idx, level in enumerate(quant_levels)
                }
            return torch.as_tensor(mean).to(self.device), quant_dict
        mean, quant = self.model(x.to(self.device))
        return mean, quant

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict[float, torch.Tensor]]]:
        self.eval()
        with torch.inference_mode():
            mu, quant = self.forward(x)
        if quant:
            numpy_dict = {
                level: tensor.detach().cpu().numpy()
                for level, tensor in quant.items()
            }
            ordered = ensure_monotonic(self.quantiles, np.stack(list(numpy_dict.values()), axis=1))  # type: ignore
            quant = {
                level: torch.from_numpy(values).to(self.device)
                for level, values in ordered.items()
            }
        return mu, quant

    def save_checkpoint(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.backend == "timesfm":  # pragma: no cover - depends on external package
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(str(path))
                return
        torch.save({"state_dict": self.state_dict(), "backend": self.backend}, path)

    def load_checkpoint(self, path: pathlib.Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location=self.device)
        backend = payload.get("backend")
        if backend and backend != self.backend:
            LOGGER.warning("Checkpoint backend %s differs from current %s", backend, self.backend)
        state_dict = payload.get("state_dict") or payload
        self.load_state_dict(state_dict, strict=False)


__all__ = ["TimesFMAdapter"]
