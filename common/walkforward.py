"""Purged walk-forward split utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np

from .config import WalkforwardSettings


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray


class PurgedWalkForward:
    """Iterator over purged walk-forward splits."""

    def __init__(self, n_samples: int, settings: WalkforwardSettings):
        self.n_samples = n_samples
        self.settings = settings

    def __iter__(self) -> Iterator[Split]:
        wf = self.settings
        test_size = wf.test_size
        step_size = wf.step_size or test_size
        min_train = wf.min_train_size
        gap = wf.gap

        start = min_train
        splits: List[Split] = []
        count = 0
        while start + gap + test_size <= self.n_samples:
            train_end = start
            test_start = train_end + gap
            test_end = test_start + test_size
            train_idx = np.arange(train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append(Split(train_idx=train_idx, test_idx=test_idx))
            start += step_size
            count += 1
            if wf.max_splits and count >= wf.max_splits:
                break

        if not splits:
            raise ValueError("Unable to create walk-forward split; adjust configuration")

        return iter(splits)


__all__ = ["Split", "PurgedWalkForward"]
