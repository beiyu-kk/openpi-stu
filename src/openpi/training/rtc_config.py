"""User-facing training RTC settings. None on TrainConfig disables RTC."""

import dataclasses
import math
from typing import Literal

import numpy as np


@dataclasses.dataclass(frozen=True)
class TrainingRTCConfig:
    # Inclusive: max_delay=4 trains prefixes of length 0, 1, 2, 3 and 4.
    max_delay: int = 4
    delay_weights: tuple[float, ...] | None = None
    loss_reduction: Literal["official", "per_element"] = "official"
    time_distribution: Literal["beta", "uniform"] = "beta"
    finetune_mode: Literal["lora", "full"] = "lora"
    inherited_lr_scale: float = 0.2
    validation_fraction: float = 0.1
    split_seed: int = 42
    train_episodes: tuple[int, ...] | None = None
    validation_episodes: tuple[int, ...] | None = None
    validation_interval: int = 1000
    validation_batches: int = 4
    norm_source: Literal["train_split", "checkpoint"] = "train_split"

    def __post_init__(self):
        if type(self.max_delay) is not int or self.max_delay < 0:
            raise ValueError("RTC max_delay must be a nonnegative integer")
        if self.delay_weights is not None:
            weights = self.delay_weights
            if (
                len(weights) != self.max_delay + 1
                or any(not math.isfinite(w) or w < 0 for w in weights)
                or not math.isfinite(sum(weights))
                or sum(weights) <= 0
            ):
                raise ValueError("RTC weights must be finite, nonnegative, have length max_delay+1 and positive sum")
        if self.loss_reduction not in ("official", "per_element"):
            raise ValueError("Unknown RTC loss reduction")
        if self.time_distribution not in ("beta", "uniform"):
            raise ValueError("Unknown RTC time distribution")
        if self.finetune_mode not in ("lora", "full"):
            raise ValueError("Unknown RTC finetune mode")
        if not 0 < self.inherited_lr_scale <= 1:
            raise ValueError("RTC inherited_lr_scale must be in (0,1]")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("RTC validation_fraction must be in (0,1)")
        if type(self.split_seed) is not int or self.split_seed < 0:
            raise ValueError("RTC split_seed must be a nonnegative integer")
        if type(self.validation_interval) is not int or self.validation_interval < 0:
            raise ValueError("RTC validation_interval must be a nonnegative integer")
        if type(self.validation_batches) is not int or self.validation_batches < 1:
            raise ValueError("RTC validation_batches must be a positive integer")
        if self.norm_source not in ("train_split", "checkpoint"):
            raise ValueError("Unknown RTC norm_source")
        if (self.train_episodes is None) != (self.validation_episodes is None):
            raise ValueError("Explicit RTC split requires both train and validation episode lists")


def episode_split(total: int, config: TrainingRTCConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Keep original episode IDs, including for sparse subsets and annotation lookup."""
    if config.train_episodes is not None:
        train, validation = config.train_episodes, config.validation_episodes
        for ids in (train, validation):
            if not ids or any(type(i) is not int or not 0 <= i < total for i in ids):
                raise ValueError("RTC episode IDs must be nonempty integer lists in dataset range")
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate RTC episode IDs")
        if set(train) & set(validation) or set(train) | set(validation) != set(range(total)):
            raise ValueError("RTC episode splits must be disjoint and cover the dataset")
        return tuple(sorted(train)), tuple(sorted(validation))
    if total < 2:
        raise ValueError("RTC requires at least two episodes for training and validation")
    ids = np.random.default_rng(config.split_seed).permutation(total)
    count = min(total - 1, max(1, math.ceil(total * config.validation_fraction)))
    return tuple(sorted(ids[count:].tolist())), tuple(sorted(ids[:count].tolist()))
