"""RTC dataset split and normalization asset selection."""

import dataclasses
import hashlib
import json
import pathlib

import numpy as np


def validation_indices(lengths, count):
    """Fixed round-robin episode/time coverage instead of the first video's first frames."""
    if not lengths or any(length <= 0 for length in lengths) or not 0 < count <= sum(lengths):
        raise ValueError("Invalid validation lengths or sample count")
    allocations = [0] * len(lengths)
    remaining = count
    while remaining:
        for episode, length in enumerate(lengths):
            if allocations[episode] < length and remaining:
                allocations[episode] += 1
                remaining -= 1
    offsets = np.cumsum([0, *lengths[:-1]])
    per_episode = [
        (offset + np.linspace(0, length - 1, num=n, dtype=np.int64)).tolist()
        for offset, length, n in zip(offsets, lengths, allocations, strict=True)
    ]
    return [values[i] for i in range(max(allocations)) for values in per_episode if i < len(values)]


def default_norm_stats_dir(config):
    rtc = config.training_rtc
    if rtc.norm_source == "checkpoint":
        root = pathlib.Path(config.weight_loader.params_path).expanduser().resolve().parent / "assets"
        asset = config.data.assets.asset_id or config.data.repo_id
        if (root / asset / "norm_stats.json").is_file():
            return root / asset
        candidates = sorted(root.rglob("norm_stats.json"))
        if len(candidates) != 1:
            raise ValueError("Cannot identify checkpoint norm stats; set --norm-stats-dir to its asset directory")
        return candidates[0].parent
    signature = {
        "horizon": config.model.action_horizon,
        "fraction": rtc.validation_fraction,
        "seed": rtc.split_seed,
        "train_episodes": rtc.train_episodes,
        "validation_episodes": rtc.validation_episodes,
        "delta_mask": [True] * 7,
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
    return pathlib.Path(config.data.dataset_root) / "rtc_norm_stats" / digest


def validation_config(config):
    data = dataclasses.replace(
        config.data,
        base_config=dataclasses.replace(config.data.base_config, rtc_split="validation", episodes=None),
    )
    return dataclasses.replace(config, data=data)
