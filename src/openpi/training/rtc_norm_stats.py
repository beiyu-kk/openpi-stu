"""Train-only normalization of local LeRobot v2.1 Piper delta action windows.

Read numeric Parquet columns only; video decoding cannot affect these statistics.
The window indexing is exactly action[t:t+H] relative to state[t]. Padding outside
an episode is excluded, matching the RTC loss mask. Delay sampling is applied
later, after normalization; it does not change this fixed coordinate transform.
"""

import hashlib
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize
from openpi.training import config as configs
from openpi.training.rtc_config import episode_split
from openpi.training.rtc_manifest import stats_digest
from openpi.transforms import DeltaActions


def delta_windows(state, actions, horizon):
    """Returns all valid [start,offset] delta targets, using the training transform."""
    state = np.asarray(state, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if state.ndim != 2 or state.shape != actions.shape or state.shape[1] != 7 or len(state) < 1:
        raise ValueError("Expected matching nonempty [frames,7] state and absolute actions")
    if horizon < 1 or not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Invalid horizon or nonfinite numeric data")
    indices = np.arange(len(state))[:, None] + np.arange(horizon)[None, :]
    valid = indices < len(state)
    chunk = actions[np.minimum(indices, len(state) - 1)].copy()
    delta = DeltaActions((True,) * 7)({"state": state, "actions": chunk})["actions"]
    return delta[valid]


def compute(config, *, output_dir=None, max_episodes=None):
    """A limited smoke run must write to a separate, explicit output directory."""
    if not isinstance(config.data, configs.LeRobotPiperDataConfig):
        raise ValueError("This numeric-only estimator supports the Piper RTC all-delta adapter")
    if config.data.base_config.rtc_split != "train":
        raise ValueError("Normalization must use the training episode split")
    if config.model.simulated_delay is None or not config.model.pi05:
        raise ValueError("Expected a pi0.5 RTC model config")
    if max_episodes is not None and (max_episodes <= 0 or output_dir is None):
        raise ValueError("Smoke computation requires max_episodes > 0 and explicit output_dir")
    if config.training_rtc.norm_source != "train_split":
        raise ValueError("Inherited checkpoint statistics must not be recomputed")
    root = pathlib.Path(config.data.dataset_root)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    if info["codebase_version"] != "v2.1" or info["fps"] <= 0:
        raise ValueError("Expected local LeRobot v2.1 metadata with positive fps")
    episodes_path = root / "meta/episodes.jsonl"
    episode_info = {row["episode_index"]: row for row in map(json.loads, episodes_path.read_text().splitlines())}
    if sorted(episode_info) != list(range(info["total_episodes"])):
        raise ValueError("Episode metadata is incomplete or non-contiguous")
    for name in ("observation.state", "action"):
        if info["features"][name]["shape"] != [7]:
            raise ValueError(f"{name} must be seven-dimensional")
    contract_path = root / "meta/capture_contract.json"
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else None
    if contract is not None:
        expected = {
            "action_representation": "absolute",
            "joint_unit": "radian",
            "gripper_unit": "meter",
            "openpi_action_delta_timestamps_start": 0,
        }
        if any(contract.get(k) != v for k, v in expected.items()):
            raise ValueError("Capture contract differs from Piper absolute/radian/meter/offset-zero assumptions")
    train_ids, val_ids = episode_split(info["total_episodes"], config.training_rtc)
    selected = train_ids if max_episodes is None else train_ids[:max_episodes]
    from openpi.training.rtc_data import default_norm_stats_dir

    production_target = (
        pathlib.Path(config.data.norm_stats_dir) if config.data.norm_stats_dir else default_norm_stats_dir(config)
    )
    target = pathlib.Path(output_dir) if output_dir is not None else production_target
    if max_episodes is not None and target.resolve() == production_target.resolve():
        raise ValueError("Smoke output cannot overwrite the configured production assets")
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    total_states, total_actions = 0, 0
    sources = []
    print(f"START stats: {len(selected)} train episodes, H={config.model.action_horizon}, output={target}", flush=True)
    for episode_id in selected:
        path = root / info["data_path"].format(
            episode_chunk=episode_id // info["chunks_size"], episode_index=episode_id
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        table = pq.read_table(
            path, columns=["observation.state", "action", "frame_index", "episode_index", "timestamp"]
        )
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        length = episode_info[episode_id]["length"]
        if len(state) != length or not np.array_equal(table["frame_index"].to_numpy(), np.arange(length)):
            raise ValueError(f"Unordered/missing frames in episode {episode_id}")
        if not np.all(table["episode_index"].to_numpy() == episode_id):
            raise ValueError(f"Wrong episode_index in {path}")
        timestamps = table["timestamp"].to_numpy()
        if not np.allclose(timestamps, np.arange(length) / info["fps"], atol=1e-4, rtol=0):
            raise ValueError(f"Irregular timestamps in episode {episode_id}; cannot assume frame offsets")
        delta = delta_windows(state, actions, config.model.action_horizon)
        stats["state"].update(state)
        stats["actions"].update(delta)
        total_states += length
        total_actions += len(delta)
        sources.append({"episode": episode_id, "frames": length, "valid_action_tokens": len(delta), "sha256": digest})
        print(f"episode={episode_id} states={length} valid_action_tokens={len(delta)}", flush=True)
    norm_stats = {key: accumulator.get_statistics() for key, accumulator in stats.items()}
    for name, values in norm_stats.items():
        for field in ("mean", "std", "q01", "q99"):
            if not np.isfinite(getattr(values, field)).all():
                raise ValueError(f"Nonfinite {name}/{field}")
        if np.any(values.q99 <= values.q01):
            raise ValueError(f"Degenerate quantile range in {name}; inspect dataset before normalizing")
    target.mkdir(parents=True, exist_ok=True)
    normalize.save(target, norm_stats)
    manifest = {
        "config_name": config.name,
        "repo_id": str(root.resolve()),
        "asset_id": config.data.assets.asset_id,
        "split_method": "explicit" if config.training_rtc.train_episodes is not None else "seeded_random",
        "split_seed": config.training_rtc.split_seed,
        "validation_fraction": config.training_rtc.validation_fraction,
        "train_episodes": list(train_ids),
        "validation_episodes": list(val_ids),
        "computed_episodes": list(selected),
        "smoke_only": max_episodes is not None,
        "state_count": total_states,
        "valid_action_token_count": total_actions,
        "action_horizon": config.model.action_horizon,
        "action_offsets": [0, config.model.action_horizon - 1],
        "delta_mask": [True] * 7,
        "delta_reference": "state at chunk observation/start frame",
        "exclude_episode_padding": True,
        "statistics": "OpenPI RunningStats mean/std/histogram q01/q99",
        "norm_stats_sha256": stats_digest(norm_stats),
        "capture_contract": contract,
        "metadata_sha256": {
            "info": hashlib.sha256(info_path.read_bytes()).hexdigest(),
            "episodes": hashlib.sha256(episodes_path.read_bytes()).hexdigest(),
        },
        "sources": sources,
    }
    (target / "norm_stats_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"DONE states={total_states} valid_action_tokens={total_actions}; saved {target / 'norm_stats.json'}",
        flush=True,
    )
    return manifest
