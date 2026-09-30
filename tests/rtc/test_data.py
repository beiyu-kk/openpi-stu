import dataclasses
import json
from types import SimpleNamespace

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.models.model import Observation
from openpi.shared import normalize
from openpi.training import config
from openpi.training import data_loader
from openpi.training import rtc_data
from openpi.training import rtc_manifest
from openpi.training import rtc_norm_stats
from openpi.training.rtc_config import TrainingRTCConfig
from openpi.training.rtc_config import episode_split


def test_padding_survives_the_piper_transform_chain(tmp_path, monkeypatch):
    class Tokenizer:
        def tokenize(self, *args, **kwargs):
            return np.ones(4, np.int32), np.ones(4, bool)

    monkeypatch.setattr(config._tokenizer, "PaligemmaTokenizer", lambda *_: Tokenizer())  # noqa: SLF001
    cfg = config.get_config("pi05_piper_full_finetune_rtc")
    data = dataclasses.replace(cfg.data, repo_id="test", norm_stats_dir=str(tmp_path)).create(
        cfg.assets_dirs, cfg.model
    )
    pad = np.arange(30) >= 3
    sample = {
        "observation.images.top_head": np.zeros((28, 28, 3), np.uint8),
        "observation.images.hand_right": np.zeros((28, 28, 3), np.uint8),
        "observation.state": np.arange(7, dtype=np.float32),
        "action": np.ones((30, 7), np.float32) * 10,
        "prompt": "task",
        "action_is_pad": pad,
    }
    result = data_loader.transform_dataset([sample], data, skip_norm_stats=True)[0]
    np.testing.assert_array_equal(result["action_is_pad"], pad)
    np.testing.assert_array_equal(result["actions"][0, :7], 10 - np.arange(7))
    assert result["actions"].shape == (30, 32)
    np.testing.assert_array_equal(
        Observation.from_dict(jax.tree.map(lambda x: np.asarray(x)[None], result)).action_is_pad[0], pad
    )


def test_dataset_and_metadata_receive_local_root_and_sparse_episode_ids(monkeypatch, tmp_path):
    calls = []

    def metadata(repo_id, root=None):
        calls.append((repo_id, root))
        return SimpleNamespace(total_episodes=4, fps=30, tasks={})

    def dataset(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return [kwargs]

    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", metadata)
    monkeypatch.setattr(data_loader, "RTCLeRobotDataset", dataset)
    rtc = TrainingRTCConfig(train_episodes=(0, 2), validation_episodes=(1, 3))
    data = config.DataConfig(
        repo_id="logical/id", dataset_root=str(tmp_path), training_rtc=rtc, action_sequence_keys=("action",)
    )
    cfg = config.get_config("pi05_piper_full_finetune_rtc")
    data_loader.create_torch_dataset(data, 30, cfg.model)
    assert calls[-1][1]["root"] == str(tmp_path)
    assert calls[-1][1]["episodes"] == [0, 2]
    assert calls[-1][1]["delta_timestamps"]["action"][0] == 0
    assert all(call[1] == str(tmp_path) for call in calls[:-1])
    assert episode_split(4, rtc) == ((0, 2), (1, 3))
    indices = rtc_data.validation_indices([4, 10], 6)
    assert len(indices) == len(set(indices)) == 6
    assert any(i < 4 for i in indices)
    assert any(i >= 4 for i in indices)


def numeric_dataset(root):
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    info = {
        "codebase_version": "v2.1",
        "fps": 10,
        "total_episodes": 4,
        "data_path": "data/episode_{episode_index:06d}.parquet",
        "chunks_size": 1000,
        "features": {k: {"shape": [7]} for k in ("observation.state", "action")},
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    episodes = [{"episode_index": i, "length": 5} for i in range(4)]
    (root / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, episodes)))
    for episode in range(4):
        state = np.arange(35).reshape(5, 7) * 0.01 + episode * 0.1
        action = state + np.arange(5)[:, None] * 0.03
        pq.write_table(
            pa.table(
                {
                    "observation.state": state.tolist(),
                    "action": action.tolist(),
                    "frame_index": np.arange(5),
                    "episode_index": np.full(5, episode),
                    "timestamp": np.arange(5) / 10,
                }
            ),
            root / f"data/episode_{episode:06d}.parquet",
        )


def test_numeric_statistics_use_only_training_windows_and_validate_provenance(tmp_path, monkeypatch):
    numeric_dataset(tmp_path / "data")
    rtc = TrainingRTCConfig(max_delay=2, train_episodes=(0, 2), validation_episodes=(1, 3))
    cfg = config.get_config("pi05_piper_lora_finetune_rtc")
    cfg = config.resolve_training_config(
        dataclasses.replace(
            cfg,
            model=dataclasses.replace(cfg.model, action_horizon=3, simulated_delay=3),
            training_rtc=rtc,
            data=dataclasses.replace(
                cfg.data, repo_id="test", dataset_root=str(tmp_path / "data"), norm_stats_dir=str(tmp_path / "stats")
            ),
        )
    )
    info = rtc_norm_stats.compute(cfg)
    assert info["state_count"] == 10
    assert info["valid_action_token_count"] == 24
    assert info["computed_episodes"] == [0, 2]
    stats = normalize.load(tmp_path / "stats")
    data = config.DataConfig(repo_id="test", dataset_root=str(tmp_path / "data"), norm_stats=stats)
    monkeypatch.setattr(
        data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda *a, **k: SimpleNamespace(total_episodes=4)
    )
    rtc_manifest.validate_norm_stats(cfg, data)
    changed = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, action_horizon=4))
    with pytest.raises(ValueError, match="does not match"):
        rtc_manifest.validate_norm_stats(changed, data)
    # Merely finding norm_stats.json is not enough to reuse untracked all-data statistics.
    (tmp_path / "stats/norm_stats_manifest.json").unlink()
    with pytest.raises(ValueError, match="provenance"):
        rtc_manifest.validate_norm_stats(cfg, data)
