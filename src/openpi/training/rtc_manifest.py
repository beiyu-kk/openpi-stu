"""Persist RTC semantics with each checkpoint and validate them before resuming."""

import dataclasses
import hashlib
import json
import pathlib

import numpy as np

from openpi.models.region_guidance import RegionGuidanceConfig
from openpi.training.rtc_config import TrainingRTCConfig
from openpi.training.rtc_config import episode_split


def stats_digest(stats):
    digest = hashlib.sha256()
    for name in sorted(stats):
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stats[name], field)
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                digest.update(f"{name}/{field}/{array.shape}".encode())
                digest.update(array.tobytes())
    return digest.hexdigest()


def json_values(value):
    return json.loads(json.dumps(value))


def source_digest():
    root = pathlib.Path(__file__).resolve().parents[3]
    files = [root / "scripts/train.py", root / "scripts/train_piper.py", root / "scripts/compute_norm_stats.py"]
    for folder in ("src/openpi/training", "src/openpi/models", "src/openpi/policies"):
        files.extend((root / folder).glob("*.py"))
    files.append(root / "src/openpi/transforms.py")
    digest = hashlib.sha256()
    for file in sorted(p for p in files if not p.name.endswith("_test.py") and p.name != "conftest.py"):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def model_values(values):
    values = dict(values)
    values["image_resolution"] = tuple(values["image_resolution"])
    if values.get("simulated_delay_weights") is not None:
        values["simulated_delay_weights"] = tuple(values["simulated_delay_weights"])
    if values.get("region_guidance") is not None:
        guidance = dict(values["region_guidance"])
        for name in ("layers", "heads"):
            if guidance.get(name) is not None:
                guidance[name] = tuple(guidance[name])
        values["region_guidance"] = RegionGuidanceConfig(**guidance)
    return values


def rtc_values(values):
    values = dict(values)
    for name in ("delay_weights", "train_episodes", "validation_episodes"):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    return TrainingRTCConfig(**values)


def validate_norm_stats(config, data_config):
    stats = data_config.norm_stats
    if stats is None or not {"state", "actions"}.issubset(stats):
        raise ValueError("RTC requires state and action normalization statistics")
    for name in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stats[name], field)
            if value is None or np.asarray(value).shape != (7,) or not np.isfinite(value).all():
                raise ValueError(f"RTC requires finite 7D {name}/{field} statistics")
    if config.training_rtc.norm_source == "checkpoint":
        # Inherited coordinates can include the held-out episodes of this adaptation.
        return
    path = pathlib.Path(config.data.norm_stats_dir) / "norm_stats_manifest.json"
    if not path.is_file():
        raise ValueError(f"RTC train-split statistics need provenance: {path}; compute RTC normalization first")
    info = json.loads(path.read_text())
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(data_config.repo_id, root=data_config.dataset_root)
    train, validation = episode_split(metadata.total_episodes, config.training_rtc)
    expected = {
        "train_episodes": list(train),
        "validation_episodes": list(validation),
        "computed_episodes": list(train),
        "action_horizon": config.model.action_horizon,
        "delta_mask": [True] * 7,
        "exclude_episode_padding": True,
        "norm_stats_sha256": stats_digest(stats),
        "smoke_only": False,
        "repo_id": str(pathlib.Path(data_config.dataset_root).resolve()),
    }
    if any(info.get(k) != v for k, v in expected.items()):
        raise ValueError("RTC normalization does not match this training split/horizon/dataset; recompute it")
    for name, file in (("info", "info.json"), ("episodes", "episodes.jsonl")):
        current = hashlib.sha256((pathlib.Path(data_config.dataset_root) / "meta" / file).read_bytes()).hexdigest()
        if info.get("metadata_sha256", {}).get(name) != current:
            raise ValueError("RTC normalization dataset metadata has changed; recompute it")


def build(config, data_config):
    if data_config.repo_id != "fake":
        validate_norm_stats(config, data_config)
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        metadata = LeRobotDatasetMetadata(data_config.repo_id, root=data_config.dataset_root)
        train, validation = episode_split(metadata.total_episodes, config.training_rtc)
        if data_config.episodes != train:
            raise ValueError("RTC training loader must use the configured training episodes")
    else:
        train, validation = (), ()
    return json_values(
        {
            "schema": 1,
            "source_sha256": source_digest(),
            "training_rtc": dataclasses.asdict(config.training_rtc),
            "model": dataclasses.asdict(config.model),
            "optimizer": {"type": type(config.optimizer).__name__, **dataclasses.asdict(config.optimizer)},
            "lr_schedule": {"type": type(config.lr_schedule).__name__, **dataclasses.asdict(config.lr_schedule)},
            "base_checkpoint": dataclasses.asdict(config.weight_loader),
            "batch_size": config.batch_size,
            "seed": config.seed,
            "num_train_steps": config.num_train_steps,
            "ema_decay": config.ema_decay,
            "repo_id": data_config.repo_id,
            "dataset_root": data_config.dataset_root,
            "asset_id": data_config.asset_id,
            "raw_action_dim": 7,
            "delta_mask": [True] * 7,
            "use_quantile_norm": data_config.use_quantile_norm,
            "norm_stats_sha256": stats_digest(data_config.norm_stats or {}),
            "train_episodes": train,
            "validation_episodes": validation,
            "freeze_strategy": "full" if config.training_rtc.finetune_mode == "full" else "action_lora_adarms_heads",
        }
    )


def prepare_resume(config):
    """Check mode before initialization, restoring only the auto-detected VLM variant."""
    path = config.checkpoint_dir / "training_rtc.json"
    if not config.resume or config.overwrite:
        return config
    if config.training_rtc is None:
        if path.exists():
            raise ValueError("This run used training RTC; resume with its RTC config")
        return config
    if not path.exists():
        # An interrupted new run may have created an empty directory, but a normal
        # training checkpoint must never be resumed under a different objective.
        if config.checkpoint_dir.exists() and any(p.name.isdigit() for p in config.checkpoint_dir.iterdir()):
            raise ValueError("Cannot resume an ordinary checkpoint as RTC; start a new experiment")
        return config
    previous = json.loads(path.read_text())
    if previous["training_rtc"] != json_values(dataclasses.asdict(config.training_rtc)):
        raise ValueError("RTC resume settings differ from the saved experiment")
    return dataclasses.replace(
        config, model=dataclasses.replace(config.model, paligemma_variant=previous["model"]["paligemma_variant"])
    )


def record(config, manifest, *, resuming):
    path = config.checkpoint_dir / "training_rtc.json"
    if resuming:
        if not path.exists() or json.loads(path.read_text()) != manifest:
            raise ValueError("RTC resume configuration, data, or normalization differs from the saved experiment")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def read(checkpoint_dir):
    path = pathlib.Path(checkpoint_dir) / "assets" / "rtc_manifest.json"
    if not path.is_file():
        raise ValueError(f"RTC checkpoint lacks its manifest: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported RTC manifest schema")
    return manifest


def restore_config(config, manifest, checkpoint_dir):
    from openpi.training import config as configs

    model = dataclasses.replace(config.model, **model_values(manifest["model"]))
    assets_dir = str(pathlib.Path(checkpoint_dir) / "assets")
    asset_id = manifest["asset_id"]
    data = dataclasses.replace(
        config.data,
        assets=configs.AssetsConfig(assets_dir=assets_dir, asset_id=asset_id),
        norm_stats_dir=str(pathlib.Path(assets_dir) / asset_id),
        base_config=dataclasses.replace(config.data.base_config or configs.DataConfig(), region_annotations_dir=None),
    )
    return configs.resolve_training_config(
        dataclasses.replace(config, model=model, data=data, training_rtc=rtc_values(manifest["training_rtc"]))
    )
