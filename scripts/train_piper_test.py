import argparse
import dataclasses
import sys

import numpy as np
import pytest

from openpi import transforms
from openpi.training import config as _config
from scripts import compute_norm_stats
from scripts import train_piper


def test_resolve_base_params_path(tmp_path):
    params_dir = tmp_path / "pi05_base" / "params"
    params_dir.mkdir(parents=True)

    assert train_piper._resolve_base_params_path(str(params_dir.parent)) == str(params_dir)  # noqa: SLF001
    assert (
        train_piper._resolve_base_params_path("gs://bucket/pi05_base")  # noqa: SLF001
        == "gs://bucket/pi05_base/params"
    )


@pytest.mark.parametrize("config_name", ["pi05_piper_full_finetune", "pi05_piper_lora_finetune"])
@pytest.mark.parametrize("image_size", [None, 224, 336, 448])
def test_build_config_with_custom_paths(tmp_path, monkeypatch, config_name, image_size):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text("{}")
    (dataset_dir / "data").mkdir()
    params_dir = tmp_path / "pi05_base" / "params"
    params_dir.mkdir(parents=True)
    norm_stats_dir = tmp_path / "normalization"
    checkpoint_dir = tmp_path / "output"
    args = argparse.Namespace(
        config=config_name,
        image_size=image_size,
        dataset_dir=str(dataset_dir),
        dataset_repo_id="local/piper",
        base_model_dir=str(params_dir.parent),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        checkpoint_dir=str(checkpoint_dir),
        norm_stats_dir=str(norm_stats_dir),
        exp_name="test",
        overwrite=False,
        resume=False,
        batch_size=None,
        num_workers=None,
        num_train_steps=None,
        fsdp_devices=None,
        wandb_enabled=False,
    )

    config = train_piper._build_config(args)  # noqa: SLF001
    data_config = config.data.create_base_config(config.assets_dirs, config.model)
    base = _config.get_config(config_name)
    expected_size = image_size or 224

    assert config.name == base.name
    assert config.batch_size == (16 if "lora" in config_name else 32)
    assert config.model.image_resolution == (expected_size, expected_size)
    assert base.model.image_resolution == (224, 224)
    assert config.model.paligemma_variant == base.model.paligemma_variant
    assert config.freeze_filter == base.freeze_filter
    assert config.optimizer is base.optimizer
    if image_size is None:
        assert config.model is base.model
    assert config.weight_loader.resize_siglip_posemb
    assert config.weight_loader.params_path == str(params_dir)
    assert config.checkpoint_dir == checkpoint_dir
    assert data_config.dataset_root == str(dataset_dir)
    assert data_config.asset_id == "local/piper"
    assert compute_norm_stats.get_norm_stats_dir(config, data_config) == norm_stats_dir

    monkeypatch.setattr(_config._tokenizer, "PaligemmaTokenizer", lambda *_: None)  # noqa: SLF001
    group = _config.ModelTransformFactory()(config.model)
    resize = next(transform for transform in group.inputs if isinstance(transform, transforms.ResizeImages))
    image = np.full((423, 628, 3), 127, dtype=np.uint8)
    result = resize({"image": {"base_0_rgb": image}})
    assert result["image"]["base_0_rgb"].shape == (expected_size, expected_size, 3)


@pytest.mark.parametrize(
    ("provided_argument", "value", "missing_argument"),
    [
        ("--dataset-repo-id", "local/piper", "--dataset-dir"),
        ("--dataset-dir", "/tmp/piper", "--dataset-repo-id"),
    ],
)
def test_dataset_arguments_are_required(monkeypatch, capsys, provided_argument, value, missing_argument):
    monkeypatch.setattr(sys, "argv", ["train_piper.py", provided_argument, value, "--exp-name", "test"])

    with pytest.raises(SystemExit, match="2"):
        train_piper._parse_args()  # noqa: SLF001

    assert missing_argument in capsys.readouterr().err


@pytest.mark.parametrize("image_size", ["0", "-14", "225", "abc"])
def test_invalid_image_size_is_rejected(monkeypatch, capsys, image_size):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_piper.py",
            "--dataset-dir",
            "/tmp/piper",
            "--dataset-repo-id",
            "local/piper",
            "--exp-name",
            "test",
            "--image-size",
            image_size,
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        train_piper._parse_args()  # noqa: SLF001
    assert "--image-size" in capsys.readouterr().err


def test_launcher_uses_registered_train_config_defaults(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta/info.json").write_text("{}")
    (dataset_dir / "data").mkdir()
    base = _config.get_config("pi05_piper_lora_finetune")
    registered = dataclasses.replace(
        base,
        name="custom_piper",
        batch_size=8,
        num_train_steps=1234,
        checkpoint_base_dir=str(tmp_path / "custom_checkpoints"),
        checkpoint_dir_override=str(tmp_path / "custom_run"),
        weight_loader=dataclasses.replace(base.weight_loader, params_path="gs://bucket/custom/params"),
        data=dataclasses.replace(base.data, assets=_config.AssetsConfig(assets_dir="/custom/assets")),
    )
    monkeypatch.setitem(_config._CONFIGS_DICT, registered.name, registered)  # noqa: SLF001
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_piper.py",
            "--config",
            registered.name,
            "--dataset-dir",
            str(dataset_dir),
            "--dataset-repo-id",
            "local/piper",
            "--exp-name",
            "test",
        ],
    )

    config = train_piper._build_config(train_piper._parse_args())  # noqa: SLF001
    assert config.name == registered.name
    assert config.batch_size == registered.batch_size
    assert config.num_train_steps == registered.num_train_steps
    assert config.checkpoint_base_dir == registered.checkpoint_base_dir
    assert config.checkpoint_dir == registered.checkpoint_dir
    assert config.weight_loader is registered.weight_loader
    assert config.data.assets.assets_dir == registered.data.assets.assets_dir
    assert config.model is registered.model


@pytest.mark.parametrize("size", [224, 336, 448])
def test_launcher_preserves_resolution_edited_in_train_config(tmp_path, monkeypatch, size):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text("{}")
    (dataset_dir / "data").mkdir()
    base = _config.get_config("pi05_piper_full_finetune")
    configured = dataclasses.replace(base, model=dataclasses.replace(base.model, image_resolution=(size, size)))
    monkeypatch.setattr(train_piper._config, "get_config", lambda _: configured)  # noqa: SLF001
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_piper.py",
            "--dataset-dir",
            str(dataset_dir),
            "--dataset-repo-id",
            "local/piper",
            "--exp-name",
            "test",
            "--base-model-dir",
            "gs://bucket/pi05_base",
        ],
    )
    config = train_piper._build_config(train_piper._parse_args())  # noqa: SLF001
    assert config.model is configured.model
    assert config.model.image_resolution == (size, size)
    assert config.weight_loader.resize_siglip_posemb == configured.weight_loader.resize_siglip_posemb


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("image_size", [224, 336, 448])
def test_region_guidance_is_opt_in(tmp_path, monkeypatch, enabled, image_size):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta/info.json").write_text("{}")
    (dataset_dir / "data").mkdir()
    if enabled:
        (dataset_dir / "annotations").mkdir()
        (dataset_dir / "annotations/all_frames.parquet").touch()
    argv = [
        "train_piper.py",
        "--config",
        "pi05_piper_lora_finetune",
        "--image-size",
        str(image_size),
        "--dataset-dir",
        str(dataset_dir),
        "--dataset-repo-id",
        "local/book",
        "--exp-name",
        "test",
        "--base-model-dir",
        "gs://bucket/base",
    ]
    if enabled:
        argv += ["--region-guidance"]
    monkeypatch.setattr(sys, "argv", argv)
    base = _config.get_config("pi05_piper_lora_finetune")
    config = train_piper._build_config(train_piper._parse_args())  # noqa: SLF001
    assert config.model.image_resolution == (image_size, image_size)
    assert base.model.image_resolution == (224, 224)
    assert config.freeze_filter == base.freeze_filter
    assert config.data.base_config.prompt_from_task
    assert base.model.region_guidance is None
    assert base.data.base_config.region_annotations_dir is None
    if enabled:
        assert config.model.region_guidance.strength == 0.5
        assert config.data.base_config.region_annotations_dir == str(dataset_dir / "annotations")
    else:
        assert config.model.region_guidance is None
        assert config.data.base_config.region_annotations_dir is None
