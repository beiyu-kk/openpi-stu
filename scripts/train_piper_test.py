import argparse
import dataclasses
import sys

import pytest

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


@pytest.mark.parametrize(
    ("config_name", "expected_size"),
    [
        ("pi05_piper_full_finetune", 224),
        ("pi05_piper_lora_finetune", 224),
        ("pi05_piper_full_finetune_336", 336),
        ("pi05_piper_lora_finetune_336", 336),
        ("pi05_piper_full_finetune_448", 448),
        ("pi05_piper_lora_finetune_448", 448),
    ],
)
def test_build_config_with_custom_paths(tmp_path, config_name, expected_size):
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

    assert config.batch_size == 32
    assert config.model.image_resolution == (expected_size, expected_size)
    assert config.model is _config.get_config(config_name).model
    assert config.weight_loader.resize_siglip_posemb
    assert config.weight_loader.params_path == str(params_dir)
    assert config.checkpoint_dir == checkpoint_dir
    assert data_config.dataset_root == str(dataset_dir)
    assert data_config.asset_id == "local/piper"
    assert compute_norm_stats.get_norm_stats_dir(config, data_config) == norm_stats_dir


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


def test_image_size_is_not_a_launch_argument(monkeypatch, capsys):
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
            "448",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        train_piper._parse_args()  # noqa: SLF001
    assert "unrecognized arguments: --image-size 448" in capsys.readouterr().err


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
