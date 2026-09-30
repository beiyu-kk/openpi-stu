import dataclasses
import sys

import flax.nnx as nnx
import numpy as np
import orbax.checkpoint as ocp
import pytest

from openpi.training import config
from openpi.training import optimizer
from openpi.training import weight_loaders
from openpi.training.rtc_config import TrainingRTCConfig
from scripts import train_piper


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_launcher_selects_rtc_and_preserves_baseline(tmp_path, monkeypatch, mode):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text("{}")
    (dataset / "data").mkdir()
    base = config.get_config(f"pi05_piper_{mode}_finetune")
    name = f"pi05_piper_{mode}_finetune_rtc"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_piper.py",
            "--config",
            name,
            "--dataset-dir",
            str(dataset),
            "--dataset-repo-id",
            "local/piper",
            "--exp-name",
            "rtc",
            "--image-size",
            "336",
        ],
    )
    resolved = train_piper._build_config(train_piper._parse_args())  # noqa: SLF001
    assert base.training_rtc is None
    assert base.model.simulated_delay is None
    assert resolved.training_rtc.finetune_mode == mode
    assert resolved.model.simulated_delay == 5
    assert resolved.model.image_resolution == (336, 336)
    assert resolved.ema_decay is None
    assert isinstance(resolved.weight_loader, weight_loaders.RTCCheckpointWeightLoader)
    assert resolved.data.norm_stats_dir.startswith(str(dataset / "rtc_norm_stats"))
    if mode == "full":
        assert resolved.freeze_filter is nnx.Nothing
        assert type(resolved.optimizer) is optimizer.AdamW
        assert "lora" not in resolved.model.action_expert_variant
    else:
        assert resolved.model.paligemma_variant == "gemma_2b"
        assert isinstance(resolved.optimizer, optimizer.RTCAdamW)
    assert config.resolve_training_config(resolved).model == resolved.model


def test_overrides_rederive_optimizer_and_freeze_rules():
    original = config.get_config("pi05_piper_lora_finetune_rtc")
    full = config.resolve_training_config(
        dataclasses.replace(
            original,
            training_rtc=dataclasses.replace(original.training_rtc, finetune_mode="full"),
            freeze_filter=nnx.Everything,
        )
    )
    assert full.freeze_filter is nnx.Nothing
    assert type(full.optimizer) is optimizer.AdamW
    assert full.model.action_expert_variant == "gemma_300m"
    assert original.training_rtc.finetune_mode == "lora"
    with pytest.raises(ValueError, match="max_delay"):
        config.resolve_training_config(dataclasses.replace(original, training_rtc=TrainingRTCConfig(max_delay=30)))


@pytest.mark.parametrize(
    "key", ["PaliGemma/llm/layers/attn/q_einsum/lora_a", "PaliGemma/llm/layers/attn/q_einsum_1/lora_a"]
)
def test_checkpoint_inspection_preserves_adapters(tmp_path, key):
    from flax.traverse_util import unflatten_dict

    params = unflatten_dict({key: np.ones((2, 2), np.float32)}, sep="/")
    path = tmp_path / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, {"params": params})
    loader = weight_loaders.RTCCheckpointWeightLoader(str(path), resize_siglip_posemb=True)
    assert loader.parameter_keys() == {key}
    cfg = dataclasses.replace(config.get_config("pi05_piper_lora_finetune_rtc"), weight_loader=loader)
    prepared = config.prepare_rtc_base(cfg)
    expected = "gemma_2b_lora" if "_1/" not in key else "gemma_2b"
    assert prepared.model.paligemma_variant == expected
    with pytest.raises(ValueError, match="discard"):
        loader.load({"other": np.ones(2)})
    cfg = dataclasses.replace(config.get_config("pi05_piper_full_finetune_rtc"), weight_loader=loader)
    with pytest.raises(ValueError, match="discard"):
        config.prepare_rtc_base(cfg)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_delay": -1},
        {"max_delay": True},
        {"delay_weights": (1, 2)},
        {"delay_weights": (0,) * 5},
        {"delay_weights": (float("inf"),) * 5},
        {"validation_fraction": 0},
        {"inherited_lr_scale": 0},
        {"validation_batches": 0},
        {"train_episodes": (0,)},
    ],
)
def test_invalid_rtc_settings(kwargs):
    with pytest.raises(ValueError, match="RTC"):
        TrainingRTCConfig(**kwargs)
