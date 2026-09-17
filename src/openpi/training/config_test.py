import dataclasses

import numpy as np
import pytest
import tyro

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config as _config


def test_piper_finetune_configs():
    full = _config.get_config("pi05_piper_full_finetune")
    lora = _config.get_config("pi05_piper_lora_finetune")

    assert isinstance(full.model, pi0_config.Pi0Config)
    assert full.model.pi05
    assert full.model.action_horizon == 30
    assert full.batch_size == 32
    assert isinstance(full.data, _config.LeRobotPiperDataConfig)
    assert full.data.use_delta_actions
    assert full.data.repo_id is tyro.MISSING
    assert full.data.dataset_root is None
    assert full.data.norm_stats_dir is None
    assert full.data.assets.asset_id is None

    assert isinstance(lora.model, pi0_config.Pi0Config)
    assert lora.model.pi05
    assert lora.model.action_horizon == 30
    assert "lora" in lora.model.paligemma_variant
    assert "lora" in lora.model.action_expert_variant
    assert lora.batch_size == 16
    assert lora.ema_decay is None
    assert lora.data.repo_id is tyro.MISSING
    assert lora.data.dataset_root is None
    assert lora.data.norm_stats_dir is None
    assert lora.data.assets.asset_id is None


def test_checkpoint_dir_override(tmp_path):
    config = _config.get_config("pi05_piper_full_finetune")
    config = dataclasses.replace(config, exp_name="test", checkpoint_dir_override=str(tmp_path))
    assert config.checkpoint_dir == tmp_path


@pytest.mark.parametrize("name", ["pi05_piper_full_finetune", "pi05_piper_lora_finetune"])
@pytest.mark.parametrize("size", [224, 336, 448])
def test_piper_resolution_configs(name, size):
    base = _config.get_config(name)
    configured = _config.get_config(name if size == 224 else f"{name}_{size}")
    base_size = 448 if "lora" in name else 224
    assert base.model.image_resolution == (base_size, base_size)
    expected_size = base_size if size == 224 else size
    assert configured.model.image_resolution == (expected_size, expected_size)
    assert configured.weight_loader.resize_siglip_posemb
    assert configured.model.paligemma_variant == base.model.paligemma_variant
    assert configured.freeze_filter == base.freeze_filter


@pytest.mark.parametrize("pi05", [False, True])
@pytest.mark.parametrize("size", [224, 336, 448])
def test_model_resize_uses_config(monkeypatch, pi05, size):
    # Tokenization is unrelated to image resizing and would download tokenizer assets.
    monkeypatch.setattr(_config._tokenizer, "PaligemmaTokenizer", lambda *_: None)  # noqa: SLF001
    model = pi0_config.Pi0Config(pi05=pi05, image_resolution=(size, size))
    group = _config.ModelTransformFactory()(model)
    resize = next(transform for transform in group.inputs if isinstance(transform, transforms.ResizeImages))
    image = np.full((423, 628, 3), 127, dtype=np.uint8)
    result = resize({"image": {"base_0_rgb": image, "right_wrist_0_rgb": image}})
    assert all(value.shape == (size, size, 3) for value in result["image"].values())
