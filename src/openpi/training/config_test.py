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
    assert full.model.image_resolution == (224, 224)
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
    assert lora.model.image_resolution == (224, 224)
    assert "lora" in lora.model.paligemma_variant
    assert "lora" in lora.model.action_expert_variant
    assert lora.batch_size == 16
    assert lora.ema_decay is None
    assert lora.data.repo_id is tyro.MISSING
    assert lora.data.dataset_root is None
    assert lora.data.norm_stats_dir is None
    assert lora.data.assets.asset_id is None

    full_rtc = _config.get_config("pi05_piper_full_finetune_rtc")
    assert full_rtc.model.image_resolution == (224, 224)
    assert full_rtc.training_rtc is not None
    assert full_rtc.training_rtc.max_delay == 4

    lora_rtc = _config.get_config("pi05_piper_lora_finetune_rtc")
    assert lora_rtc.model.image_resolution == (224, 224)
    assert lora_rtc.training_rtc is not None
    assert lora_rtc.training_rtc.max_delay == 4

    attention_bias = _config.get_config("pi05_piper_full_finetune_rtc_attention_bias")
    assert attention_bias.model.pi05
    assert attention_bias.training_rtc is not None
    assert attention_bias.training_rtc.finetune_mode == "full"
    assert attention_bias.training_rtc.max_delay == 4
    assert attention_bias.model.image_resolution == (224, 224)
    assert attention_bias.model.region_guidance is not None
    assert attention_bias.data.base_config.region_annotation_mode == "book"

    attention_bias_d9 = _config.get_config("pi05_piper_full_finetune_rtc_attention_bias_448_d9")
    assert attention_bias_d9.training_rtc is not None
    assert attention_bias_d9.training_rtc.max_delay == 9
    expected_delay_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 12.0, 6.0)
    assert attention_bias_d9.training_rtc.delay_weights == expected_delay_weights
    assert attention_bias_d9.model.image_resolution == (448, 448)
    assert attention_bias_d9.model.simulated_delay == 10
    assert attention_bias_d9.model.simulated_delay_weights == expected_delay_weights
    assert attention_bias_d9.model.region_guidance is not None
    assert attention_bias_d9.data.base_config.region_annotation_mode == "book"


def test_checkpoint_dir_override(tmp_path):
    config = _config.get_config("pi05_piper_full_finetune")
    config = dataclasses.replace(config, exp_name="test", checkpoint_dir_override=str(tmp_path))
    assert config.checkpoint_dir == tmp_path


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
