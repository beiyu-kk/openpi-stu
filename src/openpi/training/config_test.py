import dataclasses

import numpy as np
import pytest
import tyro

from openpi import transforms
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.training import config as _config


@pytest.mark.parametrize("family", ["pi05", "pi0_fast"])
@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("rtc", [False, True])
def test_dynamic_square_configs_are_registered(family, mode, rtc):
    name = f"{family}_piper_{mode}_finetune{'_rtc' if rtc else ''}_pad_to_square"
    cfg = _config.get_config(name)
    assert cfg.model.pad_to_square is True
    assert cfg.model.image_resize_mode == "resize_with_pad"
    assert (cfg.training_rtc is not None) == rtc
    assert cfg.model.image_resolution == (224, 224)  # Reference grid, not input size.


@pytest.mark.parametrize("model_cls", [pi0_config.Pi0Config, pi0_fast.Pi0FASTConfig])
@pytest.mark.parametrize("fixed_mode", ["resize_with_pad", "pad_only"])
def test_model_pad_to_square_switch_controls_image_processing(monkeypatch, model_cls, fixed_mode):
    monkeypatch.setattr(_config._tokenizer, "PaligemmaTokenizer", lambda *_: None)  # noqa: SLF001
    monkeypatch.setattr(_config._tokenizer, "FASTTokenizer", lambda *_args, **_kwargs: object())  # noqa: SLF001
    fixed = model_cls(image_resolution=(42, 42), image_resize_mode=fixed_mode)
    assert fixed.pad_to_square is False
    dynamic = dataclasses.replace(fixed, pad_to_square=True)
    image = np.arange(17 * 29 * 3, dtype=np.uint8).reshape(17, 29, 3)
    for model_cfg, side in ((dynamic, 29), (dataclasses.replace(dynamic, pad_to_square=False), 42)):
        group = _config.ModelTransformFactory()(model_cfg)
        resize = next(transform for transform in group.inputs if isinstance(transform, transforms.ResizeImages))
        pixels = resize({"image": {"head": image}})["image"]["head"]
        assert pixels.shape == (side, side, 3)
        if model_cfg.pad_to_square:
            np.testing.assert_array_equal(pixels[6:23], image)
    with pytest.raises(ValueError, match="pad_to_square must be a bool"):
        dataclasses.replace(fixed, pad_to_square=1)
    with pytest.raises(ValueError, match="use pad_to_square=True"):
        dataclasses.replace(fixed, image_resize_mode="pad_to_square")


def test_piper_finetune_configs():
    full = _config.get_config("pi05_piper_full_finetune")
    full_448 = _config.get_config("pi05_piper_full_finetune_image448")
    full_640 = _config.get_config("pi05_piper_full_finetune_image640")
    full_640_pad_only = _config.get_config("pi05_piper_full_finetune_image640_pad_only")
    lora = _config.get_config("pi05_piper_lora_finetune")
    lora_448 = _config.get_config("pi05_piper_lora_finetune_image448")
    lora_640 = _config.get_config("pi05_piper_lora_finetune_image640")
    lora_640_pad_only = _config.get_config("pi05_piper_lora_finetune_image640_pad_only")

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
    assert full_448.model.image_resolution == (448, 448)
    assert full_448.weight_loader.resize_siglip_posemb
    assert full_640.model.image_resolution == (640, 640)
    assert full_640.weight_loader.resize_siglip_posemb
    assert full_640_pad_only.model.image_resolution == (640, 640)
    assert full_640_pad_only.model.image_resize_mode == "pad_only"

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
    assert lora_448.model.image_resolution == (448, 448)
    assert lora_448.freeze_filter == lora_448.model.get_freeze_filter()
    assert lora_448.weight_loader.resize_siglip_posemb
    assert lora_640.model.image_resolution == (640, 640)
    assert lora_640.freeze_filter == lora_640.model.get_freeze_filter()
    assert lora_640.weight_loader.resize_siglip_posemb
    assert lora_640_pad_only.model.image_resize_mode == "pad_only"

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
    assert attention_bias.model.region_guidance.attention_bias_scale == 0.5
    assert attention_bias.model.region_guidance.query_types == ("action", "image", "prompt")
    assert attention_bias.data.base_config.region_annotation_mode == "book"

    action_chunk_10 = _config.get_config("pi05_piper_ft_image448_aclen10_at3_s5")
    action_chunk_30 = _config.get_config("pi05_piper_ft_image448_at3_s5")
    assert action_chunk_10.model.pi05
    assert action_chunk_10.model.image_resolution == (448, 448)
    assert action_chunk_10.model.action_horizon == 10
    assert action_chunk_10.model.region_guidance == action_chunk_30.model.region_guidance
    assert action_chunk_10.training_rtc is None
    assert action_chunk_10.batch_size == action_chunk_30.batch_size
    assert action_chunk_10.num_train_steps == action_chunk_30.num_train_steps

    expected_delay_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 12.0, 6.0)
    for scale, scale_name in ((0.5, "0_5"), (2, "2"), (5, "5")):
        attention_bias_d9 = _config.get_config(f"pi05_piper_full_finetune_rtc_attention_bias_448_d9_scale_{scale_name}")
        assert attention_bias_d9.training_rtc is not None
        assert attention_bias_d9.training_rtc.max_delay == 9
        assert attention_bias_d9.training_rtc.delay_weights == expected_delay_weights
        assert attention_bias_d9.model.image_resolution == (448, 448)
        assert attention_bias_d9.model.simulated_delay == 10
        assert attention_bias_d9.model.simulated_delay_weights == expected_delay_weights
        assert attention_bias_d9.model.region_guidance is not None
        assert attention_bias_d9.model.region_guidance.attention_bias_scale == scale
        assert attention_bias_d9.model.region_guidance.query_types == ("action", "image", "prompt")
        assert attention_bias_d9.data.base_config.region_annotation_mode == "book"

    merged = _config.get_config("pi05_piper_full_finetune_rtc_attention_bias_448_d9_scale_0_5_merge2_KI")
    assert merged.model.image_resolution == (448, 448)
    assert merged.model.image_token_merger == "spatial_2x2"
    assert merged.model.image_token_merge_size == 2
    assert merged.model.image_token_camera_keys == ("base_0_rgb", "right_wrist_0_rgb")
    assert merged.model.region_guidance.query_types == ("action", "image", "prompt")
    assert merged.model.knowledge_insulation


def test_pi0_fast_piper_configs(monkeypatch, tmp_path):
    full = _config.get_config("pi0_fast_piper_full_finetune")
    full_448 = _config.get_config("pi0_fast_piper_full_finetune_image448")
    full_640 = _config.get_config("pi0_fast_piper_full_finetune_image640")
    full_640_pad_only = _config.get_config("pi0_fast_piper_full_finetune_image640_pad_only")
    lora = _config.get_config("pi0_fast_piper_lora_finetune")
    lora_448 = _config.get_config("pi0_fast_piper_lora_finetune_image448")
    lora_640 = _config.get_config("pi0_fast_piper_lora_finetune_image640")
    lora_640_pad_only = _config.get_config("pi0_fast_piper_lora_finetune_image640_pad_only")

    assert isinstance(full.model, pi0_fast.Pi0FASTConfig)
    assert full.model.action_dim == 7
    assert full.model.action_horizon == 30
    assert full.model.max_token_len == 250
    assert full.model.image_resolution == (224, 224)
    assert isinstance(full.data, _config.LeRobotPiperDataConfig)
    assert full.data.base_config.prompt_from_task
    assert isinstance(full.weight_loader, _config.weight_loaders.CheckpointWeightLoader)
    assert full.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi0_fast_base/params"
    assert full.weight_loader.resize_siglip_posemb
    assert full_448.model.image_resolution == (448, 448)
    assert full_448.weight_loader.resize_siglip_posemb
    assert full_448.model.paligemma_variant == full.model.paligemma_variant
    assert full_640.model.image_resolution == (640, 640)
    assert full_640.weight_loader.resize_siglip_posemb
    assert full_640_pad_only.model.image_resolution == (640, 640)
    assert full_640_pad_only.model.image_resize_mode == "pad_only"

    assert isinstance(lora.model, pi0_fast.Pi0FASTConfig)
    assert "lora" in lora.model.paligemma_variant
    assert lora.freeze_filter == lora.model.get_freeze_filter()
    assert lora.ema_decay is None
    assert lora.model.image_resolution == (224, 224)
    assert lora_448.model.image_resolution == (448, 448)
    assert lora_448.freeze_filter == lora_448.model.get_freeze_filter()
    assert lora_448.weight_loader.resize_siglip_posemb
    assert lora_640.model.image_resolution == (640, 640)
    assert lora_640.freeze_filter == lora_640.model.get_freeze_filter()
    assert lora_640.weight_loader.resize_siglip_posemb
    assert lora_640_pad_only.model.image_resize_mode == "pad_only"

    monkeypatch.setattr(_config._tokenizer, "FASTTokenizer", lambda *_args, **_kwargs: object())  # noqa: SLF001
    data = full.data.create(tmp_path, full.model)
    assert any(isinstance(transform, transforms.TokenizeFASTInputs) for transform in data.model_transforms.inputs)
    resize = next(
        transform for transform in data.model_transforms.inputs if isinstance(transform, transforms.ResizeImages)
    )
    image = np.full((423, 628, 3), 127, dtype=np.uint8)
    resized = resize({"image": {"base_0_rgb": image}})
    assert resized["image"]["base_0_rgb"].shape == (224, 224, 3)
    data_448 = full_448.data.create(tmp_path, full_448.model)
    resize_448 = next(
        transform for transform in data_448.model_transforms.inputs if isinstance(transform, transforms.ResizeImages)
    )
    resized_448 = resize_448({"image": {"base_0_rgb": image}})
    observation_spec_448, _ = full_448.model.inputs_spec()
    assert observation_spec_448.images["base_0_rgb"].shape[1:3] == (448, 448)
    assert resized_448["image"]["base_0_rgb"].shape == (448, 448, 3)
    data_640_pad_only = full_640_pad_only.data.create(tmp_path, full_640_pad_only.model)
    resize_640_pad_only = next(
        transform
        for transform in data_640_pad_only.model_transforms.inputs
        if isinstance(transform, transforms.ResizeImages)
    )
    assert resize_640_pad_only.mode == "pad_only"
    image_640x480 = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    padded_640 = resize_640_pad_only({"image": {"base_0_rgb": image_640x480}})["image"]["base_0_rgb"]
    assert np.array_equal(padded_640[80:560], image_640x480)
    assert np.all(padded_640[:80] == 0)
    assert np.all(padded_640[560:] == 0)
    assert isinstance(data.model_transforms.outputs[0], transforms.ExtractFASTActions)
    assert data.model_transforms.outputs[0].action_horizon == full.model.action_horizon


@pytest.mark.parametrize("size", [224, 336, 448, 640])
def test_pi0_fast_resolution_is_configurable(monkeypatch, tmp_path, size):
    monkeypatch.setattr(_config._tokenizer, "FASTTokenizer", lambda *_args, **_kwargs: object())  # noqa: SLF001
    model = pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=30, image_resolution=(size, size))
    observation_spec, _ = model.inputs_spec()
    assert observation_spec.images["base_0_rgb"].shape[1:3] == (size, size)

    data_config = _config.LeRobotPiperDataConfig(base_config=_config.DataConfig(prompt_from_task=True))
    data = data_config.create(tmp_path, model)
    resize = next(
        transform for transform in data.model_transforms.inputs if isinstance(transform, transforms.ResizeImages)
    )
    image = np.full((423, 628, 3), 127, dtype=np.uint8)
    resized = resize({"image": {"base_0_rgb": image}})
    assert resized["image"]["base_0_rgb"].shape == (size, size, 3)


@pytest.mark.parametrize("resolution", [(224, 336), (0, 224)])
def test_pi0_fast_rejects_invalid_resolution(resolution):
    with pytest.raises(ValueError, match="image_resolution"):
        pi0_fast.Pi0FASTConfig(image_resolution=resolution)


def test_pi0_fast_rejects_invalid_image_resize_mode():
    with pytest.raises(ValueError, match="image_resize_mode"):
        pi0_fast.Pi0FASTConfig(image_resize_mode="resize")


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
