import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import siglip
from openpi.models.region_guidance import RegionGuidanceConfig
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import region_annotations
from openpi.training import weight_loaders
from scripts import train


def test_resume_preserves_guidance_schedule_and_method(tmp_path):
    base = _config.get_config("pi05_piper_lora_finetune")
    config = dataclasses.replace(
        base,
        checkpoint_dir_override=str(tmp_path),
        exp_name="region_test",
        num_train_steps=10000,
        model=dataclasses.replace(base.model, region_guidance=RegionGuidanceConfig()),
        data=dataclasses.replace(
            base.data,
            dataset_root="/dataset",
            base_config=dataclasses.replace(base.data.base_config, region_annotations_dir="/dataset/annotations"),
        ),
    )
    train._record_region_guidance(config, resuming=False)  # noqa: SLF001
    train._record_region_guidance(config, resuming=True)  # noqa: SLF001
    with pytest.raises(ValueError, match="resume settings differ"):
        train._record_region_guidance(dataclasses.replace(config, num_train_steps=20000), resuming=True)  # noqa: SLF001
    with pytest.raises(ValueError, match="same guidance options"):
        train._record_region_guidance(dataclasses.replace(config, model=base.model), resuming=True)  # noqa: SLF001


def test_baseline_does_not_create_guidance_metadata(tmp_path):
    config = dataclasses.replace(_config.get_config("pi05_piper_lora_finetune"), checkpoint_dir_override=str(tmp_path))
    train._record_region_guidance(config, resuming=False)  # noqa: SLF001
    train._record_region_guidance(config, resuming=True)  # noqa: SLF001
    assert not (tmp_path / "region_guidance.json").exists()


@pytest.fixture
def fresh_cpu_compilation_cache():
    # JAX 0.5.3 cached CPU executables can crash on the first donated update after
    # Orbax restore. Recompile this small integration test; keep GPU settings intact.
    original = jax.config.jax_enable_compilation_cache
    if jax.default_backend() == "cpu":
        jax.config.update("jax_enable_compilation_cache", False)  # noqa: FBT003
    try:
        yield
    finally:
        jax.config.update("jax_enable_compilation_cache", original)


def test_training_checkpoint_resume_and_original_policy_loading(tmp_path, monkeypatch, fresh_cpu_compilation_cache):
    original_siglip = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda *a, **k: original_siglip(*a, **{**k, "variant": "mu/14"}))
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda _: gemma.Config(width=32, depth=2, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16),
    )
    base = _config.get_config("pi05_piper_full_finetune")
    model_config = pi0_config.Pi0Config(
        pi05=True,
        image_resolution=(28, 28),
        action_horizon=2,
        max_token_len=4,
        dtype="float32",
        region_guidance=RegionGuidanceConfig(),
    )
    config = dataclasses.replace(
        base,
        model=model_config,
        batch_size=1,
        num_train_steps=4,
        num_workers=0,
        log_interval=1,
        save_interval=1,
        checkpoint_dir_override=str(tmp_path / "run"),
        exp_name="test",
        wandb_enabled=False,
        ema_decay=None,
        weight_loader=weight_loaders.NoOpWeightLoader(),
        data=dataclasses.replace(
            base.data,
            dataset_root=str(tmp_path),
            base_config=dataclasses.replace(
                base.data.base_config, region_annotations_dir=str(tmp_path / "annotations")
            ),
        ),
    )
    obs = model_config.fake_obs()
    obs = dataclasses.replace(
        obs,
        region_masks={k: jnp.ones((1, 28, 28)) for k in obs.images},
        region_valid={k: jnp.ones(1, bool) for k in obs.images},
    )
    batch = (obs, model_config.fake_act())

    class Loader:
        def __iter__(self):
            while True:
                yield batch

        def data_config(self):
            return _config.DataConfig()

    monkeypatch.setattr(train._data_loader, "create_data_loader", lambda *a, **k: Loader())  # noqa: SLF001
    monkeypatch.setattr(region_annotations, "validate_annotation_source", lambda *a: 0)
    monkeypatch.setattr(train, "init_wandb", lambda *a, **k: None)
    metrics = []
    monkeypatch.setattr(train.wandb, "log", lambda data, **k: metrics.append(data))
    original_save = train._checkpoints.save_state  # noqa: SLF001

    def save_then_interrupt(manager, state, loader, step):
        original_save(manager, state, loader, step)
        if step == 1:
            manager.wait_until_finished()
            manager.close()
            raise InterruptedError("simulated restart")

    monkeypatch.setattr(train._checkpoints, "save_state", save_then_interrupt)  # noqa: SLF001
    with pytest.raises(InterruptedError, match="simulated restart"):
        train.main(config)
    monkeypatch.setattr(train._checkpoints, "save_state", original_save)  # noqa: SLF001
    metrics.clear()
    train.main(dataclasses.replace(config, resume=True))
    strengths = [float(m["region/strength"]) for m in metrics if "region/strength" in m]
    np.testing.assert_allclose(strengths, [0.25, 0.0], atol=1e-7)
    original_config = dataclasses.replace(model_config, region_guidance=None)
    restored = original_config.load(_model.restore_params(tmp_path / "run/3/params", dtype=jnp.float32))
    params = nnx.state(restored).to_pure_dict()
    assert set(params) == set(nnx.state(nnx.eval_shape(original_config.create, jax.random.key(0))).to_pure_dict())
    prediction = nnx_utils.module_jit(restored.sample_actions)(
        jax.random.key(0), original_config.fake_obs(), num_steps=1
    )
    assert prediction.shape == (1, 2, 32)
    assert np.isfinite(np.asarray(prediction)).all()
