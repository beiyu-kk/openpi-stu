import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from openpi.models import gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import siglip
from openpi.shared import nnx_utils
from openpi.training import weight_loaders


@pytest.mark.parametrize("size", [224, 336, 448])
def test_image_specs_and_siglip_parameter_shape(size):
    config = pi0_config.Pi0Config(pi05=True, image_resolution=(size, size))
    obs, _ = config.inputs_spec(batch_size=2)
    assert all(image.shape == (2, size, size, 3) for image in obs.images.values())
    model = nnx.eval_shape(config.create, jax.random.key(0))
    params = nnx.state(model).to_pure_dict()
    assert params["PaliGemma"]["img"]["pos_embedding"].shape == (1, (size // 14) ** 2, 1152)


@pytest.mark.parametrize("resolution", [(0, 0), (-224, -224), (225, 225), (224, 448), (224,), (224.0, 224.0)])
def test_invalid_image_resolution(resolution):
    with pytest.raises(ValueError, match="image_resolution"):
        pi0_config.Pi0Config(image_resolution=resolution)


@pytest.fixture
def small_backbones(monkeypatch):
    # Keep the production patch extraction, attention and Pi0 forward paths with smaller weights.
    original_siglip = siglip.Module

    def small_siglip(*args, **kwargs):
        return original_siglip(*args, **{**kwargs, "variant": "mu/14"})

    monkeypatch.setattr(siglip, "Module", small_siglip)
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda _: gemma.Config(width=32, depth=1, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16),
    )


@pytest.mark.parametrize("size", [224, 336, 448])
def test_loss_gradients_and_inference_keep_resolution(small_backbones, monkeypatch, size):
    config = pi0_config.Pi0Config(
        pi05=True, image_resolution=(size, size), action_horizon=2, max_token_len=4, dtype="float32"
    )
    model = config.create(jax.random.key(0))
    obs, actions = config.fake_obs(), config.fake_act()
    # Nonconstant pixels make image augmentation and visual gradients meaningful.
    pixels = jax.random.uniform(jax.random.key(1), (1, size, size, 3), minval=-1, maxval=1)
    obs = dataclasses.replace(obs, images=dict.fromkeys(obs.images, pixels))
    original_preprocess = _model.preprocess_observation
    calls = []

    def checked_preprocess(*args, **kwargs):
        result = original_preprocess(*args, **kwargs)
        assert all(image.shape[1:3] == (size, size) for image in result.images.values())
        calls.append(kwargs["train"])
        return result

    monkeypatch.setattr(_model, "preprocess_observation", checked_preprocess)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))

    @nnx.jit
    def train_step(model, optimizer):
        def loss_fn(model):
            return jnp.mean(model.compute_loss(jax.random.key(2), obs, actions, train=True))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss, grads

    # The initial adaRMS gates and SigLIP output head are zero; updates open the visual gradient path.
    for _ in range(3):
        loss, grads = train_step(model, optimizer)
        assert np.isfinite(np.asarray(loss))
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(grads))
    assert np.any(np.asarray(grads.to_pure_dict()["PaliGemma"]["img"]["pos_embedding"]) != 0)
    prediction = nnx_utils.module_jit(model.sample_actions)(jax.random.key(3), obs, num_steps=1)
    assert prediction.shape == (1, 2, 32)
    assert np.isfinite(np.asarray(prediction)).all()
    assert True in calls
    assert False in calls


@pytest.mark.parametrize("size", [224, 336, 448])
def test_224_weights_to_configured_model_checkpoint(small_backbones, tmp_path, size):
    config = pi0_config.Pi0Config(pi05=True, action_horizon=2, max_token_len=4, dtype="float32")
    source = config.create(jax.random.key(0))
    target_config = dataclasses.replace(config, image_resolution=(size, size))
    target = nnx.eval_shape(target_config.create, jax.random.key(0))
    target_params = nnx.state(target).to_pure_dict()
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(tmp_path / "base", {"params": nnx.state(source).to_pure_dict()})
        converted = weight_loaders.CheckpointWeightLoader(str(tmp_path / "base"), resize_siglip_posemb=True).load(
            target_params
        )
        high_res_model = target_config.load(converted)
        checkpointer.save(tmp_path / "high_res", {"params": nnx.state(high_res_model).to_pure_dict()})
    restored = target_config.load(_model.restore_params(tmp_path / "high_res"))
    obs = target_config.fake_obs()
    encoded, masks, _ = nnx_utils.module_jit(restored.embed_prefix)(obs)
    assert encoded.shape == (1, 3 * (size // 14) ** 2 + 4, 32)
    assert masks.shape == encoded.shape[:2]
    prediction = nnx_utils.module_jit(restored.sample_actions)(jax.random.key(1), obs, num_steps=1)
    assert prediction.shape == (1, 2, 32)
    assert np.isfinite(np.asarray(prediction)).all()


def test_pytorch_rejects_custom_resolution():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    with pytest.raises(ValueError, match="JAX"):
        PI0Pytorch(pi0_config.Pi0Config(image_resolution=(448, 448)))
