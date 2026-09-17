import dataclasses

import augmax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models import gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import region_guidance as rg
from openpi.models import siglip
from openpi.shared import nnx_utils


def test_attention_bias_scope_and_softmax():
    logits = jnp.zeros((1, 2, 2, 3, 4))
    keys = jnp.array([[0.0, 1.0, 0.0, 0.0]])
    queries = jnp.array([False, False, True])
    heads = jnp.array([False, True, False, False])
    actual = rg.add_attention_bias(logits, keys, queries, heads, layer_enabled=True)
    expected = np.zeros(logits.shape)
    expected[0, 0, 1, 2, 1] = 1
    np.testing.assert_array_equal(actual, expected)
    probability = jax.nn.softmax(actual, axis=-1)[0, 0, 1, 2, 1]
    np.testing.assert_allclose(probability, np.e / (np.e + 3), rtol=1e-6)
    np.testing.assert_array_equal(rg.add_attention_bias(logits, keys, queries, heads, layer_enabled=False), logits)


def test_attention_bias_preserves_hard_mask():
    config = gemma.Config(width=8, depth=1, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4)
    attention = gemma.Attention(configs=(config,))
    xs = [jax.random.normal(jax.random.key(0), (1, 4, 8))]
    positions = jnp.arange(4)[None, :]
    mask = jnp.ones((1, 1, 4, 4), bool).at[..., 1].set(False)
    variables = attention.init(jax.random.key(1), xs, positions, mask, None)
    baseline, _ = attention.apply(variables, xs, positions, mask, None)
    biased, _ = attention.apply(
        variables,
        xs,
        positions,
        mask,
        None,
        (jnp.array([[0.0, 100.0, 0.0, 0.0]]), jnp.ones(4, bool), jnp.ones(2, bool)),
        region_layer_enabled=True,
    )
    np.testing.assert_array_equal(biased[0], baseline[0])


@pytest.mark.parametrize(("step", "expected"), [(0, 0.5), (3000, 0.5), (5000, 0.25), (7000, 0), (9999, 0)])
def test_schedule(step, expected):
    strength, probability = jax.jit(lambda s: rg.RegionGuidanceConfig().schedule(s, 10000))(step)
    np.testing.assert_allclose([strength, probability], expected, atol=1e-7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength": -1},
        {"strength": float("nan")},
        {"keep_probability": 1.1},
        {"decay_start": 0.8},
        {"decay_end": 1},
        {"layers": ()},
        {"heads": (0, 0)},
    ],
)
def test_invalid_guidance_config(kwargs):
    with pytest.raises(ValueError, match="Region guidance"):
        rg.RegionGuidanceConfig(**kwargs)


def annotated_observation(config, batch_size=1):
    observation = config.fake_obs(batch_size)
    h, w = config.image_resolution
    pixels = jax.random.uniform(jax.random.key(5), (batch_size, h, w, 3), minval=-1, maxval=1)
    masks = jnp.zeros((batch_size, h, w)).at[:, 5:15, 7:18].set(1)
    return dataclasses.replace(
        observation,
        images=dict.fromkeys(observation.images, pixels),
        region_masks=dict.fromkeys(observation.images, masks),
        region_valid=dict.fromkeys(observation.images, jnp.ones(batch_size, bool)),
    )


def test_augmentation_preserves_baseline_images_and_transforms_masks():
    config = pi0_config.Pi0Config(image_resolution=(28, 28))
    obs = annotated_observation(config)
    rng = jax.random.key(19)
    transformed = _model.preprocess_observation(rng, obs, train=True, image_resolution=(28, 28))
    baseline = _model.preprocess_observation(
        rng, dataclasses.replace(obs, region_masks=None, region_valid=None), train=True, image_resolution=(28, 28)
    )
    for key in obs.images:
        np.testing.assert_array_equal(transformed.images[key], baseline.images[key])
    augment = augmax.Chain(
        augmax.RandomCrop(26, 26),
        augmax.Resize(28, 28),
        augmax.Rotate((-5, 5)),
        augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
        input_types=augmax.InputType.DENSE,
    )
    expected = augment(jax.random.split(rng, 1)[0], obs.region_masks["base_0_rgb"][0, ..., None])[..., 0]
    np.testing.assert_allclose(transformed.region_masks["base_0_rgb"][0], expected, atol=1e-6)
    np.testing.assert_array_equal(transformed.region_masks["right_wrist_0_rgb"], obs.region_masks["right_wrist_0_rgb"])
    assert "region_masks" not in dataclasses.replace(obs, region_masks=None, region_valid=None).to_dict()


def test_patch_order_camera_validity_and_whole_camera_dropout():
    config = pi0_config.Pi0Config(image_resolution=(28, 28))
    obs = annotated_observation(config, batch_size=32)
    full = jnp.ones((32, 28, 28))
    obs = dataclasses.replace(
        obs,
        region_masks=dict.fromkeys(obs.images, full),
        region_valid={**obs.region_valid, "right_wrist_0_rgb": jnp.zeros(32, bool)},
        image_masks={**obs.image_masks, "left_wrist_0_rgb": jnp.zeros(32, bool)},
    )
    bias = rg.image_key_bias(obs, jax.random.key(0), 0.5, 0.5)
    np.testing.assert_array_equal(bias[:, 4:], 0)
    assert set(np.unique(bias[:, :4])) == {0.0, 0.5}
    np.testing.assert_array_equal(bias[:, :4], np.repeat(np.asarray(bias[:, :1]), 4, axis=1))
    np.testing.assert_array_equal(rg.image_key_bias(obs, jax.random.key(0), 0.5, 0.0), 0)
    mask = jnp.zeros((1, 28, 28)).at[:, :7, 14:].set(1)
    np.testing.assert_array_equal(rg.patch_coverage(mask), [[0, 0.5, 0, 0]])


@pytest.fixture
def small_backbones(monkeypatch):
    original_siglip = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda *a, **k: original_siglip(*a, **{**k, "variant": "mu/14"}))
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda _: gemma.Config(width=32, depth=2, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16),
    )


def test_guided_loss_gradients_checkpoint_compatibility_and_unbiased_inference(small_backbones):
    config = pi0_config.Pi0Config(
        pi05=True, image_resolution=(28, 28), action_horizon=2, max_token_len=4, dtype="float32"
    )
    guided_config = dataclasses.replace(config, region_guidance=rg.RegionGuidanceConfig())
    baseline = config.create(jax.random.key(0))
    guided = guided_config.create(jax.random.key(0))
    old_params, new_params = nnx.state(baseline).to_pure_dict(), nnx.state(guided).to_pure_dict()
    assert jax.tree.structure(old_params) == jax.tree.structure(new_params)
    for old, new in zip(jax.tree.leaves(old_params), jax.tree.leaves(new_params), strict=True):
        np.testing.assert_array_equal(old, new)
    obs = annotated_observation(config)
    actions = config.fake_act()
    raw_obs = dataclasses.replace(obs, region_masks=None, region_valid=None)
    plain_loss = nnx_utils.module_jit(baseline.compute_loss, static_argnames=("train",))(
        jax.random.key(3), raw_obs, actions, train=True
    )
    zero_loss = nnx_utils.module_jit(guided.compute_loss, static_argnames=("train",))(
        jax.random.key(3), obs, actions, train=True, region_strength=0.0, region_keep_probability=0.0
    )
    np.testing.assert_allclose(zero_loss, plain_loss, rtol=1e-6, atol=1e-6)
    optimizer = nnx.Optimizer(guided, optax.adam(1e-3))

    @nnx.jit
    def update(model, optimizer):
        def loss_fn(model):
            return model.compute_loss(
                jax.random.key(3), obs, actions, train=True, region_strength=0.5, region_keep_probability=1.0
            ).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss, grads

    for _ in range(3):
        loss, grads = update(guided, optimizer)
        assert np.isfinite(np.asarray(loss))
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grads))
    # Weights trained with guidance load into the original, unmodified inference configuration.
    restored = config.load(nnx.state(guided).to_pure_dict())
    predicted = nnx_utils.module_jit(guided.sample_actions)(jax.random.key(4), obs, num_steps=2)
    plain_predicted = nnx_utils.module_jit(restored.sample_actions)(jax.random.key(4), raw_obs, num_steps=2)
    np.testing.assert_allclose(predicted, plain_predicted, rtol=1e-6, atol=1e-6)
    assert predicted.shape == (1, 2, 32)
