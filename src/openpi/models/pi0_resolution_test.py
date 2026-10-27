import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from openpi.models import gemma
from openpi.models import gemma_fast
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.models import siglip
from openpi.shared import nnx_utils
from openpi.training import weight_loaders


@pytest.mark.parametrize("size", [224, 336, 448, 640])
def test_image_specs_and_siglip_parameter_shape(size):
    config = pi0_config.Pi0Config(pi05=True, image_resolution=(size, size))
    obs, _ = config.inputs_spec(batch_size=2)
    assert all(image.shape == (2, size, size, 3) for image in obs.images.values())
    model = nnx.eval_shape(config.create, jax.random.key(0))
    params = nnx.state(model).to_pure_dict()
    patch_side = (size + 13) // 14
    assert params["PaliGemma"]["img"]["pos_embedding"].shape == (1, patch_side**2, 1152)


@pytest.mark.parametrize("resolution", [(0, 0), (-224, -224), (224, 448), (224,), (224.0, 224.0)])
def test_invalid_image_resolution(resolution):
    with pytest.raises(ValueError, match="image_resolution"):
        pi0_config.Pi0Config(image_resolution=resolution)


def test_image_resolution_can_be_non_multiple_of_siglip_patch_size():
    config = pi0_config.Pi0Config(image_resolution=(640, 640))
    assert config.image_resolution == (640, 640)


def test_pad_only_canvas_can_have_an_odd_patch_grid_without_merging():
    config = pi0_config.Pi0Config(image_resolution=(320, 320), image_resize_mode="pad_only")
    shaped = nnx.eval_shape(config.create, jax.random.key(0))
    assert shaped.image_tokens_per_camera == 23 * 23
    assert nnx.state(shaped).to_pure_dict()["PaliGemma"]["img"]["pos_embedding"].shape[1] == 23 * 23
    with pytest.raises(ValueError, match="patch grid"):
        dataclasses.replace(config, image_token_merger="spatial_2x2")


def test_pad_only_model_preprocessing_preserves_image_pixels():
    config = pi0_config.Pi0Config(image_resolution=(640, 640), image_resize_mode="pad_only")
    observation = config.fake_obs()
    image = jnp.arange(480 * 640 * 3, dtype=jnp.float32).reshape(1, 480, 640, 3)
    image = image / image.max() * 2 - 1
    observation = dataclasses.replace(observation, images=dict.fromkeys(observation.images, image))

    processed = _model.preprocess_observation(
        None, observation, image_resolution=config.image_resolution, image_resize_mode=config.image_resize_mode
    )

    assert processed.images["base_0_rgb"].shape == (1, 640, 640, 3)
    np.testing.assert_array_equal(processed.images["base_0_rgb"][:, 80:560], image)


def test_pad_only_training_keeps_region_geometry():
    config = pi0_config.Pi0Config(image_resolution=(28, 28), image_resize_mode="pad_only")
    obs = config.fake_obs()
    pixels = jnp.linspace(-1, 1, 12 * 20 * 3).reshape(1, 12, 20, 3)
    region = jnp.zeros((1, 12, 20), dtype=jnp.float32).at[:, 2:9, 3:10].set(1)
    obs = dataclasses.replace(
        obs, images=dict.fromkeys(obs.images, pixels), region_masks=dict.fromkeys(obs.images, region)
    )
    processed = _model.preprocess_observation(
        jax.random.key(0), obs, train=True, image_resolution=(28, 28), image_resize_mode="pad_only"
    )
    expected = np.zeros((1, 28, 28), dtype=np.float32)
    expected[:, 8:20, 4:24] = region
    for mask in processed.region_masks.values():
        np.testing.assert_array_equal(mask, expected)


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


@pytest.mark.parametrize("family", ["pi0", "pi05", "pi0_fast"])
def test_dynamic_model_trains_and_decodes_multiple_sizes_with_fixed_parameters(small_backbones, monkeypatch, family):
    options = {
        "image_resolution": (28, 28),
        "pad_to_square": True,
        "action_horizon": 2,
        "max_token_len": 4,
        "dtype": "float32",
    }
    if family == "pi0_fast":
        original = gemma_fast.get_config

        def tiny_fast(variant):
            cfg = original(variant)
            cfg.update(
                {
                    "width": 32,
                    "depth": 1,
                    "mlp_dim": 64,
                    "num_heads": 2,
                    "num_kv_heads": 1,
                    "head_dim": 16,
                    "vocab_size": 32,
                }
            )
            return cfg

        monkeypatch.setattr(gemma_fast, "get_config", tiny_fast)
        config = pi0_fast.Pi0FASTConfig(**options)
    else:
        config = pi0_config.Pi0Config(pi05=family == "pi05", **options)
    model = config.create(jax.random.key(0))
    if family == "pi0_fast":
        params = nnx.state(model)
        embedding = params["PaliGemma"]["llm"]["embedder"]["input_embedding"]
        embedding.value = jax.random.normal(jax.random.key(1), embedding.value.shape) * 0.02
        nnx.update(model, params)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))
    initial_shapes = jax.tree.map(lambda x: x.shape, nnx.state(model).to_pure_dict())

    @nnx.jit
    def step(current, optimizer, obs):
        loss, grads = nnx.value_and_grad(
            lambda current: jnp.mean(current.compute_loss(jax.random.key(2), obs, config.fake_act(), train=True))
        )(current)
        optimizer.update(grads)
        return loss, grads.to_pure_dict()["PaliGemma"]["img"]["pos_embedding"]

    sample = nnx_utils.module_jit(
        model.sample_actions, **({"static_argnames": ("max_decoding_steps",)} if family == "pi0_fast" else {})
    )
    position_gradient = 0.0
    for side in (28, 42):
        obs = config.fake_obs()
        images = {
            key: jax.random.uniform(jax.random.key(i), (1, side + i * 7, side + i * 7, 3), minval=-1, maxval=1)
            for i, key in enumerate(obs.images)
        }
        obs = dataclasses.replace(
            obs,
            images=images,
            tokenized_prompt=jnp.array([[1, 2, 3, 4]], dtype=jnp.int32),
            tokenized_prompt_mask=jnp.ones((1, 4), dtype=jnp.bool_),
            token_ar_mask=jnp.array([[0, 0, 1, 1]], dtype=jnp.int32),
            token_loss_mask=jnp.array([[False, False, True, True]]),
        )
        # Update repeatedly to open the initially zero visual heads / adaRMS gates.
        for _ in range(3):
            loss, pos_grad = step(model, optimizer, obs)
            assert np.isfinite(np.asarray(loss))
            assert np.isfinite(np.asarray(pos_grad)).all()
            position_gradient += float(jnp.linalg.norm(pos_grad))
        prediction = sample(
            jax.random.key(3), obs, **({"max_decoding_steps": 2} if family == "pi0_fast" else {"num_steps": 1})
        )
        assert prediction.shape == ((1, 2) if family == "pi0_fast" else (1, 2, 32))
        assert np.isfinite(np.asarray(prediction)).all()
        assert jax.tree.map(lambda x: x.shape, nnx.state(model).to_pure_dict()) == initial_shapes
    assert position_gradient > 0


def test_dynamic_training_only_pads_image_and_region_geometry():
    config = pi0_config.Pi0Config(image_resolution=(28, 28), pad_to_square=True)
    obs = config.fake_obs()
    image = jnp.linspace(-1, 1, 30 * 43 * 3).reshape(1, 30, 43, 3)
    region = jnp.zeros((1, 30, 43), jnp.float32).at[:, 2:9, 3:10].set(1)
    obs = dataclasses.replace(
        obs, images=dict.fromkeys(obs.images, image), region_masks=dict.fromkeys(obs.images, region)
    )
    result = _model.preprocess_observation(
        jax.random.key(0), obs, train=True, image_resolution=(28, 28), pad_to_square=config.pad_to_square
    )
    expected_region = np.pad(np.asarray(region), ((0, 0), (6, 7), (0, 0)))
    for mask in result.region_masks.values():
        np.testing.assert_array_equal(mask, expected_region)
    result = _model.preprocess_observation(None, obs, pad_to_square=config.pad_to_square)
    for pixels in result.images.values():
        np.testing.assert_array_equal(pixels[:, 6:36], image)


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


def test_knowledge_insulation_trains_both_representations_and_stops_flow_gradient(small_backbones):
    config = pi0_config.Pi0Config(
        pi05=True,
        knowledge_insulation=True,
        image_resolution=(224, 224),
        action_horizon=2,
        max_token_len=8,
        dtype="float32",
    )
    model = config.create(jax.random.key(10))
    observation = dataclasses.replace(
        config.fake_obs(),
        tokenized_prompt=jnp.arange(2, 10, dtype=jnp.int32)[None, :],
        tokenized_prompt_mask=jnp.ones((1, 8), dtype=jnp.bool_),
        token_ar_mask=jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32),
        token_loss_mask=jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.bool_),
    )
    actions = config.fake_act()

    flow_loss, flow_grads = nnx.value_and_grad(
        lambda current: jnp.mean(current.flow_errors(jax.random.key(11), observation, actions)[0])
    )(model)
    discrete_loss, discrete_grads = nnx.value_and_grad(
        lambda current: jnp.mean(current.discrete_action_loss(jax.random.key(12), observation))
    )(model)
    joint_loss, joint_grads = nnx.value_and_grad(
        lambda current: jnp.mean(current.compute_loss(jax.random.key(13), observation, actions))
    )(model)

    assert np.isfinite(np.asarray(flow_loss))
    assert np.isfinite(np.asarray(discrete_loss))
    assert np.isfinite(np.asarray(joint_loss))

    def norm_matching(grads, predicate):
        leaves = []
        for path, value in jax.tree_util.tree_flatten_with_path(grads.to_pure_dict())[0]:
            path_keys = tuple(getattr(part, "key", part) for part in path)
            if predicate(path_keys):
                leaves.append(jnp.linalg.norm(value))
        return sum((float(value) for value in leaves), 0.0)

    # Flow matching must update the action expert while leaving all VLM-only
    # parameters untouched. FAST CE supplies the missing VLM gradient.
    assert norm_matching(flow_grads, lambda path: path[:2] == ("PaliGemma", "img")) == 0.0
    assert (
        norm_matching(
            flow_grads, lambda path: path[:2] == ("PaliGemma", "llm") and any(str(key).endswith("_1") for key in path)
        )
        > 0.0
    )
    assert norm_matching(discrete_grads, lambda path: path[:2] == ("PaliGemma", "img")) > 0.0
    assert norm_matching(discrete_grads, lambda path: path[:2] == ("PaliGemma", "llm")) > 0.0
    assert norm_matching(joint_grads, lambda path: path[:2] == ("PaliGemma", "img")) > 0.0


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
    patch_side = (size + 13) // 14
    assert encoded.shape == (1, 3 * patch_side**2 + 4, 32)
    assert masks.shape == encoded.shape[:2]
    prediction = nnx_utils.module_jit(restored.sample_actions)(jax.random.key(1), obs, num_steps=1)
    assert prediction.shape == (1, 2, 32)
    assert np.isfinite(np.asarray(prediction)).all()


def test_pytorch_rejects_custom_resolution():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    with pytest.raises(ValueError, match="JAX"):
        PI0Pytorch(pi0_config.Pi0Config(image_resolution=(448, 448)))
    with pytest.raises(ValueError, match="pad_to_square"):
        PI0Pytorch(pi0_config.Pi0Config(pad_to_square=True))


@pytest.mark.parametrize(("size", "tokens_per_image"), [(224, 64), (448, 256)])
def test_spatial_token_merger_reduces_prefix_tokens(small_backbones, size, tokens_per_image):
    config = pi0_config.Pi0Config(
        pi05=True,
        image_resolution=(size, size),
        image_token_merger="spatial_2x2",
        action_horizon=2,
        max_token_len=4,
        dtype="float32",
    )
    model = config.create(jax.random.key(0))
    encoded, masks, _ = nnx_utils.module_jit(model.embed_prefix)(config.fake_obs())
    assert encoded.shape == (1, 3 * tokens_per_image + 4, 32)
    assert masks.shape == encoded.shape[:2]


def test_spatial_piper_camera_selection_uses_two_image_slots(small_backbones):
    config = pi0_config.Pi0Config(
        pi05=True,
        image_resolution=(448, 448),
        image_token_merger="spatial_2x2",
        image_token_camera_keys=("base_0_rgb", "right_wrist_0_rgb"),
        action_horizon=2,
        max_token_len=4,
        dtype="float32",
    )
    model = config.create(jax.random.key(0))
    encoded, masks, _ = nnx_utils.module_jit(model.embed_prefix)(config.fake_obs())
    assert encoded.shape == (1, 2 * 256 + 4, 32)
    assert masks.shape == encoded.shape[:2]
