import dataclasses
import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.region_guidance import RegionGuidanceConfig
from openpi.training import config
from openpi.training import utils
from openpi.training.rtc_config import TrainingRTCConfig
from scripts import train


def training_config(model_config, mode):
    return config.resolve_training_config(
        config.TrainConfig(
            name="small_rtc",
            model=model_config,
            data=config.FakeDataConfig(),
            training_rtc=TrainingRTCConfig(finetune_mode=mode),
            num_train_steps=10,
        )
    )


@pytest.mark.parametrize("mode", ["full", "lora"])
@pytest.mark.parametrize("guidance", [False, True])
def test_real_update_obeys_rtc_parameter_selection(tiny_model, mode, guidance):
    tiny_config, _, _ = tiny_model
    model_config = dataclasses.replace(
        tiny_config,
        action_dim=7,
        action_expert_variant="gemma_300m" if mode == "full" else "gemma_300m_lora",
        rtc_finetune_mode=mode,
        region_guidance=RegionGuidanceConfig() if guidance else None,
    )
    cfg = training_config(model_config, mode)
    model = cfg.model.create(jax.random.key(2))
    params = nnx.state(model)
    # Open the randomly initialized adaRMS gates as a pretrained checkpoint does.
    for path, value in params.flat_state().items():
        if "_norm_1/Dense_0/" in "/".join(path):
            value.value = jax.random.normal(jax.random.key(4), value.value.shape) * 0.1
    nnx.update(model, params)
    tx = cfg.optimizer.create(1e-3)
    state = utils.TrainState(
        step=jnp.array(0),
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(cfg.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )
    obs = cfg.model.fake_obs(batch_size=2)
    if guidance:
        obs = dataclasses.replace(
            obs,
            region_masks={k: jnp.ones((2, 28, 28)) for k in obs.images},
            region_valid={k: jnp.ones(2, bool) for k in obs.images},
        )
    actions = jax.random.normal(jax.random.key(8), (2, 6, 7))
    obs = dataclasses.replace(obs, action_is_pad=jnp.array([[False] * 4 + [True] * 2, [False] * 6]))
    step = jax.jit(functools.partial(train.train_step, cfg))
    new_state, metrics = step(jax.random.key(3), state, (obs, actions))
    assert np.isfinite(float(metrics["loss"]))
    assert float(metrics["grad_norm"]) > 0
    before = params.flat_state()
    changed = {p for p, v in new_state.params.flat_state().items() if not np.array_equal(v.value, before[p].value)}
    assert changed
    allowed = set(params.filter(cfg.trainable_filter).flat_state())
    assert changed <= allowed
    names = {"/".join(p) for p in changed}
    if mode == "full":
        assert len(allowed) == len(before)
        assert not any("lora" in "/".join(p) for p in before)
        assert any(p.startswith("PaliGemma/img/") for p in names)
        assert any(p.startswith("PaliGemma/llm/embedder/") for p in names)
        assert any("/attn/" in p and "_1/" not in p for p in names)
    else:
        assert all(not p.startswith("PaliGemma/img/") for p in names)
        assert any("lora" in p for p in names)
    assert any("_norm_1/Dense_0/" in p for p in names)
    assert any(p.startswith("time_mlp_in/") for p in names)
    # Momentum must not update parameters on batches with no valid targets.
    empty_obs = dataclasses.replace(obs, action_is_pad=jnp.ones((2, 6), bool))
    skipped, empty_metrics = step(jax.random.key(3), new_state, (empty_obs, actions))
    assert float(empty_metrics["rtc/active_tokens"]) == 0
    for actual, expected in zip(jax.tree.leaves(skipped.params), jax.tree.leaves(new_state.params), strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(jax.tree.leaves(skipped.opt_state), jax.tree.leaves(new_state.opt_state), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_zero_delay_matches_original_loss_and_padding_does_not_leak(tiny_model):
    cfg, model, obs = tiny_model
    baseline_cfg = dataclasses.replace(cfg, simulated_delay=None)
    baseline = baseline_cfg.load(nnx.state(model).to_pure_dict())
    actions = cfg.fake_act(batch_size=2)
    rng = jax.random.key(12)
    model.rtc_loss_reduction = "per_element"
    np.testing.assert_allclose(
        model.compute_loss(rng, obs, actions, rtc_delay=0),
        baseline.compute_loss(rng, obs, actions),
        rtol=2e-6,
        atol=2e-6,
    )
    params = nnx.state(model)
    for path, value in params.flat_state().items():
        if "_norm_1/Dense_0/bias" in "/".join(path):
            value.value = jnp.ones_like(value.value) * 0.1
    nnx.update(model, params)
    pad = jnp.array([[False] * 4 + [True] * 2] * 2)
    obs = dataclasses.replace(obs, action_is_pad=pad)
    a = model.compute_loss(rng, obs, actions, rtc_delay=2)
    b = model.compute_loss(rng, obs, actions.at[:, 4:].set(900), rtc_delay=2)
    np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6)
