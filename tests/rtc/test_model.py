import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.shared import nnx_utils
from openpi.training.optimizer import RTCAdamW


def test_actual_parameter_paths_match_freeze_design(tiny_model):
    cfg, model, _ = tiny_model
    trainable = nnx.state(model, nnx.All(nnx.Param, nnx.Not(cfg.get_rtc_freeze_filter())))
    paths = {"/".join(map(str, p)) for p in trainable.flat_state()}
    assert paths
    assert all(not p.startswith("PaliGemma/img") for p in paths)
    expert = [p for p in paths if p.startswith("PaliGemma/llm")]
    assert expert
    assert all("_1/" in p for p in expert)
    assert any("pre_attention_norm_1/Dense_0/kernel" in p for p in expert)
    assert any("pre_ffw_norm_1/Dense_0/bias" in p for p in expert)
    assert any("final_norm_1/Dense_0/kernel" in p for p in expert)
    assert any("lora_a" in p for p in expert)
    assert all("lora" in p or "/Dense_0/" in p for p in expert)
    for name in ("time_mlp_in", "time_mlp_out", "action_in_proj", "action_out_proj"):
        assert f"{name}/kernel" in paths
        assert f"{name}/bias" in paths
    # Optimizer partition must also work on real NNX State, not just a Python dict.
    tx = RTCAdamW().create(1e-3)
    gradients = jax.tree.map(jnp.ones_like, trainable)
    updates, _ = tx.update(gradients, tx.init(trainable), trainable)
    flat = updates.flat_state()
    lora = next(v.value for p, v in flat.items() if "lora" in "/".join(p))
    time = flat[("time_mlp_in", "kernel")].value
    np.testing.assert_allclose(float(jnp.mean(time)), float(jnp.mean(lora)) * 0.2, rtol=1e-4)


def test_eval_retains_rtc_and_padding_is_excluded(tiny_model, monkeypatch):
    cfg, model, obs = tiny_model
    calls = []
    original = model.embed_suffix

    def record(observation, actions, time):
        calls.append((actions, time))
        return original(observation, actions, time)

    monkeypatch.setattr(model, "embed_suffix", record)
    actions = jnp.ones((2, 6, 4)) * 2
    noise = jnp.ones_like(actions) * 8
    pad = jnp.asarray([[False] * 5 + [True], [False] * 6])
    obs = dataclasses.replace(obs, action_is_pad=pad)
    errors, active = model.flow_errors(
        jax.random.key(3),
        obs,
        actions,
        train=False,
        rtc_delay=jnp.array([2, 3]),
        noise=noise,
        time=jnp.array([0.4, 0.6]),
    )
    np.testing.assert_allclose(calls[-1][1][0], [0, 0, 0.4, 0.4, 0.4, 0.4])
    np.testing.assert_allclose(calls[-1][0][0, :2], 2)
    np.testing.assert_allclose(calls[-1][0][0, 2:], 4.4)
    assert int(active.sum()) == 6  # 3 active in sample 0, 3 in sample 1
    assert not bool(active[0, -1])
    loss = model.compute_loss(
        jax.random.key(3),
        obs,
        actions,
        train=False,
        rtc_delay=jnp.array([2, 3]),
        noise=noise,
        time=jnp.array([0.4, 0.6]),
    )
    np.testing.assert_allclose(loss.mean(), jnp.sum(errors * active[..., None]) / active.sum(), rtol=1e-6)
    # Default train=False still samples the configured nonzero delay distribution.
    model.simulated_delay_weights = (0, 0, 1, 0, 0)
    _, active = model.flow_errors(jax.random.key(1), obs, actions, train=False)
    assert not bool(active[:, :2].any())


def test_jitted_sampler_keeps_clean_prefix_and_conditions_continuation(tiny_model):
    cfg, model, obs = tiny_model
    sample = nnx_utils.module_jit(model.sample_actions)
    noise = jax.random.normal(jax.random.key(4), (2, 6, 4))
    prefix = jnp.ones_like(noise) * 0.7
    delay = jnp.asarray([2, 3])
    result = sample(jax.random.key(1), obs, noise=noise, num_steps=3, rtc_prefix=prefix, rtc_delay=delay)
    np.testing.assert_array_equal(result[0, :2], prefix[0, :2])
    np.testing.assert_array_equal(result[1, :3], prefix[1, :3])
    assert np.isfinite(result).all()
    # adaRMS gates are zero-initialized in a random model: set them nonzero to
    # exercise real attention and distinguish conditioning from output-only copying.
    state = nnx.state(model)
    for path, value in state.flat_state().items():
        name = "/".join(path)
        if "_norm_1/Dense_0/bias" in name:
            value.value = jnp.ones_like(value.value) * 0.1
    nnx.update(model, state)
    sample = nnx_utils.module_jit(model.sample_actions)
    a = sample(jax.random.key(1), obs, noise=noise, num_steps=3, rtc_prefix=prefix, rtc_delay=delay)
    b = sample(jax.random.key(1), obs, noise=noise, num_steps=3, rtc_prefix=prefix * -1, rtc_delay=delay)
    assert not np.allclose(a[:, 3:], b[:, 3:])


def test_per_action_time_changes_adarms_numerically():
    from openpi.models.gemma import RMSNorm

    norm = RMSNorm()
    x = jnp.ones((1, 3, 8))
    condition = jnp.arange(24, dtype=jnp.float32).reshape(1, 3, 8)
    variables = norm.init(jax.random.key(0), x, condition)
    variables["params"]["Dense_0"]["kernel"] = jnp.ones((8, 24)) * 0.01
    output, gate = norm.apply(variables, x, condition)
    assert output.shape == x.shape
    assert gate.shape == x.shape
    assert not np.allclose(output[:, 0], output[:, 1])
    assert not np.allclose(gate[:, 0], gate[:, 1])
