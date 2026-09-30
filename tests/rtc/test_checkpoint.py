from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models import model as model_lib
from openpi.shared.normalize import NormStats
from openpi.training import checkpoints
from openpi.training import rtc_manifest
from openpi.training import sharding
from openpi.training.optimizer import AdamW
from openpi.training.optimizer import RTCAdamW
from openpi.training.utils import TrainState


@pytest.mark.parametrize("checkpoint_step", [0, 1])
def test_checkpoint_persists_rtc_provenance_and_parameter_precision(tiny_model, tmp_path, checkpoint_step):
    cfg, model, _ = tiny_model
    params = nnx.state(model)
    trainable = params.filter(nnx.All(nnx.Param, nnx.Not(cfg.get_rtc_freeze_filter())))
    tx = RTCAdamW().create(1e-4)
    state = TrainState(
        step=checkpoint_step + 1,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(trainable),
        ema_decay=None,
        ema_params=None,
    )
    data = SimpleNamespace(norm_stats={"actions": NormStats(mean=np.zeros(4), std=np.ones(4))}, asset_id="test")
    loader = SimpleNamespace(data_config=lambda: data)
    manifest = {
        "schema": 1,
        "model": {"simulated_delay": 5},
        "norm_stats_sha256": rtc_manifest.stats_digest(data.norm_stats),
    }
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoints", keep_period=1, overwrite=False, resume=False
    )
    try:
        checkpoints.save_state(manager, state, loader, checkpoint_step, rtc_manifest=manifest)
        manager.wait_until_finished()
    finally:
        manager.close()
    directory = tmp_path / "checkpoints" / str(checkpoint_step)
    assert rtc_manifest.read(directory) == manifest
    restored = model_lib.restore_params(directory / "params", restore_type=np.ndarray)
    np.testing.assert_array_equal(restored["time_mlp_in"]["kernel"], params["time_mlp_in"]["kernel"].value)
    assert restored["time_mlp_in"]["kernel"].dtype == np.float32
    assert checkpoints.load_norm_stats(directory / "assets", "test") is not None
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoints", keep_period=1, overwrite=False, resume=True
    )
    try:
        assert resuming  # A committed step 0 must not be mistaken for an empty directory.
        recovered = checkpoints.restore_state(manager, state, loader)
        assert int(recovered.step) == checkpoint_step + 1
    finally:
        manager.close()


def test_two_device_full_adamw_state_roundtrip(tmp_path):
    if jax.device_count() < 2:
        pytest.skip("Run with XLA_FLAGS=--xla_force_host_platform_device_count=2 on CPU")
    model = nnx.Linear(8, 8, rngs=nnx.Rngs(0))
    params = nnx.state(model)
    tx = AdamW().create(1e-5)
    updates, moments = tx.update(jax.tree.map(jnp.ones_like, params), tx.init(params), params)
    state = TrainState(
        step=jnp.array(8),
        params=optax.apply_updates(params, updates),
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=moments,
        ema_decay=None,
        ema_params=None,
    )
    mesh = sharding.make_mesh(2)
    shape = jax.eval_shape(lambda: state)
    placement = sharding.fsdp_sharding(shape, mesh, min_size_mbytes=0)
    state = jax.device_put(state, placement)
    assert not state.params["kernel"].value.sharding.is_fully_replicated
    loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=None, asset_id=None))
    manager, _ = checkpoints.initialize_checkpoint_dir(tmp_path / "full", keep_period=1, overwrite=False, resume=False)
    try:
        checkpoints.save_state(manager, state, loader, 7, rtc_manifest={"schema": 1})
        manager.wait_until_finished()
    finally:
        manager.close()
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        tmp_path / "full", keep_period=1, overwrite=False, resume=True
    )
    try:
        assert resuming
        recovered = checkpoints.restore_state(manager, shape, loader)
        assert int(recovered.step) == 8
        for actual, expected in zip(jax.tree.leaves(recovered), jax.tree.leaves(state), strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert recovered.params["kernel"].value.sharding.is_equivalent_to(state.params["kernel"].value.sharding, 2)
    finally:
        manager.close()
