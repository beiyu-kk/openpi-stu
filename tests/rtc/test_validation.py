# ruff: noqa: SLF001
import dataclasses
import functools
import importlib.util
from pathlib import Path

import flax.nnx as nnx
import jax
import numpy as np
import torch

from openpi.training import config
from openpi.training import data_loader
from openpi.training import rtc_validation
from openpi.training.optimizer import RTCAdamW
from openpi.training.utils import TrainState


def test_sparse_episode_ids_query_correct_boundary():
    dataset = object.__new__(data_loader.RTCLeRobotDataset)
    dataset.episodes = [2, 5]
    dataset.episode_data_index = {"from": torch.tensor([0, 3]), "to": torch.tensor([3, 7])}
    dataset.delta_indices = {"action": [0, 1, 2, 3]}
    indices, padding = dataset._get_query_indices(5, 5)
    assert indices["action"] == [5, 6, 6, 6]
    assert padding["action_is_pad"].tolist() == [False, False, True, True]
    indices, _ = dataset._get_query_indices(1, 2)
    assert indices["action"] == [1, 2, 2, 2]


def test_validation_sums_not_mean_of_batch_means():
    outputs = iter(
        [
            {"active_tokens": np.array([2]), "squared_error": np.array([4.0]), "raw_squared_error": np.array([4.0])},
            {"active_tokens": np.array([1]), "squared_error": np.array([9.0]), "raw_squared_error": np.array([9.0])},
        ]
    )
    metrics = rtc_validation.evaluate_batches(
        lambda *_: next(outputs),
        [None, None],
        seed=0,
        num_batches=2,
        raw_action_dim=1,
        model_action_dim=1,
        reduction="official",
    )
    assert metrics["rtc_val/d0/loss"] == 13 / 3
    assert metrics["rtc_val/d0/active_tokens"] == 3


def test_jitted_training_step_and_fixed_delay_validation(tiny_model):
    cfg, model, obs = tiny_model
    path = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("rtc_train_script", path)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    train_config = dataclasses.replace(
        config.get_config("pi05_piper_lora_finetune_rtc"), model=cfg, freeze_filter=cfg.get_rtc_freeze_filter()
    )
    params = nnx.state(model)
    tx = RTCAdamW().create(1e-4)
    state = TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(train_config.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )
    actions = jax.random.normal(jax.random.key(2), (2, 6, 4))
    step = jax.jit(functools.partial(train.train_step, train_config))
    new_state, info = step(jax.random.key(1), state, (obs, actions))
    assert int(new_state.step) == 1
    assert np.isfinite(info["loss"])
    frozen_before = params.filter(train_config.freeze_filter)
    frozen_after = new_state.params.filter(train_config.freeze_filter)
    for a, b in zip(jax.tree.leaves(frozen_before), jax.tree.leaves(frozen_after), strict=True):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(
        new_state.params["action_out_proj"]["kernel"].value, params["action_out_proj"]["kernel"].value
    )
    validate = jax.jit(functools.partial(rtc_validation.state_batch_sums, state.model_def, raw_action_dim=3))
    sums = validate(new_state.params, jax.random.key(0), (obs, actions))
    np.testing.assert_array_equal(sums["active_tokens"], [12, 10, 8, 6, 4])
    assert np.isfinite(sums["squared_error"]).all()
