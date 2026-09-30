# ruff: noqa: SLF001
import copy

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models.model import ModelType
from openpi.policies.piper_policy import PiperInputs
from openpi.policies.piper_policy import PiperOutputs
from openpi.policies.rtc_policy import TrainingRTCPolicy
from openpi.shared.normalize import NormStats


class DummyModel(nnx.Module):
    action_horizon = 6
    action_dim = 32
    simulated_delay = 5

    def sample_actions(self, rng, observation, **kwargs):
        return jnp.zeros((1, self.action_horizon, self.action_dim))


@pytest.mark.parametrize("quantile", [False, True])
def test_absolute_prefix_is_rebased_normalized_padded_once(quantile):
    stats = {
        key: NormStats(
            mean=np.arange(7) * 0.1,
            std=np.arange(7) * 0.1 + 0.2,
            q01=np.arange(7) * -0.2 - 1,
            q99=np.arange(7) * 0.2 + 1,
        )
        for key in ("state", "actions")
    }
    policy = TrainingRTCPolicy(
        DummyModel(),
        raw_action_dim=7,
        transforms=[
            PiperInputs(model_type=ModelType.PI05),
            transforms.DeltaActions((True,) * 7),
            transforms.Normalize(stats, use_quantiles=quantile),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[
            transforms.Unnormalize(stats, use_quantiles=quantile),
            transforms.AbsoluteActions((True,) * 7),
            PiperOutputs(),
        ],
    )
    calls = []

    def sample(rng, observation, **kwargs):
        calls.append((observation, kwargs))
        return kwargs["rtc_prefix"]

    policy._sample_actions = sample
    obs = {
        "observation/top_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/state": np.arange(7, dtype=np.float32),
    }
    absolute = np.stack([np.arange(7) + 0.3, np.arange(7) + 0.5]).astype(np.float32)
    obs["rtc"] = {"prefix": absolute, "delay": 2, "start_index": 12, "request_id": 8}
    before = copy.deepcopy(obs)
    response = policy.infer(obs)
    expected = transforms.Normalize({"actions": stats["actions"]}, use_quantiles=quantile)(
        {"actions": absolute - obs["observation/state"]}
    )["actions"]
    np.testing.assert_allclose(calls[-1][1]["rtc_prefix"][0, :2, :7], expected)
    np.testing.assert_array_equal(calls[-1][1]["rtc_prefix"][0, :, 7:], 0)
    np.testing.assert_array_equal(response["actions"][:2], absolute)
    np.testing.assert_array_equal(obs["rtc"]["prefix"], before["rtc"]["prefix"])
    np.testing.assert_array_equal(obs["observation/state"], before["observation/state"])
    # Reusing the SAME absolute commands with a new observation must change model coordinates.
    obs["observation/state"] += 0.2
    policy.infer(obs)
    assert not np.allclose(calls[-1][1]["rtc_prefix"][0, :2, :7], expected)
    obs["rtc"]["delay"] = 5
    with pytest.raises(ValueError, match="trained support"):
        policy.infer(obs)
