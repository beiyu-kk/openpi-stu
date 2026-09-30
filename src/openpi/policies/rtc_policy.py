"""JAX-only hard-prefix policy. Wire actions are absolute, in robot units."""

import copy
import operator
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.policies.policy import Policy


def integer(value, name):
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


class TrainingRTCPolicy(Policy):
    """Reuses the EXACT training transforms for the prefix, including current-state deltas.

    obs['rtc'] = {prefix: float[d,raw_dim], delay: int, start_index: int, request_id: int}
    prefix[0] is the command at the observation's control tick, NOT at response arrival.
    All requests are stateless on the server. Queue ownership stays with the client.
    """

    def __init__(self, *args, raw_action_dim, **kwargs):
        if kwargs.get("is_pytorch", False):
            raise ValueError("Training RTC supports JAX only")
        super().__init__(*args, **kwargs)
        if self._model.simulated_delay is None:
            raise ValueError("TrainingRTCPolicy requires simulated_delay")
        self._raw_action_dim = raw_action_dim
        steps = self._sample_kwargs.get("num_steps", 10)
        if integer(steps, "num_steps") <= 0:
            raise ValueError("num_steps must be positive")

    def infer(self, obs, *, noise=None):
        inputs = copy.deepcopy(obs)
        request = inputs.pop("rtc", None)
        if "action_is_pad" in inputs:
            raise ValueError("action_is_pad is training metadata and must not be supplied at inference")
        if request is None:
            request = {"delay": 0, "prefix": np.empty((0, self._raw_action_dim)), "start_index": 0, "request_id": 0}
        delay = integer(request["delay"], "delay")
        start = integer(request["start_index"], "start_index")
        request_id = integer(request["request_id"], "request_id")
        if start < 0 or request_id < 0:
            raise ValueError("start_index and request_id must be nonnegative")
        if not 0 <= delay < self._model.simulated_delay:
            raise ValueError(f"delay must be in trained support [0,{self._model.simulated_delay})")
        prefix = np.asarray(request["prefix"], dtype=np.float32)
        if prefix.shape != (delay, self._raw_action_dim) or not np.isfinite(prefix).all():
            raise ValueError("prefix must be finite absolute commands with shape [delay, raw_action_dim]")
        # Fill unused positions with the current state. They are ignored by the sampler.
        # Running the same input transforms avoids a second, subtly different normalization codec.
        state = np.asarray(inputs.get("observation/state", inputs.get("state")), dtype=np.float32)
        if state.shape != (self._raw_action_dim,) or not np.isfinite(state).all():
            raise ValueError("state must match raw_action_dim and contain finite values")
        inputs["actions"] = np.broadcast_to(state, (self._model.action_horizon, self._raw_action_dim)).copy()
        inputs["actions"][:delay] = prefix
        inputs = self._input_transform(inputs)
        normalized_prefix = np.asarray(inputs.pop("actions"), dtype=np.float32)
        if normalized_prefix.shape != (self._model.action_horizon, self._model.action_dim):
            raise ValueError("RTC transforms must normalize delta actions and pad to model action_dim")
        if not np.isfinite(normalized_prefix).all():
            raise ValueError("Nonfinite normalized RTC prefix")
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        self._rng, rng = jax.random.split(self._rng)
        kwargs = dict(self._sample_kwargs)
        kwargs.update(rtc_prefix=jnp.asarray(normalized_prefix)[None], rtc_delay=jnp.asarray([delay], dtype=jnp.int32))
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            if noise.shape != (self._model.action_horizon, self._model.action_dim) or not np.isfinite(noise).all():
                raise ValueError("noise must be finite [H, model_action_dim]")
            kwargs["noise"] = jnp.asarray(noise)[None]
        t0 = time.monotonic()
        actions = self._sample_actions(rng, _model.Observation.from_dict(inputs), **kwargs)
        # Conversion synchronizes JAX so infer_ms includes device execution.
        outputs = {"state": np.asarray(inputs["state"][0]), "actions": np.asarray(actions[0])}
        outputs = self._output_transform(outputs)
        actions = np.asarray(outputs["actions"], dtype=np.float32).copy()
        if actions.shape != (self._model.action_horizon, self._raw_action_dim) or not np.isfinite(actions).all():
            raise ValueError("Invalid decoded action chunk")
        # Avoid roundoff in normalize/unnormalize for already committed absolute commands.
        actions[:delay] = prefix
        return {
            "actions": actions,
            "rtc": {"request_id": request_id, "start_index": start, "delay": delay, "action_space": "absolute"},
            "policy_timing": {"infer_ms": (time.monotonic() - t0) * 1000},
        }
