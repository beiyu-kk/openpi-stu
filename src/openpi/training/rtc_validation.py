"""Fixed-delay RTC validation with augmentation disabled and token-weighted aggregation."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


def batch_sums(model, rng, observation, actions, *, raw_action_dim):
    if model.simulated_delay is None:
        raise ValueError("Validation requires RTC model")
    # Same RNG across delay strata gives the same noise/time, improving comparisons.
    return jax.lax.map(
        lambda delay: model.rtc_validation_sums(rng, observation, actions, delay=delay, raw_action_dim=raw_action_dim),
        jnp.arange(model.simulated_delay, dtype=jnp.int32),
    )


def state_batch_sums(model_def, params, rng, batch, *, raw_action_dim):
    model = nnx.merge(model_def, params)
    model.eval()
    return batch_sums(model, rng, *batch, raw_action_dim=raw_action_dim)


def summarize(totals, *, raw_action_dim, model_action_dim, reduction):
    result = {}
    for d, count in enumerate(totals["active_tokens"]):
        key = f"rtc_val/d{d}"
        result[f"{key}/active_tokens"] = int(count)
        # A delay stratum without targets is not a zero-error success.
        if count == 0:
            result[f"{key}/loss"] = None
            result[f"{key}/raw_mse"] = None
            continue
        official = float(totals["squared_error"][d] / count)
        result[f"{key}/official_token_loss"] = official
        result[f"{key}/loss"] = official if reduction == "official" else official / model_action_dim
        result[f"{key}/raw_mse"] = float(totals["raw_squared_error"][d] / (count * raw_action_dim))
    return result


def evaluate_batches(evaluate, loader, *, seed, num_batches, raw_action_dim, model_action_dim, reduction):
    totals = None
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    for i, batch in enumerate(loader):
        sums = jax.device_get(evaluate(jax.random.fold_in(jax.random.key(seed), i), batch))
        if totals is None:
            totals = {k: np.asarray(v, dtype=np.float64) for k, v in sums.items()}
        else:
            for k, v in sums.items():
                totals[k] += np.asarray(v, dtype=np.float64)
        if i + 1 >= num_batches:
            break
    if totals is None:
        raise ValueError("Validation loader was empty")
    result = summarize(totals, raw_action_dim=raw_action_dim, model_action_dim=model_action_dim, reduction=reduction)
    result["rtc_val/batches"] = i + 1
    return result
