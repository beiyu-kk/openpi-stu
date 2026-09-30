"""Training-time RTC math in OpenPI's t=1 noise, t=0 clean convention."""

import jax
import jax.numpy as jnp


def draw_delay(rng, batch_shape, upper_bound, weights=None):
    # Stable equivalent of the official exp(S-1-d) distribution.
    logits = -jnp.arange(upper_bound, dtype=jnp.float32)
    if weights is not None:
        logits = jnp.log(jnp.asarray(weights, dtype=jnp.float32))
    return jax.random.categorical(rng, logits, shape=batch_shape)


def loss_grid(errors, active, reduction="official"):
    """mean(result) equals the global active-token loss, including action dimension sum.

    `official` matches real-time-chunking-kinetix/model.py, including its lack of
    a D divisor. `per_element` explicitly divides by D; neither weights samples
    equally when their active lengths differ. On a sharded global JAX array the
    sum is global (no per-device normalization).
    """
    numerator = jnp.sum(jnp.where(active[..., None], errors, 0.0), axis=-1)
    denominator = jnp.maximum(jnp.sum(active), 1)
    if reduction == "per_element":
        denominator = denominator * errors.shape[-1]
    elif reduction != "official":
        raise ValueError(f"Unknown reduction: {reduction}")
    return numerator * (active.size / denominator)
