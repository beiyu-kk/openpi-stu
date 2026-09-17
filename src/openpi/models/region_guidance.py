"""Training-only, parameter-free region guidance for action-to-image attention."""

import dataclasses
import math

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class RegionGuidanceConfig:
    strength: float = 0.5
    keep_probability: float = 0.5
    decay_start: float = 0.3
    decay_end: float = 0.7
    # None selects the two middle layers and the first quarter of query heads.
    layers: tuple[int, ...] | None = None
    heads: tuple[int, ...] | None = None

    def __post_init__(self):
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("Region guidance strength must be finite and nonnegative")
        if not 0 <= self.keep_probability <= 1:
            raise ValueError("Region guidance keep_probability must be in [0, 1]")
        if not 0 <= self.decay_start < self.decay_end < 1:
            raise ValueError("Region guidance requires 0 <= decay_start < decay_end < 1")
        for name in ("layers", "heads"):
            indices = getattr(self, name)
            if indices is not None and (
                not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 for i in indices)
            ):
                raise ValueError(f"Region guidance {name} must contain distinct nonnegative indices")

    def layer_indices(self, depth: int) -> tuple[int, ...]:
        start = max(0, depth // 2 - 1)
        indices = self.layers if self.layers is not None else tuple(range(start, min(depth, start + 2)))
        if max(indices) >= depth:
            raise ValueError(f"Region guidance layer index exceeds model depth {depth}")
        return indices

    def head_indices(self, num_heads: int) -> tuple[int, ...]:
        indices = self.heads if self.heads is not None else tuple(range(max(1, num_heads // 4)))
        if max(indices) >= num_heads:
            raise ValueError(f"Region guidance head index exceeds query head count {num_heads}")
        return indices

    def schedule(self, step, total_steps: int):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        progress = jnp.asarray(step, dtype=jnp.float32) / total_steps
        factor = jnp.clip((self.decay_end - progress) / (self.decay_end - self.decay_start), 0.0, 1.0)
        return self.strength * factor, self.keep_probability * factor


def patch_coverage(masks, patch_size: int = 14):
    """Average pixel coverage, preserving the encoder's row-major patch order."""
    batch, height, width = masks.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Region masks must have dimensions divisible by the image patch size")
    return (
        masks.reshape(batch, height // patch_size, patch_size, width // patch_size, patch_size)
        .mean(axis=(2, 4))
        .reshape(batch, -1)
    )


def image_key_bias(observation, rng, strength, keep_probability):
    """Sample one gate per example and camera, not per patch or attention head."""
    if observation.region_masks is None or observation.region_valid is None:
        raise ValueError("Region-guided training requires region_masks and region_valid from the annotation loader")
    keys = tuple(observation.images)
    if set(observation.region_masks) != set(keys) or set(observation.region_valid) != set(keys):
        raise ValueError("Region annotations must match all model camera keys")
    gates = jax.random.bernoulli(rng, keep_probability, (observation.state.shape[0], len(keys)))
    biases = []
    for camera_index, key in enumerate(keys):
        coverage = patch_coverage(jnp.clip(observation.region_masks[key], 0.0, 1.0))
        valid = observation.region_valid[key] & observation.image_masks[key]
        enabled = gates[:, camera_index] & valid
        biases.append(coverage * (strength * enabled[:, None]))
    return jnp.concatenate(biases, axis=-1)


def add_attention_bias(logits, key_bias, query_mask, head_mask, layer_enabled):
    """Apply a separable bias to B,K,G,T,S logits without constructing a full mask."""
    batch, kv_heads, groups, queries, keys = logits.shape
    if key_bias.shape != (batch, keys) or query_mask.shape != (queries,):
        raise ValueError("Region attention bias does not match query/key lengths")
    if head_mask.shape != (kv_heads * groups,):
        raise ValueError("Region attention head mask does not match grouped query heads")
    bias = (
        key_bias[:, None, None, None, :]
        * query_mask[None, None, None, :, None]
        * head_mask.reshape(1, kv_heads, groups, 1, 1)
        * layer_enabled
    )
    return logits + bias
