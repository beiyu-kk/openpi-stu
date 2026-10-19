"""Training-only, parameter-free region guidance for selected query-to-image attention."""

import dataclasses
import math
from typing import Literal

import jax
import jax.numpy as jnp

RegionQueryType = Literal["action", "image", "prompt"]
REGION_QUERY_TYPES = ("action", "image", "prompt")


@dataclasses.dataclass(frozen=True)
class RegionGuidanceConfig:
    # Initial additive attention-logit multiplier. For a fully covered patch,
    # the pre-softmax logit receives ``attention_bias_scale`` before decay.
    attention_bias_scale: float = 0.5
    keep_probability: float = 0.5
    decay_start: float = 0.3
    decay_end: float = 0.7
    # Query token types that may attend to annotated image patches. The default
    # preserves the original action-query-only behavior. Named experiments can
    # explicitly opt into image and prompt queries as well.
    query_types: tuple[RegionQueryType, ...] = ("action",)
    # None selects the two middle layers and the first quarter of query heads.
    layers: tuple[int, ...] | None = None
    heads: tuple[int, ...] | None = None

    def __post_init__(self):
        if not math.isfinite(self.attention_bias_scale) or self.attention_bias_scale < 0:
            raise ValueError("Region guidance attention_bias_scale must be finite and nonnegative")
        if not 0 <= self.keep_probability <= 1:
            raise ValueError("Region guidance keep_probability must be in [0, 1]")
        if not 0 <= self.decay_start < self.decay_end < 1:
            raise ValueError("Region guidance requires 0 <= decay_start < decay_end < 1")
        if (
            not self.query_types
            or len(set(self.query_types)) != len(self.query_types)
            or any(query_type not in REGION_QUERY_TYPES for query_type in self.query_types)
        ):
            raise ValueError("Region guidance query_types must contain distinct values from action, image, prompt")
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
        return self.attention_bias_scale * factor, self.keep_probability * factor

    @property
    def strength(self) -> float:
        """Backward-compatible name for the scheduled initial scale."""
        return self.attention_bias_scale


def query_mask(
    total_queries: int,
    *,
    image_range: tuple[int, int] | None = None,
    prompt_range: tuple[int, int] | None = None,
    action_range: tuple[int, int] | None = None,
    query_types: tuple[RegionQueryType, ...] = ("action",),
) -> jax.Array:
    """Build a static query mask for the configured token-type scopes.

    The ranges are half-open positions in the full Transformer sequence. Keeping
    this mask one-dimensional matches the existing grouped-head attention API;
    padding and hard attention visibility are still enforced by the normal
    per-example attention mask.
    """
    if total_queries < 0:
        raise ValueError("total_queries must be nonnegative")
    selected = jnp.zeros((total_queries,), dtype=jnp.bool_)
    ranges = {"image": image_range, "prompt": prompt_range, "action": action_range}
    for query_type in query_types:
        if query_type not in REGION_QUERY_TYPES:
            raise ValueError(f"Unknown region query type: {query_type}")
        bounds = ranges[query_type]
        if bounds is None:
            continue
        start, end = bounds
        if not 0 <= start <= end <= total_queries:
            raise ValueError(f"Invalid {query_type} query range {bounds} for sequence length {total_queries}")
        selected = selected.at[start:end].set(True)
    return selected


def patch_coverage(masks, patch_size: int = 14, merge_size: int = 1):
    """Average pixel coverage, preserving the encoder's row-major patch order."""
    batch, height, width = masks.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Region masks must have dimensions divisible by the image patch size")
    coverage = masks.reshape(batch, height // patch_size, patch_size, width // patch_size, patch_size).mean(axis=(2, 4))
    patch_height, patch_width = coverage.shape[1:]
    if type(merge_size) is not int or merge_size < 1:
        raise ValueError("merge_size must be a positive integer")
    if patch_height % merge_size or patch_width % merge_size:
        raise ValueError("Patch grid must be divisible by merge_size")
    if merge_size > 1:
        coverage = coverage.reshape(
            batch, patch_height // merge_size, merge_size, patch_width // merge_size, merge_size
        ).mean(axis=(2, 4))
    return coverage.reshape(batch, -1)


def image_key_bias(observation, rng, strength, keep_probability, *, merge_size: int = 1, camera_keys=None):
    """Sample one gate per example and camera, not per patch or attention head."""
    if observation.region_masks is None or observation.region_valid is None:
        raise ValueError("Region-guided training requires region_masks and region_valid from the annotation loader")
    keys = tuple(observation.images) if camera_keys is None else tuple(camera_keys)
    if not keys:
        raise ValueError("At least one camera key is required for region attention bias")
    if any(key not in observation.images for key in keys):
        raise ValueError("Region attention camera keys must exist in observation.images")
    if not set(keys).issubset(observation.region_masks) or not set(keys).issubset(observation.region_valid):
        raise ValueError("Region annotations must include all selected model camera keys")
    gates = jax.random.bernoulli(rng, keep_probability, (observation.state.shape[0], len(keys)))
    biases = []
    for camera_index, key in enumerate(keys):
        coverage = patch_coverage(jnp.clip(observation.region_masks[key], 0.0, 1.0), merge_size=merge_size)
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
