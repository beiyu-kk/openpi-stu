from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models.image_token_merger import SpatialPatchMerger


def test_spatial_merger_reduces_patch_grid():
    merger = SpatialPatchMerger(embed_dim=4, merge_size=2, rngs=nnx.Rngs(0))
    output = merger(jax.random.normal(jax.random.key(1), (2, 32 * 32, 4)), 32, 32)
    assert output.shape == (2, 16 * 16, 4)


@pytest.mark.parametrize(("side", "expected"), [(16, 64), (32, 256)])
def test_spatial_merger_resolution(side, expected):
    merger = SpatialPatchMerger(embed_dim=8, merge_size=2, rngs=nnx.Rngs(0))
    output = merger(jnp.zeros((1, side * side, 8)), side, side)
    assert output.shape == (1, expected, 8)


def test_spatial_merger_rejects_non_divisible_grid():
    merger = SpatialPatchMerger(embed_dim=4, merge_size=2, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="not divisible"):
        merger(jnp.zeros((1, 25, 4)), 5, 5)
