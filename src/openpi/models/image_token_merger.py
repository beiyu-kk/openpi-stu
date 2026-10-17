"""Optional spatial reduction of SigLIP image tokens.

The merger follows the spatial part of Qwen's vision token merger: neighbouring
patch embeddings are concatenated, normalized, and projected back to the
backbone width.  It runs after the SigLIP encoder, so SigLIP's own positional
embedding and patch grid remain unchanged.
"""

import flax.nnx as nnx
import jax


class SpatialPatchMerger(nnx.Module):
    """Merge a square ``merge_size`` by ``merge_size`` patch neighbourhood."""

    def __init__(self, embed_dim: int, merge_size: int, *, rngs: nnx.Rngs):
        if merge_size < 1:
            raise ValueError("merge_size must be positive")
        self.merge_size = merge_size
        merged_dim = merge_size * merge_size * embed_dim
        self.norm = nnx.LayerNorm(num_features=merged_dim, rngs=rngs)
        self.proj_in = nnx.Linear(merged_dim, embed_dim, rngs=rngs)
        self.proj_out = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, tokens: jax.Array, height: int, width: int) -> jax.Array:
        if tokens.ndim != 3:
            raise ValueError(f"Expected image tokens [batch, patches, dim], got {tokens.shape}")
        batch, patch_count, embed_dim = tokens.shape
        if patch_count != height * width:
            raise ValueError(f"Patch grid {height}x{width} does not match {patch_count} tokens")
        if height % self.merge_size or width % self.merge_size:
            raise ValueError(f"Patch grid {height}x{width} is not divisible by merge_size={self.merge_size}")

        size = self.merge_size
        grid = tokens.reshape(batch, height, width, embed_dim)
        # [B, H/s, s, W/s, s, D] -> [B, H/s, W/s, s*s*D].
        grid = grid.reshape(batch, height // size, size, width // size, size, embed_dim)
        grid = grid.transpose(0, 1, 3, 2, 4, 5).reshape(
            batch, height // size, width // size, size * size * embed_dim
        )
        grid = self.norm(grid)
        grid = jax.nn.gelu(self.proj_in(grid))
        grid = self.proj_out(grid)
        return grid.reshape(batch, -1, embed_dim)
