import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from openpi.shared import array_typing as at
from openpi.training import weight_loaders


def _params(posemb, *, projection_width=4):
    return {
        "PaliGemma": {
            "img": {
                "pos_embedding": posemb,
                "head": {"kernel": np.ones((4, projection_width), dtype=np.float32)},
            }
        }
    }


def _load(monkeypatch, source, target, *, resize=True):
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *args, **kwargs: source)  # noqa: SLF001
    return weight_loaders.CheckpointWeightLoader("unused", resize_siglip_posemb=resize).load(target)


@pytest.mark.parametrize("target_side", [16, 24, 32])
def test_posemb_resize_preserves_spatial_axes_and_channels(monkeypatch, target_side):
    y, x = np.mgrid[:16, :16].astype(np.float32)
    grid = np.stack([x, y, np.full_like(x, 7), x + y], axis=-1)
    source = _params(grid.reshape(1, 256, 4))
    target = _params(jax.ShapeDtypeStruct((1, target_side**2, 4), jnp.float32))
    loaded = _load(monkeypatch, source, target)
    at.check_pytree_equality(expected=target, got=loaded, check_shapes=True, check_dtypes=True)
    resized = loaded["PaliGemma"]["img"]["pos_embedding"].reshape(target_side, target_side, 4)
    np.testing.assert_allclose(resized[..., 2], 7, atol=1e-5)
    np.testing.assert_allclose(resized[0, :, 0], resized[-1, :, 0], atol=1e-5)
    np.testing.assert_allclose(resized[:, 0, 1], resized[:, -1, 1], atol=1e-5)
    assert np.all(np.diff(resized[target_side // 2, :, 0]) > 0)
    assert np.all(np.diff(resized[:, target_side // 2, 1]) > 0)
    np.testing.assert_allclose(resized[..., 3], resized[..., 0] + resized[..., 1], atol=1e-5)
    assert loaded["PaliGemma"]["img"]["head"]["kernel"] is source["PaliGemma"]["img"]["head"]["kernel"]
    assert source["PaliGemma"]["img"]["pos_embedding"].shape == (1, 256, 4)


@pytest.mark.parametrize("tokens", [256, 576, 1024])
def test_matching_posemb_is_unchanged(monkeypatch, tokens):
    source = _params(np.arange(tokens * 4, dtype=np.float32).reshape(1, tokens, 4))
    loaded = _load(monkeypatch, source, source)
    assert loaded["PaliGemma"]["img"]["pos_embedding"] is source["PaliGemma"]["img"]["pos_embedding"]


def test_resize_requires_opt_in(monkeypatch):
    source = _params(np.ones((1, 256, 4), dtype=np.float32))
    target = _params(jax.ShapeDtypeStruct((1, 1024, 4), jnp.float32))
    loaded = _load(monkeypatch, source, target, resize=False)
    with pytest.raises(ValueError, match="Shape mismatch"):
        at.check_pytree_equality(expected=target, got=loaded, check_shapes=True)


def test_other_shape_mismatches_are_not_adapted(monkeypatch):
    source = _params(np.ones((1, 256, 4), dtype=np.float32))
    target = _params(jax.ShapeDtypeStruct((1, 1024, 4), jnp.float32), projection_width=8)
    loaded = _load(monkeypatch, source, target)
    with pytest.raises(ValueError, match="Shape mismatch.*kernel"):
        at.check_pytree_equality(expected=target, got=loaded, check_shapes=True)


@pytest.mark.parametrize("shape", [(1, 1000, 4), (1, 1024, 8), (2, 1024, 4)])
def test_invalid_posemb_conversion_fails(monkeypatch, shape):
    source = _params(np.ones((1, 256, 4), dtype=np.float32))
    target = _params(jax.ShapeDtypeStruct(shape, jnp.float32))
    with pytest.raises(ValueError, match="SigLIP"):
        _load(monkeypatch, source, target)


def test_resized_checkpoint_roundtrip(tmp_path):
    source = _params(np.ones((1, 256, 4), dtype=np.float32))
    target = _params(jax.ShapeDtypeStruct((1, 1024, 4), jnp.float32))
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(tmp_path / "base", {"params": source})
        converted = weight_loaders.CheckpointWeightLoader(str(tmp_path / "base"), resize_siglip_posemb=True).load(
            target
        )
        checkpointer.save(tmp_path / "high_res", {"params": converted})
    loaded = weight_loaders.CheckpointWeightLoader(str(tmp_path / "high_res")).load(target)
    at.check_pytree_equality(expected=target, got=loaded, check_shapes=True, check_dtypes=True)
    np.testing.assert_array_equal(
        loaded["PaliGemma"]["img"]["pos_embedding"], converted["PaliGemma"]["img"]["pos_embedding"]
    )
