import dataclasses
import logging
import math
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import jax
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Adapt only the learned SigLIP spatial embedding when changing image resolution.
    resize_siglip_posemb: bool = False

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(
            loaded_params, params, missing_regex=".*lora.*", resize_siglip_posemb=self.resize_siglip_posemb
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _resize_siglip_posemb(posemb: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if len(posemb.shape) != 3 or len(target_shape) != 3 or posemb.shape[0] != 1 or target_shape[0] != 1:
        raise ValueError(
            f"Expected SigLIP position embeddings shaped [1, patches, width]: {posemb.shape}, {target_shape}"
        )
    if posemb.shape[-1] != target_shape[-1]:
        raise ValueError(f"Cannot change SigLIP embedding width: {posemb.shape} -> {target_shape}")
    source_side, target_side = math.isqrt(posemb.shape[1]), math.isqrt(target_shape[1])
    if source_side == 0 or target_side == 0 or source_side**2 != posemb.shape[1] or target_side**2 != target_shape[1]:
        raise ValueError(f"SigLIP position embeddings must represent square grids: {posemb.shape} -> {target_shape}")

    logger.info("Resizing SigLIP position embedding from %s to %s", posemb.shape, target_shape)
    grid = posemb.astype(np.float32).reshape(1, source_side, source_side, posemb.shape[-1])
    # Interpolate spatial axes only, on CPU before the training code shards the weights.
    with jax.default_device(jax.devices("cpu")[0]):
        resized = jax.image.resize(grid, (1, target_side, target_side, posemb.shape[-1]), method="cubic")
        return np.asarray(resized).reshape(target_shape).astype(posemb.dtype)


def _merge_params(
    loaded_params: at.Params, params: at.Params, *, missing_regex: str, resize_siglip_posemb: bool = False
) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.
        resize_siglip_posemb: Interpolate the SigLIP position embedding to the reference grid if necessary.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            value = v
            if resize_siglip_posemb and k == "PaliGemma/img/pos_embedding" and v.shape != flat_ref[k].shape:
                value = _resize_siglip_posemb(v, flat_ref[k].shape)
            result[k] = value.astype(flat_ref[k].dtype) if value.dtype != flat_ref[k].dtype else value

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
