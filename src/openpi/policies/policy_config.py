import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    from openpi.training import rtc_manifest

    manifest = None
    if (checkpoint_dir / "assets" / "rtc_manifest.json").exists():
        manifest = rtc_manifest.read(checkpoint_dir)
        train_config = rtc_manifest.restore_config(train_config, manifest, checkpoint_dir)
    elif train_config.training_rtc is not None:
        raise ValueError("RTC policy requires a checkpoint with rtc_manifest.json")
    if manifest is not None and is_pytorch:
        raise ValueError("Training RTC supports JAX only")
    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(
            _model.restore_params(checkpoint_dir / "params", dtype=None if manifest is not None else jnp.bfloat16)
        )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    policy_cls = _policy.Policy
    rtc_kwargs = {}
    if manifest is not None:
        from openpi.policies.rtc_policy import TrainingRTCPolicy

        if rtc_manifest.stats_digest(norm_stats) != manifest["norm_stats_sha256"]:
            raise ValueError("RTC normalization differs from the checkpoint")
        if data_config.use_quantile_norm != manifest["use_quantile_norm"]:
            raise ValueError("RTC normalization mode differs from the checkpoint")
        if (
            manifest["raw_action_dim"] != 7
            or manifest["delta_mask"] != [True] * 7
            or not train_config.data.use_delta_actions
        ):
            raise ValueError("RTC policy requires the training all-delta Piper convention")
        policy_cls = TrainingRTCPolicy
        rtc_kwargs["raw_action_dim"] = 7
    return policy_cls(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata={
            **(train_config.policy_metadata or {}),
            **(
                {
                    "training_rtc": True,
                    "max_delay": train_config.training_rtc.max_delay,
                    "wire_action_space": "absolute",
                    "raw_action_dim": 7,
                    "action_horizon": train_config.model.action_horizon,
                }
                if manifest is not None
                else {}
            ),
        },
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
        **rtc_kwargs,
    )
