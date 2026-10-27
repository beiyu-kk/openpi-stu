import dataclasses
import math
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.models.region_guidance import RegionGuidanceConfig
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    # Fixed input size, or position embedding reference size when pad_to_square=True.
    image_resolution: tuple[int, int] = (224, 224)
    # True: pad each image to its longest side without resizing; allow dynamic sizes.
    pad_to_square: bool = False
    # Fixed-canvas processing used only when pad_to_square=False.
    image_resize_mode: Literal["resize_with_pad", "pad_only"] = "resize_with_pad"
    # Optional Qwen-style spatial reduction after SigLIP.  ``none`` preserves
    # the original image-token sequence and all existing checkpoints.
    image_token_merger: Literal["none", "spatial_2x2"] = "none"
    image_token_merge_size: int = 2
    # Static image keys embedded in the Transformer.  The default preserves
    # the three-camera Pi0 layout; Piper's merged experiment can omit its
    # permanently invalid left-wrist slot to avoid paying for masked tokens.
    image_token_camera_keys: tuple[str, ...] = _model.IMAGE_KEYS
    # Disabled by default. No extra parameters or inference inputs are introduced.
    region_guidance: RegionGuidanceConfig | None = None
    # When enabled, train the VLM backbone with FAST discrete action tokens while
    # insulating it from continuous action-expert gradients, as in Knowledge
    # Insulation (Driess et al., 2025).
    knowledge_insulation: bool = False
    discrete_action_loss_weight: float = 1.0

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Runtime settings derived exclusively from TrainConfig.training_rtc.
    simulated_delay: int | None = None
    simulated_delay_weights: tuple[float, ...] | None = None
    rtc_loss_reduction: Literal["official", "per_element"] = "official"
    rtc_time_distribution: Literal["beta", "uniform"] = "beta"
    rtc_finetune_mode: Literal["lora", "full"] = "lora"

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.simulated_delay is not None:
            if not self.pi05:
                raise ValueError("Training RTC supports JAX pi0.5 only")
            if not 1 <= self.simulated_delay <= self.action_horizon:
                raise ValueError("simulated_delay must be in [1, action_horizon]")
            if self.simulated_delay_weights is not None:
                w = self.simulated_delay_weights
                if len(w) != self.simulated_delay or any(not math.isfinite(v) or v < 0 for v in w) or sum(w) <= 0:
                    raise ValueError("RTC weights must be finite, nonnegative, have length S and positive sum")
        elif self.simulated_delay_weights is not None:
            raise ValueError("RTC weights require simulated_delay")
        if self.rtc_loss_reduction not in ("official", "per_element"):
            raise ValueError("Unknown RTC loss reduction")
        if self.rtc_time_distribution not in ("beta", "uniform"):
            raise ValueError("Unknown RTC time distribution")
        if self.rtc_finetune_mode not in ("lora", "full"):
            raise ValueError("Unknown RTC finetune mode")
        if self.rtc_finetune_mode == "full":
            if not self.pi05 or self.simulated_delay is None:
                raise ValueError("Full RTC fine-tuning requires a pi05 RTC model")
            if "lora" in self.paligemma_variant or "lora" in self.action_expert_variant:
                raise ValueError("Full RTC fine-tuning uses base Transformer variants without LoRA adapters")
        if not math.isfinite(self.discrete_action_loss_weight) or self.discrete_action_loss_weight < 0:
            raise ValueError("discrete_action_loss_weight must be finite and nonnegative")
        if self.knowledge_insulation and self.discrete_action_loss_weight <= 0:
            raise ValueError("knowledge_insulation requires a positive discrete_action_loss_weight")
        if (
            len(self.image_resolution) != 2
            or any(type(size) is not int or size <= 0 for size in self.image_resolution)
            or self.image_resolution[0] != self.image_resolution[1]
        ):
            raise ValueError("image_resolution must be square with positive dimensions")
        if type(self.pad_to_square) is not bool:
            raise ValueError("pad_to_square must be a bool")
        if self.image_resize_mode not in ("resize_with_pad", "pad_only"):
            raise ValueError(
                "image_resize_mode must be 'resize_with_pad' or 'pad_only'; use pad_to_square=True for dynamic sizes"
            )
        if self.image_token_merger not in ("none", "spatial_2x2"):
            raise ValueError("image_token_merger must be 'none' or 'spatial_2x2'")
        if type(self.image_token_merge_size) is not int or self.image_token_merge_size < 1:
            raise ValueError("image_token_merge_size must be a positive integer")
        if self.image_token_merger == "spatial_2x2" and self.image_token_merge_size != 2:
            raise ValueError("spatial_2x2 requires image_token_merge_size=2")
        if self.pad_to_square and self.image_token_merger != "none":
            raise ValueError("pad_to_square currently requires image_token_merger='none'")
        patch_side = (self.image_resolution[0] + 13) // 14
        if self.image_token_merger == "spatial_2x2" and patch_side % self.image_token_merge_size:
            raise ValueError("image patch grid must be divisible by image_token_merge_size")
        if self.image_token_merger == "none" and self.image_token_merge_size != 2:
            raise ValueError("image_token_merge_size is only configurable when image_token_merger='spatial_2x2'")
        if not self.image_token_camera_keys or len(set(self.image_token_camera_keys)) != len(
            self.image_token_camera_keys
        ):
            raise ValueError("image_token_camera_keys must contain distinct camera keys")
        if any(key not in _model.IMAGE_KEYS for key in self.image_token_camera_keys):
            raise ValueError("image_token_camera_keys contains an unknown camera key")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.region_guidance is not None:
            backbone = _gemma.get_config(self.paligemma_variant)
            self.region_guidance.layer_indices(backbone.depth)
            self.region_guidance.head_indices(backbone.num_heads)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *self.image_resolution, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32)
                    if self.knowledge_insulation
                    else None
                ),
                token_loss_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_)
                    if self.knowledge_insulation
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def get_rtc_freeze_filter(self) -> nnx.filterlib.Filter:
        """Resolve the explicitly selected full or action-side LoRA RTC training mode."""
        if self.rtc_finetune_mode == "full":
            return nnx.Nothing
        if not self.pi05 or "lora" not in self.action_expert_variant:
            raise ValueError("RTC adaptation requires pi05 and an action expert LoRA variant")
        # _1 identifies the action expert in the two-expert Gemma module.
        trainable = nnx.Any(
            nnx_utils.PathRegex(r"PaliGemma/llm/.*_1/.*lora.*"),
            nnx_utils.PathRegex(
                r"PaliGemma/llm/(layers/(pre_attention_norm_1|pre_ffw_norm_1)|final_norm_1)/Dense_0/(kernel|bias)"
            ),
            nnx_utils.PathRegex(r"(time_mlp_in|time_mlp_out|action_in_proj|action_out_proj)/(kernel|bias)"),
        )
        return nnx.All(nnx.Param, nnx.Not(trainable))
