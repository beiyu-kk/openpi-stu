import dataclasses
import logging
import math
from typing import Any, Literal

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import training_rtc as rtc
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # Due to vmap, this is operating in a single example (not batch level).
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@dataclasses.dataclass(frozen=True)
class Pi0FASTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # Fixed input size, or position embedding reference size when pad_to_square=True.
    image_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION
    # True: pad each image to its longest side without resizing; allow dynamic sizes.
    pad_to_square: bool = False
    # Fixed-canvas processing used only when pad_to_square=False.
    image_resize_mode: Literal["resize_with_pad", "pad_only"] = "resize_with_pad"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250

    # Tokenizer for the fast model.
    fast_model_tokenizer: Any | None = None
    # Keyword arguments for the fast model tokenizer.
    fast_model_tokenizer_kwargs: dict[str, Any] | None = None
    # Derived from TrainConfig.training_rtc. Delay counts action frames, never FAST tokens.
    simulated_delay: int | None = None
    simulated_delay_weights: tuple[float, ...] | None = None
    rtc_finetune_mode: Literal["lora", "full"] = "full"

    def __post_init__(self):
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
        if self.simulated_delay is not None:
            if type(self.simulated_delay) is not int or not 1 <= self.simulated_delay <= self.action_horizon:
                raise ValueError("simulated_delay must be in [1, action_horizon]")
            weights = self.simulated_delay_weights
            if weights is not None and (
                len(weights) != self.simulated_delay
                or any(not math.isfinite(w) or w < 0 for w in weights)
                or sum(weights) <= 0
            ):
                raise ValueError("Invalid FAST RTC delay weights")
        elif self.simulated_delay_weights is not None:
            raise ValueError("RTC weights require simulated_delay")
        if self.rtc_finetune_mode not in ("lora", "full"):
            raise ValueError("Unknown RTC finetune mode")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FAST":
        return Pi0FAST(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *self.image_resolution, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "base_1_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "base_1_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                rtc_token_inputs=(
                    {
                        name: jax.ShapeDtypeStruct([batch_size, self.simulated_delay, self.max_token_len], dtype)
                        for name, dtype in (
                            ("tokenized_prompt", jnp.int32),
                            ("tokenized_prompt_mask", jnp.bool_),
                            ("token_ar_mask", jnp.int32),
                            ("token_loss_mask", jnp.bool_),
                        )
                    }
                    if self.simulated_delay is not None
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing

    def get_rtc_freeze_filter(self) -> nnx.filterlib.Filter:
        if self.rtc_finetune_mode == "full":
            return nnx.Nothing
        if "lora" not in self.paligemma_variant:
            raise ValueError("FAST RTC LoRA requires a PaliGemma LoRA variant")
        # FAST has one autoregressive backbone; RTC LoRA trains only its adapters.
        return nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*lora.*")))


class Pi0FAST(_model.BaseModel):
    def __init__(self, config: Pi0FASTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.image_resolution = config.image_resolution
        self.pad_to_square = config.pad_to_square
        self.image_resize_mode = config.image_resize_mode
        self.simulated_delay = config.simulated_delay
        self.simulated_delay_weights = config.simulated_delay_weights
        self.rtc_action_tokens = True
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
                **(
                    {"posemb_grid": tuple((side + 13) // 14 for side in config.image_resolution)}
                    if config.pad_to_square
                    else {}
                ),
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

    @at.typecheck
    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        token_embeddings = []
        # embed images
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)

            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            # image tokens attend to each other --> AR mask = 0
            ar_mask.append(0 * input_mask[-1])

        # add tokenized inputs
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        # return embeddings, input mask, and ar mask
        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.simulated_delay is not None:
            loss, _ = self.rtc_loss(rng, observation, train=train)
            return jnp.broadcast_to(loss, observation.state.shape[:-1])
        errors, mask, _ = self.token_errors(rng, observation, train=train)
        return jnp.sum(errors, axis=-1) / jnp.maximum(jnp.sum(mask, axis=-1), 1)

    def token_errors(self, rng, observation, *, train=False):
        observation = _model.preprocess_observation(
            rng,
            observation,
            train=train,
            image_keys=list(observation.images.keys()),
            image_resolution=self.image_resolution,
            pad_to_square=self.pad_to_square,
            image_resize_mode=self.image_resize_mode,
        )

        # Compute inputs: one big forward pass of prefix + suffix at once
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # Predict the next token. Integer targets avoid allocating another
        # [batch, token_length, vocabulary] tensor for the longer RTC context.
        targets = observation.tokenized_prompt[:, 1:]

        # Each input predicts *next* token, so we don't input the last token.
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        # Only decode logits for the target tokens to save memory
        # (decoding matmul is large because it is a seq_len x vocab_size dense layer).
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1] :],
        )
        logp = jax.nn.log_softmax(logits, axis=-1)

        # Compute CE loss on token targets
        assert observation.token_loss_mask is not None, "Token loss mask is required"
        loss_mask = observation.token_loss_mask[:, 1:] & observation.tokenized_prompt_mask[:, 1:]
        token_pplx = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        errors = jnp.where(loss_mask, -token_pplx, 0.0)
        correct = (jnp.argmax(logits, axis=-1) == observation.tokenized_prompt[:, 1:]) & loss_mask
        return errors, loss_mask, correct

    def select_rtc_delay(self, observation, delay):
        if self.simulated_delay is None:
            raise ValueError("FAST RTC requires simulated_delay")
        bank = observation.rtc_token_inputs
        if bank is None:
            raise ValueError("FAST RTC requires independently encoded prefix/suffix candidates")
        batch_size = observation.state.shape[0]
        delay = jnp.broadcast_to(jnp.asarray(delay, jnp.int32), (batch_size,))
        values = {}
        for name in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
            if bank[name].shape != (batch_size, self.simulated_delay, self.max_token_len):
                raise ValueError("FAST RTC token bank shape does not match the model")
            values[name] = bank[name][jnp.arange(batch_size), delay]
        return dataclasses.replace(observation, **values, rtc_token_inputs=None)

    def rtc_loss(self, rng, observation, *, train=False):
        if self.simulated_delay is None:
            raise ValueError("FAST RTC requires simulated_delay")
        delay_rng, image_rng = jax.random.split(rng)
        delay = rtc.draw_delay(
            delay_rng, observation.state.shape[:-1], self.simulated_delay, self.simulated_delay_weights
        )
        errors, active, _ = self.token_errors(image_rng, self.select_rtc_delay(observation, delay), train=train)
        count = jnp.sum(active)
        loss = jnp.sum(errors) / jnp.maximum(count, 1)
        return loss, {"rtc/active_tokens": count, "rtc/token_ce": loss}

    def rtc_validation_sums(self, rng, observation, actions, *, delay, raw_action_dim):
        errors, active, correct = self.token_errors(rng, self.select_rtc_delay(observation, delay), train=False)
        return {
            "negative_log_likelihood": jnp.sum(errors),
            "active_tokens": jnp.sum(active),
            "correct_tokens": jnp.sum(correct),
        }

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:
        # TODO: this is a hack to get the image keys.
        observation = _model.preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=list(observation.images.keys()),
            image_resolution=self.image_resolution,
            pad_to_square=self.pad_to_square,
            image_resize_mode=self.image_resize_mode,
        )

        # embed inputs
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # left to right align all input token sequences
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # first fill KV cache with a forward pass of the prefix
        # pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, _ = self.PaliGemma.llm(
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )

        # prepare decoding -- final logit decodes the first token
        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry

            # Sample token from last logit
            # Split RNG for this step
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping --> stop if all batch elements have EOS token
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )

            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        _, _, output_tokens, _, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )
        return output_tokens
