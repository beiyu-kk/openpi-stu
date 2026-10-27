"""CPU tokenization for prefix-conditioned FAST RTC training and inference."""

import dataclasses

import numpy as np

from openpi.models.tokenizer import FASTTokenizer


@dataclasses.dataclass(frozen=True)
class TokenizeFASTRTCInputs:
    """Prepare every delay on CPU; the training step samples with its JAX RNG."""

    tokenizer: FASTTokenizer
    action_horizon: int
    max_delay: int

    def __call__(self, data):
        prompt = data.pop("prompt")
        prompt = prompt if isinstance(prompt, str) else prompt.item()
        actions = data.get("actions")
        delay = data.pop("rtc_delay", None)
        names = ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask")
        if actions is None or delay is not None:
            delay = 0 if delay is None else int(delay)
            if not 0 <= delay <= self.max_delay:
                raise ValueError("FAST RTC delay outside trained support")
            if delay and actions is None:
                raise ValueError("FAST RTC delay requires committed actions")
            prefix = np.empty((0, len(data["state"])), np.float32) if actions is None else actions[:delay]
            tokens = self.tokenizer.tokenize_rtc(
                prompt, data["state"], prefix, None, delay=delay, suffix_horizon=self.action_horizon - delay
            )
            return {**data, **dict(zip(names, tokens, strict=True))}
        actions = np.asarray(actions)
        if actions.ndim != 2 or len(actions) != self.action_horizon:
            raise ValueError("FAST RTC actions must be [action_horizon, action_dim]")
        if "action_is_pad" not in data:
            raise ValueError("FAST RTC training requires action_is_pad")
        padding = np.asarray(data["action_is_pad"], dtype=bool)
        valid = int(np.sum(~padding))
        if padding.shape != (self.action_horizon,) or not np.array_equal(
            padding, np.arange(self.action_horizon) >= valid
        ):
            raise ValueError("FAST RTC action padding must be a contiguous episode tail")
        if not np.isfinite(actions[:valid]).all():
            raise ValueError("FAST RTC valid actions must be finite")
        candidates = [
            self.tokenizer.tokenize_rtc(
                prompt,
                data["state"],
                actions[: min(d, valid)],
                actions[d:valid],
                delay=d,
                suffix_horizon=max(valid - d, 0),
            )
            for d in range(self.max_delay + 1)
        ]
        bank = {name: np.stack([candidate[i] for candidate in candidates]) for i, name in enumerate(names)}
        return {**data, **{name: value[0] for name, value in bank.items()}, "rtc_token_inputs": bank}


@dataclasses.dataclass(frozen=True)
class ExtractFASTRTCActions:
    tokenizer: FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data):
        delay = int(data.pop("rtc_delay", 0))
        prefix = data.pop("rtc_prefix", np.empty((0, self.action_dim), dtype=np.float32))
        if not 0 <= delay < self.action_horizon or prefix.shape[0] < delay:
            raise ValueError("Invalid FAST RTC decoding prefix")
        suffix = self.tokenizer.extract_rtc_actions(
            np.asarray(data["actions"], dtype=np.int32), self.action_horizon - delay, self.action_dim
        )
        if suffix.shape != (self.action_horizon - delay, self.action_dim) or not np.isfinite(suffix).all():
            raise ValueError("FAST RTC tokenizer returned an invalid decoded suffix")
        if prefix.ndim != 2 or not np.isfinite(prefix).all():
            raise ValueError("FAST RTC prefix must be a finite action matrix")
        return {**data, "actions": np.concatenate([prefix[:delay], suffix], axis=0)}
