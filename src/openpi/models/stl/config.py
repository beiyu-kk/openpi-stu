"""Runtime settings, separate from the checkpoint's architecture configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class STLConfig:
    model_path: str
    device: str = "cuda"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    text_attention: Literal["sdpa", "magi"] = "sdpa"
    vision_attention: Literal["auto", "sdpa", "eager", "flash_attention_2"] = "auto"
    generation_mode: Literal["fast", "slow", "hybrid"] = "hybrid"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.1
    verbose: bool = False
    revision: str | None = None
    local_files_only: bool = False

    def __post_init__(self):
        if not self.model_path:
            raise ValueError("model_path is required")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.text_attention not in {"sdpa", "magi"}:
            raise ValueError(f"Unsupported text attention: {self.text_attention}")
        if self.vision_attention not in {"auto", "sdpa", "eager", "flash_attention_2"}:
            raise ValueError(f"Unsupported vision attention: {self.vision_attention}")
        if self.generation_mode not in {"fast", "slow", "hybrid"}:
            raise ValueError(f"Unsupported generation mode: {self.generation_mode}")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if (
            self.temperature < 0
            or not 0 < self.top_p <= 1
            or self.repetition_penalty <= 0
        ):
            raise ValueError("Invalid sampling parameters")
