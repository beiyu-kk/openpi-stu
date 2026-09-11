"""Load checkpoint assets using local model and processor implementations."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from transformers.utils import is_flash_attn_2_available


@dataclass
class ModelComponents:
    tokenizer: Any
    processor: Any
    model: Any
    model_path: Path
    loading_info: dict


def resolve_model_path(
    model_path: str, *, revision=None, local_files_only=False
) -> Path:
    path = Path(model_path).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.exists() or path.is_absolute() or model_path.startswith((".", "~")):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_path,
            revision=revision,
            local_files_only=local_files_only,
            cache_dir=Path(__file__).parent / "artifacts" / "hub",
            allow_patterns=[
                "*.json",
                "*.jinja",
                "*.safetensors",
                "*.model",
                "*.txt",
                "*.tiktoken",
                "LICENSE*",
            ],
        )
    )


def _read_config(path: Path, filename: str) -> dict:
    with (path / filename).open(encoding="utf-8") as source:
        result = json.load(source)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path / filename}")
    return result


def load_preprocessing(model_path: Path):
    from .locate_anything.image_processing_locateanything import (
        LocateAnythingImageProcessor,
    )
    from .locate_anything.processing_locateanything import LocateAnythingProcessor

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=False, local_files_only=True
    )
    image_config = _read_config(model_path, "preprocessor_config.json")
    processor_config = _read_config(model_path, "processor_config.json")
    # Auto mappings refer to checkpoint Python files, never to executable assets here.
    for key in ("auto_map", "processor_class", "image_processor_type"):
        image_config.pop(key, None)
        processor_config.pop(key, None)
    template_path = model_path / "chat_template.json"
    if template_path.is_file():
        processor_config["chat_template"] = _read_config(
            model_path, template_path.name
        )["chat_template"]
    elif (model_path / "chat_template.jinja").is_file():
        processor_config["chat_template"] = (
            model_path / "chat_template.jinja"
        ).read_text(encoding="utf-8")
    image_processor = LocateAnythingImageProcessor(**image_config)
    processor = LocateAnythingProcessor(
        image_processor=image_processor, tokenizer=tokenizer, **processor_config
    )
    return tokenizer, processor


def load_components(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    text_attention: str = "sdpa",
    vision_attention: str = "auto",
    revision: str | None = None,
    local_files_only: bool = False,
) -> ModelComponents:
    from .locate_anything.configuration_locateanything import LocateAnythingConfig
    from .locate_anything.modeling_locateanything import (
        LocateAnythingForConditionalGeneration,
    )

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; use device='cpu' and dtype=torch.float32"
        )
    if text_attention not in {"sdpa", "magi"}:
        raise ValueError(f"Unsupported single-image text attention: {text_attention}")
    if vision_attention == "auto":
        vision_attention = (
            "flash_attention_2"
            if is_flash_attn_2_available() and target.type == "cuda"
            else "sdpa"
        )
    if vision_attention not in {"sdpa", "eager", "flash_attention_2"}:
        raise ValueError(f"Unsupported vision attention: {vision_attention}")
    path = resolve_model_path(
        model_path, revision=revision, local_files_only=local_files_only
    )
    tokenizer, processor = load_preprocessing(path)
    config = LocateAnythingConfig.from_pretrained(path, local_files_only=True)
    # In Transformers 4.53 an unset child attention defaults to eager, which
    # LocateAnything's block-mask decoder does not support.
    config._attn_implementation = text_attention
    config.text_config._attn_implementation = text_attention
    config.vision_config._attn_implementation = vision_attention
    model, loading_info = LocateAnythingForConditionalGeneration.from_pretrained(
        path,
        config=config,
        torch_dtype=dtype,
        local_files_only=True,
        output_loading_info=True,
    )
    failures = {
        key: loading_info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if failures:
        raise RuntimeError(
            f"Checkpoint is not compatible with the local model: {failures}"
        )
    model = model.to(target).eval()
    model.requires_grad_(False)
    return ModelComponents(tokenizer, processor, model, path, loading_info)
