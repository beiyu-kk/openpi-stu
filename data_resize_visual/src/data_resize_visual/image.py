from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from data_resize_visual.resize import resize_with_pad

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


def process_image(
    input_path: Path,
    output_path: Path,
    *,
    height: int = 224,
    width: int = 224,
    overwrite: bool = False,
) -> Path:
    """Resize one still image using OpenPI's policy preprocessing algorithm."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")
    if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported input image extension: {input_path.suffix}")
    if output_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported output image extension: {output_path.suffix}")
    if input_path == output_path or (output_path.exists() and input_path.samefile(output_path)):
        raise ValueError("Input and output paths must be different")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite to replace it): {output_path}")
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive")

    with Image.open(input_path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Image input must contain exactly one frame")
        image = ImageOps.exif_transpose(source).convert("RGB")
        resized = resize_with_pad(np.asarray(image), height, width)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.part{output_path.suffix}")
    try:
        Image.fromarray(resized).save(temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path
