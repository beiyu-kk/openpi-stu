# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# Resize/normalization adapted from PaddleX 3.7.2 text_detection/processors.py.
"""PP-OCRv6_tiny_det with the CPU Paddle Inference runtime only."""

import dataclasses
from pathlib import Path

import cv2
import numpy as np
import yaml
import functools

from openpi.models.stl_gate.db_postprocess import decode_boxes

DEFAULT_MODEL_DIR = Path.home() / ".paddlex/official_models/PP-OCRv6_tiny_det"


@dataclasses.dataclass(frozen=True)
class TextDetections:
    polygons: np.ndarray  # (N, 4, 2), int32, original-image pixels
    scores: np.ndarray  # (N,), float32, text-region scores, not target confidence

    @property
    def has_text(self) -> bool:
        return len(self.polygons) > 0


class DBTextDetector:
    """Load once, then call detect with HWC/RGB/uint8 frames sequentially."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        limit_side_len: int = 640,
        cpu_threads: int = 2,
        box_thresh: float | None = None,
    ):
        if not 32 <= limit_side_len <= 4000:
            raise ValueError("limit_side_len must be between 32 and 4000")
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        model_dir = Path(model_dir).expanduser().resolve()
        for name in ("inference.json", "inference.pdiparams", "inference.yml"):
            if not (model_dir / name).is_file():
                raise FileNotFoundError(f"Missing {model_dir / name}; provide the local tiny inference model")
        with (model_dir / "inference.yml").open() as source:
            settings = yaml.safe_load(source)
        if settings["Global"]["model_name"] != "PP-OCRv6_tiny_det":
            raise ValueError("This detector supports PP-OCRv6_tiny_det only")
        post = settings["PostProcess"]
        self._post = {key: post[key] for key in ("thresh", "box_thresh", "unclip_ratio", "max_candidates")}
        if box_thresh is not None:
            self._post["box_thresh"] = box_thresh
        if not 0 < self._post["thresh"] < 1 or not 0 <= self._post["box_thresh"] <= 1:
            raise ValueError("Invalid DB probability thresholds")
        if self._post["unclip_ratio"] <= 0 or self._post["max_candidates"] < 1:
            raise ValueError("Invalid DB contour settings")
        self.limit_side_len = limit_side_len

        # Import only when constructing a detector; importing the package loads no model.
        from paddle import inference

        config = inference.Config(str(model_dir / "inference.json"), str(model_dir / "inference.pdiparams"))
        config.disable_gpu()
        config.enable_mkldnn()
        config.set_mkldnn_cache_capacity(10)
        config.set_cpu_math_library_num_threads(cpu_threads)
        config.enable_new_ir()
        config.enable_new_executor()
        config.set_optimization_level(3)
        config.enable_memory_optim()
        config.disable_glog_info()
        self._predictor = inference.create_predictor(config)
        self._input = self._predictor.get_input_handle(self._predictor.get_input_names()[0])
        self._output = self._predictor.get_output_handle(self._predictor.get_output_names()[0])

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        if height + width < 64:
            padded = np.zeros((max(32, height), max(32, width), 3), dtype=np.uint8)
            padded[:height, :width] = image
            image = padded
            height, width = image.shape[:2]
        ratio = min(1.0, self.limit_side_len / max(height, width))
        resized_h = max(32, round(int(height * ratio) / 32) * 32)
        resized_w = max(32, round(int(width * ratio) / 32) * 32)
        if (resized_h, resized_w) != (height, width):
            image = cv2.resize(image, (resized_w, resized_h))
        planes = []
        for plane, mean, std in zip(cv2.split(image), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), strict=True):
            normalized = plane.astype(np.float32)
            normalized *= (1.0 / 255.0) / std
            normalized += -mean / std
            planes.append(normalized)
        return np.ascontiguousarray(cv2.merge(planes).transpose(2, 0, 1)[None])

    def detect(self, rgb: np.ndarray) -> TextDetections:
        if self._predictor is None:
            raise RuntimeError("Detector is closed")
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Expected a HWC RGB uint8 NumPy image")
        height, width = rgb.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Image must not be empty")
        tensor = self._preprocess(rgb)
        self._input.reshape(tensor.shape)
        self._input.copy_from_cpu(tensor)
        self._predictor.run()
        probability = self._output.copy_to_cpu()
        if probability.ndim != 4 or probability.shape[:2] != (1, 1):
            raise RuntimeError(f"Unexpected DB output shape: {probability.shape}")
        polygons, scores = decode_boxes(probability[0, 0], (width, height), **self._post)
        return TextDetections(polygons, scores)

    def close(self) -> None:
        self._input = self._output = self._predictor = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@functools.lru_cache(maxsize=1)
def get_detector() -> DBTextDetector:
    """Create the detector on first use and reuse it in this process."""
    return DBTextDetector()