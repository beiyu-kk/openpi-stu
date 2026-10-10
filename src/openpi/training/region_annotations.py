"""Build training masks from book/title boxes embedded in LeRobot samples."""

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = {
    "base_0_rgb": "observation.images.top_head",
    "right_wrist_0_rgb": "observation.images.hand_right",
}
RegionMode = Literal["none", "book", "title", "both"]
BOOK_BBOX_KEY = "observation.region.hand_right.book_bbox_xyxy"
TITLE_BBOX_KEY = "observation.region.hand_right.title_bbox_xyxy"
EMBEDDED_KEYS = (BOOK_BBOX_KEY, TITLE_BBOX_KEY)


def validate_embedded_annotations(dataset_root: str | Path) -> int:
    """Validate the declared schema and every episode parquet without decoding videos."""
    dataset_root = Path(dataset_root)
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    expected = {"dtype": "float32", "shape": [4], "names": ["x1", "y1", "x2", "y2"]}
    for key in EMBEDDED_KEYS:
        if info.get("features", {}).get(key) != expected:
            raise ValueError(f"Missing or incompatible embedded annotation feature: {key}")
    paths = sorted((dataset_root / "data").rglob("episode_*.parquet"))
    if len(paths) != int(info["total_episodes"]):
        raise ValueError("Embedded annotation episode count does not match meta/info.json")
    frames = 0
    for path in paths:
        schema = pq.read_schema(path)
        missing = [key for key in EMBEDDED_KEYS if key not in schema.names]
        if missing:
            raise ValueError(f"Missing embedded annotation columns in {path}: {missing}")
        for key in EMBEDDED_KEYS:
            field_type = schema.field(key).type
            if (
                not pa.types.is_fixed_size_list(field_type)
                or field_type.list_size != 4
                or field_type.value_type != pa.float32()
            ):
                raise ValueError(f"Embedded annotation column {key} is not float32[4] in {path}")
        frames += pq.ParquetFile(path).metadata.num_rows
    if frames != int(info["total_frames"]):
        raise ValueError("Embedded annotation frame count does not match meta/info.json")
    return len(paths)


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Match ``ResizeImages`` geometry, retaining fractional mask coverage."""
    source_height, source_width = mask.shape
    if (source_height, source_width) == (height, width):
        return mask.astype(np.float32)
    ratio = max(source_width / width, source_height / height)
    resized_height, resized_width = int(source_height / ratio), int(source_width / ratio)
    resized = Image.fromarray(mask.astype(np.float32)).resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    result = np.zeros((height, width), dtype=np.float32)
    top, left = (height - resized_height) // 2, (width - resized_width) // 2
    result[top : top + resized_height, left : left + resized_width] = np.asarray(resized)
    return result


def _image_shape(image: np.ndarray) -> tuple[int, int]:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    return image.shape[1:] if image.shape[0] == 3 else image.shape[:2]


def _valid_boxes(value, width: int, height: int) -> list[list[float]]:
    result = []
    value = np.asarray(value) if value is not None else np.empty((0, 4))
    if value.shape == (4,):
        value = value[None, :]
    for candidate in value:
        box = candidate
        if isinstance(candidate, dict):
            box = [candidate.get("x1"), candidate.get("y1"), candidate.get("x2"), candidate.get("y2")]
        if len(box) != 4 or any(v is None or not np.isfinite(float(v)) for v in box):
            continue
        x1, y1, x2, y2 = map(float, box)
        x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
        y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
        if x2 > x1 and y2 > y1:
            result.append([x1, y1, x2, y2])
    return result


def boxes_to_mask(boxes: list[list[float]], height: int, width: int) -> np.ndarray:
    """Rasterize xyxy boxes as their union in an HWC image's ``(H, W)`` plane."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        polygon = np.rint([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask


def _box_collection(value) -> list:
    if value is None:
        return []
    array = np.asarray(value)
    if array.shape == (4,):
        return [array]
    return list(value)


def _selected_boxes(row: dict, mode: RegionMode) -> list:
    if mode == "none":
        return []
    if mode == "book":
        return _box_collection(row.get("book_bboxes"))
    if mode == "title":
        return _box_collection(row.get("title_bboxes"))
    if mode == "both":
        return _box_collection(row.get("book_bboxes")) + _box_collection(row.get("title_bboxes"))
    raise ValueError(f"Unknown region annotation mode: {mode}")


class RegionAnnotatedDataset:
    """Attach masks from bbox fields already present in each source sample."""

    def __init__(self, dataset, transform, mode: RegionMode = "book"):
        self._dataset = dataset
        self._transform = transform
        self._mode = mode

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = self._dataset[index]
        missing = [name for name in EMBEDDED_KEYS if name not in sample]
        if missing:
            raise ValueError(f"Missing embedded annotation fields: {missing}")
        row = {
            "book_bboxes": sample[BOOK_BBOX_KEY],
            "title_bboxes": sample[TITLE_BBOX_KEY],
        }

        regions, valid = {}, {}
        source_camera = CAMERAS["right_wrist_0_rgb"]
        source_height, source_width = _image_shape(np.asarray(sample[source_camera]))
        boxes = _valid_boxes(_selected_boxes(row, self._mode), source_width, source_height)
        selected_valid = bool(boxes)

        for model_camera, source_name in CAMERAS.items():
            image_shape = _image_shape(np.asarray(sample[source_name]))
            if model_camera == "right_wrist_0_rgb" and selected_valid:
                regions[model_camera] = boxes_to_mask(boxes, *image_shape)
                valid[model_camera] = np.bool_(1)
            else:
                regions[model_camera] = np.zeros(image_shape, dtype=np.uint8)
                valid[model_camera] = np.bool_(0)

        result = self._transform(sample)
        result["region_masks"] = {}
        result["region_valid"] = {}
        for camera, image in result["image"].items():
            height, width = image.shape[:2]
            if camera in regions:
                result["region_masks"][camera] = resize_mask(regions[camera], height, width)
                result["region_valid"][camera] = valid[camera]
            else:
                result["region_masks"][camera] = np.zeros((height, width), dtype=np.float32)
                result["region_valid"][camera] = np.bool_(0)
        return result
