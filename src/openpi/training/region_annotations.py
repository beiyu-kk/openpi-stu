"""Read-only sidecar attachment after the standard Piper data transforms."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

CAMERAS = {
    "base_0_rgb": "observation.images.top_head",
    "right_wrist_0_rgb": "observation.images.hand_right",
}


def validate_annotation_source(annotation_dir: str | Path, dataset_root: str | Path) -> int:
    """Reject another dataset with coincident episode/frame numbers before training."""
    annotation_dir, dataset_root = Path(annotation_dir), Path(dataset_root)
    if not (annotation_dir / "all_frames.parquet").is_file():
        raise FileNotFoundError(f"Missing annotation table: {annotation_dir / 'all_frames.parquet'}")
    manifest = json.loads((annotation_dir / "source_manifest.json").read_text())
    files = manifest["files"]
    if not files:
        raise ValueError("Annotation source manifest is empty")
    for entry in files:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Annotation manifest contains a path outside the dataset")
        source = dataset_root / relative
        if not source.is_file() or source.stat().st_size != entry["size"]:
            raise ValueError(f"Annotation source mismatch: {source}")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Annotation source checksum mismatch: {source}")
    return len(files)


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Match ResizeImages' PIL geometry, retaining fractional mask coverage."""
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


class RegionAnnotatedDataset:
    """Preserve existing image/action transforms and add aligned training metadata.

    Polygons reconstruct the verified label masks. This avoids decompressing an
    entire video's NPZ masks for each randomly sampled training frame.
    """

    def __init__(self, dataset, transform, annotation_dir: str):
        self._dataset = dataset
        self._transform = transform
        columns = [
            "episode_index",
            "frame_index",
            "index",
            "timestamp",
            "camera",
            "width",
            "height",
            "use_for_region_loss",
            "label_polygon_xy",
            "label_area_px",
        ]
        rows = pq.read_table(Path(annotation_dir) / "all_frames.parquet", columns=columns).to_pylist()
        self._rows = {}
        for row in rows:
            key = (row["episode_index"], row["frame_index"], row["camera"])
            if key in self._rows:
                raise ValueError(f"Duplicate annotation: {key}")
            self._rows[key] = row

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = self._dataset[index]
        episode = int(np.asarray(sample["episode_index"]).item())
        frame = int(np.asarray(sample["frame_index"]).item())
        global_index = int(np.asarray(sample["index"]).item())
        timestamp = float(np.asarray(sample["timestamp"]).item())
        regions = {}
        valid = {}
        for model_camera, source_camera in CAMERAS.items():
            key = (episode, frame, source_camera)
            if key not in self._rows:
                raise ValueError(f"Missing annotation record: {key}; regenerate sidecars before training")
            row = self._rows[key]
            if row["index"] != global_index or not np.isclose(row["timestamp"], timestamp, rtol=0, atol=1e-4):
                raise ValueError(f"Annotation frame identity mismatch: {key}")
            raw_image = np.asarray(sample[source_camera])
            image_shape = raw_image.shape[1:] if raw_image.shape[0] == 3 else raw_image.shape[:2]
            if image_shape != (row["height"], row["width"]):
                raise ValueError(f"Annotation source image dimensions mismatch: {key}")
            mask = np.zeros(image_shape, dtype=np.uint8)
            is_valid = bool(row["use_for_region_loss"])
            if is_valid:
                polygon = np.asarray(row["label_polygon_xy"], dtype=np.int32)
                if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                    raise ValueError(f"Invalid label polygon: {key}")
                cv2.fillPoly(mask, [polygon], 1)
                if int(mask.sum()) != row["label_area_px"] or not mask.any():
                    raise ValueError(f"Label polygon does not match verified mask area: {key}")
            regions[model_camera] = mask
            valid[model_camera] = np.bool_(is_valid)

        result = self._transform(sample)
        result["region_masks"] = {}
        result["region_valid"] = {}
        for camera, image in result["image"].items():
            height, width = image.shape[:2]
            result["region_masks"][camera] = (
                resize_mask(regions[camera], height, width)
                if camera in regions
                else np.zeros((height, width), dtype=np.float32)
            )
            result["region_valid"][camera] = valid.get(camera, np.False_)
        return result
