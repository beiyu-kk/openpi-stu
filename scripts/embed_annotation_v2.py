"""Embed the minimal accepted annotation.v2 boxes into a LeRobot v2 dataset.

Only two fixed-size float32 xyxy fields are added. Rejected or unannotated
frames use ``[-1, -1, -1, -1]``. The command stages and validates every
episode before replacing any dataset file, and keeps a small rollback backup.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BOOK_BBOX_KEY = "observation.region.hand_right.book_bbox_xyxy"
TITLE_BBOX_KEY = "observation.region.hand_right.title_bbox_xyxy"
ANNOTATION_KEYS = (BOOK_BBOX_KEY, TITLE_BBOX_KEY)
INVALID_BBOX = np.full(4, -1, dtype=np.float32)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _accepted_box(record: dict, source_key: str) -> np.ndarray:
    if record.get("status") != "matched" or record.get("review_status") != "accepted":
        return INVALID_BBOX.copy()
    boxes = record.get(source_key) or []
    if len(boxes) != 1:
        if not boxes:
            return INVALID_BBOX.copy()
        raise ValueError(
            f"Expected at most one {source_key} box at episode={record.get('episode_index')} "
            f"frame={record.get('frame_index')}, got {len(boxes)}"
        )
    box = boxes[0]
    if isinstance(box, dict):
        box = [box.get(name) for name in ("x1", "y1", "x2", "y2")]
    values = np.asarray(box, dtype=np.float32)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {source_key} box: {box!r}")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate {source_key} box: {box!r}")
    return values


def _bbox_array(values: list[np.ndarray]) -> pa.Array:
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, 4)


def _episode_output(table: pa.Table, records: list[dict], episode_index: int) -> pa.Table:
    if table.num_rows != len(records):
        raise ValueError(
            f"Episode {episode_index}: data has {table.num_rows} rows but annotations have {len(records)}"
        )
    expected_frame = table["frame_index"].to_numpy()
    expected_episode = table["episode_index"].to_numpy()
    expected_index = table["index"].to_numpy()
    expected_timestamp = table["timestamp"].to_numpy()
    book, title = [], []
    for offset, record in enumerate(records):
        identity = (record["episode_index"], record["frame_index"], record["index"])
        expected = (int(expected_episode[offset]), int(expected_frame[offset]), int(expected_index[offset]))
        if identity != expected:
            raise ValueError(f"Episode {episode_index}: frame identity mismatch {identity} != {expected}")
        if not np.isclose(float(record["timestamp"]), float(expected_timestamp[offset]), rtol=0, atol=1e-4):
            raise ValueError(f"Episode {episode_index} frame {offset}: timestamp mismatch")
        book.append(_accepted_box(record, "object_boxes"))
        title.append(_accepted_box(record, "text_boxes"))

    for key in ANNOTATION_KEYS:
        if key in table.column_names:
            table = table.drop([key])
    return table.append_column(BOOK_BBOX_KEY, _bbox_array(book)).append_column(
        TITLE_BBOX_KEY, _bbox_array(title)
    )


def _feature() -> dict:
    return {"dtype": "float32", "shape": [4], "names": ["x1", "y1", "x2", "y2"]}


def _updated_info(info: dict) -> dict:
    result = json.loads(json.dumps(info))
    features = result.setdefault("features", {})
    for key in ANNOTATION_KEYS:
        existing = features.get(key)
        if existing is not None and existing != _feature():
            raise ValueError(f"Dataset feature {key!r} exists with an incompatible schema: {existing}")
        features[key] = _feature()
    return result


def _verify_staged(data_dir: Path, info_path: Path, expected_frames: int, expected_episodes: int) -> None:
    info = json.loads(info_path.read_text())
    if not all(key in info["features"] for key in ANNOTATION_KEYS):
        raise ValueError("Staged info.json is missing embedded annotation features")
    files = sorted(data_dir.rglob("episode_*.parquet"))
    if len(files) != expected_episodes:
        raise ValueError(f"Staged episode count mismatch: {len(files)} != {expected_episodes}")
    frames = 0
    for path in files:
        schema = pq.read_schema(path)
        for key in ANNOTATION_KEYS:
            field = schema.field(key)
            if not pa.types.is_fixed_size_list(field.type) or field.type.list_size != 4:
                raise ValueError(f"Invalid staged field {key} in {path}")
            if field.type.value_type != pa.float32():
                raise ValueError(f"Invalid staged dtype for {key} in {path}")
        frames += pq.ParquetFile(path).metadata.num_rows
    if frames != expected_frames:
        raise ValueError(f"Staged frame count mismatch: {frames} != {expected_frames}")


def embed(dataset_root: Path, annotation_root: Path, backup_dir: Path | None = None) -> dict:
    dataset_root = dataset_root.resolve()
    annotation_root = annotation_root.resolve()
    info_path = dataset_root / "meta" / "info.json"
    data_dir = dataset_root / "data"
    if not info_path.is_file() or not data_dir.is_dir():
        raise FileNotFoundError(f"Not a LeRobot dataset: {dataset_root}")
    source_episodes = annotation_root / "episodes"
    if not source_episodes.is_dir():
        raise FileNotFoundError(f"Missing annotation episodes: {source_episodes}")

    info = json.loads(info_path.read_text())
    expected_episodes = int(info["total_episodes"])
    expected_frames = int(info["total_frames"])
    data_template = info["data_path"]
    stage_parent = Path(tempfile.mkdtemp(prefix=f".{dataset_root.name}.region-stage-", dir=dataset_root.parent))
    stage_data = stage_parent / "data"
    stage_info = stage_parent / "info.json"
    counts = {"frames": 0, "book_boxes": 0, "title_boxes": 0}
    try:
        for episode_index in range(expected_episodes):
            chunk = episode_index // int(info["chunks_size"])
            relative = Path(data_template.format(episode_chunk=chunk, episode_index=episode_index))
            source_data = dataset_root / relative
            source_jsonl = source_episodes / f"episode_{episode_index:06d}" / "annotations.jsonl"
            if not source_data.is_file() or not source_jsonl.is_file():
                raise FileNotFoundError(f"Missing episode input: {source_data} or {source_jsonl}")
            records = _read_jsonl(source_jsonl)
            output = _episode_output(pq.read_table(source_data), records, episode_index)
            staged_path = stage_parent / relative
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(output, staged_path, compression="zstd")
            counts["frames"] += len(records)
            counts["book_boxes"] += sum(
                not np.array_equal(_accepted_box(r, "object_boxes"), INVALID_BBOX) for r in records
            )
            counts["title_boxes"] += sum(
                not np.array_equal(_accepted_box(r, "text_boxes"), INVALID_BBOX) for r in records
            )

        stage_info.write_text(json.dumps(_updated_info(info), ensure_ascii=False, indent=2) + "\n")
        _verify_staged(stage_data, stage_info, expected_frames, expected_episodes)

        if backup_dir is None:
            stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
            backup_dir = dataset_root.parent / f"{dataset_root.name}.region_backup_{stamp}"
        backup_dir = backup_dir.resolve()
        if backup_dir.exists():
            raise FileExistsError(f"Backup directory already exists: {backup_dir}")
        (backup_dir / "meta").mkdir(parents=True)
        shutil.copy2(info_path, backup_dir / "meta" / "info.json")
        for episode_index in range(expected_episodes):
            chunk = episode_index // int(info["chunks_size"])
            relative = Path(data_template.format(episode_chunk=chunk, episode_index=episode_index))
            target = backup_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dataset_root / relative, target)
        (backup_dir / "restore_manifest.json").write_text(
            json.dumps({"dataset_root": str(dataset_root), "files": expected_episodes + 1}, indent=2) + "\n"
        )

        replaced = []
        try:
            for episode_index in range(expected_episodes):
                chunk = episode_index // int(info["chunks_size"])
                relative = Path(data_template.format(episode_chunk=chunk, episode_index=episode_index))
                os.replace(stage_parent / relative, dataset_root / relative)
                replaced.append(relative)
            os.replace(stage_info, info_path)
        except Exception:
            for relative in replaced:
                shutil.copy2(backup_dir / relative, dataset_root / relative)
            shutil.copy2(backup_dir / "meta" / "info.json", info_path)
            raise
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)

    return {**counts, "episodes": expected_episodes, "backup_dir": str(backup_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot dataset root to update")
    parser.add_argument("--annotations", type=Path, required=True, help="annotation.v2 output root")
    parser.add_argument("--backup-dir", type=Path, help="Optional explicit rollback backup directory")
    args = parser.parse_args()
    result = embed(args.dataset.expanduser(), args.annotations.expanduser(), args.backup_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
