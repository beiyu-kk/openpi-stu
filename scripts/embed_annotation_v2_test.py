import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import embed_annotation_v2 as embed


def _make_inputs(tmp_path: Path, *, bad_index: bool = False) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    annotations = tmp_path / "annotations"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data/chunk-000").mkdir(parents=True)
    (annotations / "episodes/episode_000000").mkdir(parents=True)
    info = {
        "total_episodes": 1,
        "total_frames": 2,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (dataset / "meta/info.json").write_text(json.dumps(info))
    table = pa.table(
        {
            "timestamp": pa.array([0.0, 1 / 30], pa.float32()),
            "frame_index": pa.array([0, 1], pa.int64()),
            "episode_index": pa.array([0, 0], pa.int64()),
            "index": pa.array([0, 1], pa.int64()),
        }
    )
    pq.write_table(table, dataset / "data/chunk-000/episode_000000.parquet")
    records = [
        {
            "episode_index": 0,
            "frame_index": 0,
            "index": 9 if bad_index else 0,
            "timestamp": 0.0,
            "status": "matched",
            "review_status": "accepted",
            "object_boxes": [{"x1": 1, "y1": 2, "x2": 11, "y2": 22}],
            "text_boxes": [{"x1": 3, "y1": 4, "x2": 8, "y2": 9}],
        },
        {
            "episode_index": 0,
            "frame_index": 1,
            "index": 1,
            "timestamp": 1 / 30,
            "status": "skipped_unconfirmed",
            "review_status": "rejected",
            "object_boxes": [{"x1": 1, "y1": 2, "x2": 11, "y2": 22}],
            "text_boxes": [],
        },
    ]
    (annotations / "episodes/episode_000000/annotations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    return dataset, annotations


def test_embed_minimal_fields_and_backup(tmp_path):
    dataset, annotations = _make_inputs(tmp_path)
    backup = tmp_path / "backup"
    summary = embed.embed(dataset, annotations, backup)

    assert summary == {
        "frames": 2,
        "book_boxes": 1,
        "title_boxes": 1,
        "episodes": 1,
        "backup_dir": str(backup.resolve()),
    }
    output = pq.read_table(dataset / "data/chunk-000/episode_000000.parquet")
    assert output[embed.BOOK_BBOX_KEY].to_pylist() == [[1.0, 2.0, 11.0, 22.0], [-1.0] * 4]
    assert output[embed.TITLE_BBOX_KEY].to_pylist() == [[3.0, 4.0, 8.0, 9.0], [-1.0] * 4]
    info = json.loads((dataset / "meta/info.json").read_text())
    assert info["features"][embed.BOOK_BBOX_KEY]["shape"] == [4]
    assert embed.BOOK_BBOX_KEY not in pq.read_schema(backup / "data/chunk-000/episode_000000.parquet").names


def test_identity_mismatch_aborts_before_dataset_changes(tmp_path):
    dataset, annotations = _make_inputs(tmp_path, bad_index=True)
    original = (dataset / "data/chunk-000/episode_000000.parquet").read_bytes()
    with pytest.raises(ValueError, match="identity mismatch"):
        embed.embed(dataset, annotations, tmp_path / "backup")
    assert (dataset / "data/chunk-000/episode_000000.parquet").read_bytes() == original
    assert not (tmp_path / "backup").exists()
