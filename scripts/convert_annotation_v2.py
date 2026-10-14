"""Convert LocateAnything annotation.v2 output into a training sidecar.

The source directory is never modified. The output contains one row per frame,
including frames without a usable box, so joins remain aligned with LeRobot.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _boxes(value):
    result = []
    for box in value or []:
        if isinstance(box, dict):
            values = [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
        else:
            values = list(box)
        if len(values) != 4 or any(v is None for v in values):
            continue
        result.append([float(v) for v in values])
    return result


def convert(source_dir: Path, output_dir: Path) -> dict:
    episodes = source_dir / "episodes"
    if not episodes.is_dir():
        raise FileNotFoundError(f"Missing episodes directory: {episodes}")
    rows = []
    source_files = []
    for episode_dir in sorted(episodes.glob("episode_*")):
        manifest = json.loads((episode_dir / "manifest.json").read_text())
        width, height = manifest["frame_size"]
        camera = manifest["camera"]
        jsonl = episode_dir / "annotations.jsonl"
        if not jsonl.is_file():
            raise FileNotFoundError(jsonl)
        source_files.append(jsonl)
        for line in jsonl.read_text().splitlines():
            record = json.loads(line)
            book = _boxes(record.get("object_boxes"))
            title = _boxes(record.get("text_boxes"))
            accepted = record.get("status") == "matched" and record.get("review_status") == "accepted"
            rows.append(
                {
                    "episode_index": int(record["episode_index"]),
                    "frame_index": int(record["frame_index"]),
                    "index": int(record["index"]),
                    "timestamp": float(record["timestamp"]),
                    "camera": camera,
                    "width": int(width),
                    "height": int(height),
                    "has_annotation": bool(accepted and (book or title)),
                    "has_book_bbox": bool(accepted and book),
                    "has_title_bbox": bool(accepted and title),
                    "book_bboxes": book,
                    "title_bboxes": title,
                    "status": str(record.get("status", "")),
                    "review_status": str(record.get("review_status", "")),
                    "presence_status": str(record.get("presence_status", "")),
                    "text": str(record.get("text", "")),
                    "object_query": str(record.get("object_query", "")),
                    "box_source": str(record.get("box_source") or ""),
                }
            )
    rows.sort(key=lambda row: row["index"])
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_dir / "all_frames.parquet", compression="zstd")
    files = []
    for path in source_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": str(path.relative_to(source_dir)), "size": path.stat().st_size, "sha256": digest})
    manifest = {
        "schema_version": "stu_vla.region_sidecar.v1",
        "source_schema": "annotation.v2",
        "camera": "observation.images.hand_right",
        "source_root": str(source_dir),
        "frame_count": len(rows),
        "columns": table.column_names,
        "files": files,
    }
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "frame_count": len(rows),
        "has_annotation": sum(row["has_annotation"] for row in rows),
        "has_book_bbox": sum(row["has_book_bbox"] for row in rows),
        "has_title_bbox": sum(row["has_title_bbox"] for row in rows),
        "episodes": len({row["episode_index"] for row in rows}),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="annotation.v2 directory containing episodes/")
    parser.add_argument("--output", type=Path, required=True, help="new sidecar directory to create")
    args = parser.parse_args()
    print(json.dumps(convert(args.source.expanduser().resolve(), args.output.expanduser().resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
