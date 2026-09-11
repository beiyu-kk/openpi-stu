"""Detect scene text and measure warm CPU latency using the OpenPI environment."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from openpi.models.stl_gate import DBTextDetector
from openpi.models.stl_gate.detector import DEFAULT_MODEL_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--limit-side-len", type=int, default=640)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--box-thresh", type=float)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/openpi/models/stl_gate/artifacts",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("warmup must be nonnegative and runs must be positive")
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert("RGB"))
    started = time.perf_counter()
    with DBTextDetector(
        args.model_dir,
        limit_side_len=args.limit_side_len,
        cpu_threads=args.cpu_threads,
        box_thresh=args.box_thresh,
    ) as detector:
        init_ms = (time.perf_counter() - started) * 1000
        for _ in range(args.warmup):
            detector.detect(rgb)
        durations = []
        for _ in range(args.runs):
            started = time.perf_counter()
            result = detector.detect(rgb)
            durations.append((time.perf_counter() - started) * 1000)
    report = {
        "image": str(args.image.resolve()),
        "model_dir": str(args.model_dir.expanduser().resolve()),
        "image_size": [rgb.shape[1], rgb.shape[0]],
        "limit_side_len": args.limit_side_len,
        "cpu_threads": args.cpu_threads,
        "warmup": args.warmup,
        "runs": args.runs,
        "has_text": result.has_text,
        "dt_polys": result.polygons.tolist(),
        "dt_scores": result.scores.tolist(),
        "timing_ms": {
            "init": round(init_ms, 2),
            "mean": round(float(np.mean(durations)), 2),
            "p50": round(float(np.percentile(durations, 50)), 2),
            "p95": round(float(np.percentile(durations, 95)), 2),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{args.image.stem}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    overlay = rgb.copy()
    for polygon, score in zip(result.polygons, result.scores, strict=True):
        cv2.polylines(overlay, [polygon], isClosed=True, color=(0, 210, 70), thickness=2)
        anchor = tuple(polygon[0].tolist())
        cv2.putText(overlay, f"{score:.2f}", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 40, 40), 1)
    Image.fromarray(overlay).save(args.output_dir / f"{args.image.stem}.png")
    print(json.dumps(report, indent=2))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
