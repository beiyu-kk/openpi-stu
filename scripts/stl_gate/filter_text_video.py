"""Keep frames with DBNet text detections and concatenate them into a silent MP4."""

import argparse
import contextlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import cv2

from openpi.models.stl_gate import DBTextDetector
from openpi.models.stl_gate.detector import DEFAULT_MODEL_DIR


def filter_video(
    input_path: Path,
    output_path: Path,
    detector: DBTextDetector,
    *,
    overwrite: bool = False,
    save_keyframes: bool = False,
) -> dict:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path == output_path or (output_path.exists() and input_path.samefile(output_path)):
        raise ValueError("Input and output must be different files")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite to replace it")
    keyframe_dir = output_path.with_name(f"{output_path.stem}_keyframes")
    if save_keyframes:
        if keyframe_dir.is_symlink() or (keyframe_dir.exists() and not keyframe_dir.is_dir()):
            raise ValueError(f"Keyframe destination must be a regular directory: {keyframe_dir}")
        if input_path.is_relative_to(keyframe_dir):
            raise ValueError("Input video must not be inside the keyframe directory")
        if keyframe_dir.exists() and not overwrite:
            raise FileExistsError(f"Keyframe directory already exists: {keyframe_dir}; use --overwrite to replace it")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg with the libx264 encoder is required")

    capture = cv2.VideoCapture(str(input_path))
    encoder = None
    processed = kept = keyframes_saved = 0
    started = time.perf_counter()
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has an invalid frame rate: {fps}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a fully encoded video; failures leave any existing output intact.
        with tempfile.TemporaryDirectory(prefix=".stl-gate-", dir=output_path.parent) as temporary:
            video_path = Path(temporary) / "filtered.mp4"
            staged_keyframes = Path(temporary) / "keyframes"
            if save_keyframes:
                staged_keyframes.mkdir()
            with (Path(temporary) / "ffmpeg.log").open("w+b") as encoder_log:
                try:
                    frame_size = None
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        height, width = frame.shape[:2]
                        if frame_size is None:
                            frame_size = (width, height)
                        elif frame_size != (width, height):
                            raise ValueError("Video frame dimensions changed during decoding")
                        detection = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        processed += 1
                        if detection.has_text:
                            if encoder is None:
                                command = [
                                    ffmpeg,
                                    "-hide_banner",
                                    "-loglevel",
                                    "error",
                                    "-y",
                                    "-f",
                                    "rawvideo",
                                    "-pix_fmt",
                                    "bgr24",
                                    "-video_size",
                                    f"{width}x{height}",
                                    "-framerate",
                                    str(fps),
                                    "-i",
                                    "pipe:0",
                                    "-an",
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "fast",
                                    "-crf",
                                    "18",
                                    "-threads",
                                    "2",
                                    "-vf",
                                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-movflags",
                                    "+faststart",
                                    str(video_path),
                                ]
                                encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=encoder_log)
                            encoder.stdin.write(frame.tobytes())
                            if save_keyframes and kept >= math.floor(keyframes_saved * fps):
                                # Sample on the filtered timeline and match FFmpeg's even-size padding.
                                still = cv2.copyMakeBorder(
                                    frame, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                                )
                                while kept >= math.floor(keyframes_saved * fps):
                                    image_path = staged_keyframes / f"frame_{keyframes_saved:06d}s.png"
                                    if not cv2.imwrite(str(image_path), still):
                                        raise RuntimeError(f"Cannot save keyframe: {image_path}")
                                    keyframes_saved += 1
                            kept += 1
                        if processed % 100 == 0:
                            print(f"Processed {processed}/{total or '?'} frames; kept {kept}", file=sys.stderr)
                    if processed == 0:
                        raise RuntimeError("No video frames could be decoded")
                    if total > 0 and processed < total:
                        raise RuntimeError(f"Decoding ended early: {processed} of {total} advertised frames")
                    if encoder is not None:
                        encoder.stdin.close()
                        if encoder.wait(timeout=60) != 0:
                            encoder_log.seek(0)
                            raise RuntimeError(encoder_log.read().decode(errors="replace"))
                        if output_path.exists() and not overwrite:
                            raise FileExistsError(f"Output appeared during processing: {output_path}")
                        previous_keyframes = Path(temporary) / "previous_keyframes"
                        if save_keyframes:
                            if keyframe_dir.exists():
                                if not overwrite:
                                    raise FileExistsError(
                                        f"Keyframe directory appeared during processing: {keyframe_dir}"
                                    )
                                keyframe_dir.replace(previous_keyframes)
                            try:
                                staged_keyframes.replace(keyframe_dir)
                            except OSError:
                                if previous_keyframes.exists():
                                    previous_keyframes.replace(keyframe_dir)
                                raise
                        try:
                            video_path.replace(output_path)
                        except OSError:
                            if save_keyframes:
                                keyframe_dir.replace(staged_keyframes)
                                if previous_keyframes.exists():
                                    previous_keyframes.replace(keyframe_dir)
                            raise
                except BrokenPipeError as exc:
                    encoder_log.seek(0)
                    raise RuntimeError(
                        f"FFmpeg encoding failed: {encoder_log.read().decode(errors='replace')}"
                    ) from exc
                finally:
                    if encoder is not None and encoder.poll() is None:
                        encoder.terminate()
                        try:
                            encoder.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            encoder.kill()
                            encoder.wait()
                    if encoder is not None and encoder.stdin is not None and not encoder.stdin.closed:
                        with contextlib.suppress(BrokenPipeError):
                            encoder.stdin.close()
    finally:
        capture.release()

    return {
        "input": str(input_path),
        "output": str(output_path) if kept else None,
        "processed_frames": processed,
        "kept_frames": kept,
        "removed_frames": processed - kept,
        "keyframe_dir": str(keyframe_dir) if keyframes_saved else None,
        "keyframes_saved": keyframes_saved,
        "fps": fps,
        "output_duration_seconds": round(kept / fps, 3),
        "processing_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source video")
    parser.add_argument("--output", type=Path, help="Default: <input_stem>_text_only.mp4")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--limit-side-len", type=int, default=640)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--box-thresh", type=float, help="Text-region score threshold; default: model setting (0.4)")
    parser.add_argument(
        "--save-keyframes",
        action="store_true",
        help="Save PNGs at 0s, 1s, 2s, ... of the output video to <output_stem>_keyframes/ at output resolution",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing video and, if enabled, keyframe directory"
    )
    args = parser.parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}_text_only.mp4")
    with DBTextDetector(
        args.model_dir,
        limit_side_len=args.limit_side_len,
        cpu_threads=args.cpu_threads,
        box_thresh=args.box_thresh,
    ) as detector:
        report = filter_video(
            args.input, output, detector, overwrite=args.overwrite, save_keyframes=args.save_keyframes
        )
    print(json.dumps(report, indent=2))
    if report["kept_frames"] == 0:
        print("No text frames detected; no output video was created or replaced.", file=sys.stderr)


if __name__ == "__main__":
    main()
