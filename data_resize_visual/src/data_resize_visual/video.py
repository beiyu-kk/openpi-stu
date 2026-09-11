from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from data_resize_visual.resize import resize_with_pad


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    frame_count: int | None
    rotation: int

    @property
    def decoded_width(self) -> int:
        return self.height if abs(self.rotation) % 180 == 90 else self.width

    @property
    def decoded_height(self) -> int:
        return self.width if abs(self.rotation) % 180 == 90 else self.height


def require_ffmpeg() -> None:
    missing = [program for program in ("ffmpeg", "ffprobe") if shutil.which(program) is None]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing required executable(s): {names}. Install FFmpeg and try again.")


def probe_video(input_path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        os.fspath(input_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {input_path}")

    stream = streams[0]
    fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate"))
    if fps is None:
        raise ValueError(f"Could not determine frame rate for {input_path}")

    frame_count_text = stream.get("nb_frames")
    frame_count = int(frame_count_text) if frame_count_text and frame_count_text != "N/A" else None
    rotation = int(stream.get("tags", {}).get("rotate", 0))
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = round(float(side_data["rotation"]))
            break

    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frame_count=frame_count,
        rotation=rotation,
    )


def process_video(
    input_path: Path,
    output_path: Path,
    *,
    height: int = 224,
    width: int = 224,
    batch_size: int = 16,
    keep_audio: bool = True,
    lossless_rgb: bool = False,
    overwrite: bool = False,
    quiet: bool = False,
) -> int:
    """Resize every decoded RGB frame and write an H.264 MP4."""
    require_ffmpeg()
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    _validate_arguments(input_path, output_path, height, width, batch_size, lossless_rgb, overwrite)
    info = probe_video(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.part.mp4")
    decode_command = _decoder_command(input_path)
    encode_command = _encoder_command(
        input_path,
        temporary_path,
        info.fps,
        width,
        height,
        keep_audio,
        lossless_rgb,
    )
    frame_size = info.decoded_width * info.decoded_height * 3
    decoder: subprocess.Popen[bytes] | None = None
    encoder: subprocess.Popen[bytes] | None = None
    processed_frames = 0
    started_at = time.monotonic()

    try:
        decoder = subprocess.Popen(decode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert decoder.stdout is not None
        assert encoder.stdin is not None

        while frames := _read_frame_batch(decoder.stdout, frame_size, batch_size):
            batch = np.stack(
                [
                    np.frombuffer(frame, dtype=np.uint8).reshape(info.decoded_height, info.decoded_width, 3)
                    for frame in frames
                ]
            )
            resized = resize_with_pad(batch, height, width)
            encoder.stdin.write(resized.tobytes())
            processed_frames += len(frames)
            if not quiet:
                _show_progress(processed_frames, info.frame_count, started_at)

        decoder.stdout.close()
        decoder_return_code = decoder.wait()
        decoder_error = _read_stderr(decoder)
        if decoder_return_code != 0:
            raise RuntimeError(f"FFmpeg decoder failed:\n{decoder_error}")

        encoder.stdin.close()
        encoder_return_code = encoder.wait()
        encoder_error = _read_stderr(encoder)
        if encoder_return_code != 0:
            raise RuntimeError(f"FFmpeg encoder failed:\n{encoder_error}")
        if processed_frames == 0:
            raise RuntimeError("The input video contained no decodable frames")

        os.replace(temporary_path, output_path)
        if not quiet:
            elapsed = time.monotonic() - started_at
            print(
                f"\rDone: {processed_frames} frames -> {output_path} ({width}x{height}, {elapsed:.1f}s){' ' * 12}",
                file=sys.stderr,
            )
        return processed_frames
    except BaseException:
        _stop_process(decoder)
        _stop_process(encoder)
        temporary_path.unlink(missing_ok=True)
        raise


def _parse_fps(value: str | None) -> Fraction | None:
    if not value:
        return None
    try:
        fps = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return fps if fps > 0 else None


def _validate_arguments(
    input_path: Path,
    output_path: Path,
    height: int,
    width: int,
    batch_size: int,
    lossless_rgb: bool,
    overwrite: bool,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if input_path.suffix.lower() != ".mp4":
        raise ValueError(f"Input must be an MP4 file: {input_path}")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError(f"Output must use the .mp4 extension: {output_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite to replace it): {output_path}")
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive")
    if not lossless_rgb and (height % 2 != 0 or width % 2 != 0):
        raise ValueError("Compatible yuv420p output requires even height and width; use --lossless-rgb for odd sizes")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")


def _decoder_command(input_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        os.fspath(input_path),
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]


def _encoder_command(
    input_path: Path,
    output_path: Path,
    fps: Fraction,
    width: int,
    height: int,
    keep_audio: bool,
    lossless_rgb: bool,
) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps.numerator}/{fps.denominator}",
        "-i",
        "pipe:0",
    ]
    if keep_audio:
        command.extend(["-i", os.fspath(input_path)])
    command.extend(
        [
            "-map",
            "0:v:0",
        ]
    )
    if keep_audio:
        command.extend(["-map", "1:a?", "-c:a", "copy", "-map_metadata", "1"])
    if lossless_rgb:
        command.extend(["-c:v", "libx264rgb", "-crf", "0", "-pix_fmt", "rgb24"])
    else:
        command.extend(
            [
                "-vf",
                "scale=in_range=full:out_range=limited:out_color_matrix=bt709,format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-color_range",
                "tv",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
            ]
        )
    command.extend(
        [
            "-preset",
            "medium",
            "-metadata:s:v:0",
            "rotate=0",
            "-movflags",
            "+faststart",
            os.fspath(output_path),
        ]
    )
    return command


def _read_frame_batch(stream, frame_size: int, batch_size: int) -> list[bytes]:
    frames = []
    for _ in range(batch_size):
        frame = _read_exactly(stream, frame_size)
        if not frame:
            break
        if len(frame) != frame_size:
            raise RuntimeError(f"FFmpeg returned a partial frame ({len(frame)} of {frame_size} bytes)")
        frames.append(frame)
    return frames


def _read_exactly(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_stderr(process: subprocess.Popen[bytes]) -> str:
    if process.stderr is None:
        return ""
    return process.stderr.read().decode(errors="replace").strip()


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _show_progress(processed: int, total: int | None, started_at: float) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-6)
    rate = processed / elapsed
    if total:
        percent = min(processed / total * 100, 100.0)
        status = f"{processed}/{total} frames ({percent:5.1f}%)"
    else:
        status = f"{processed} frames"
    print(f"\rProcessing: {status}, {rate:.1f} fps", end="", file=sys.stderr, flush=True)
