from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from data_resize_visual.image import IMAGE_EXTENSIONS, process_image
from data_resize_visual.video import process_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-resize-visual",
        description="Apply OpenPI policy's exact pre-SigLIP resize_with_pad operation to an image or every MP4 frame.",
    )
    parser.add_argument("input", type=Path, help="input MP4 video or PNG/JPEG/WebP/BMP/TIFF image")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path (default: <input>_resized_<width>x<height>.png for images, .mp4 for videos)",
    )
    parser.add_argument("--height", type=int, default=224, help="target height (default: 224)")
    parser.add_argument("--width", type=int, default=224, help="target width (default: 224)")
    parser.add_argument("--batch-size", type=int, default=16, help="video frames processed per batch (default: 16)")
    parser.add_argument("--no-audio", action="store_true", help="do not copy the input audio stream (video only)")
    parser.add_argument(
        "--lossless-rgb",
        action="store_true",
        help="use pixel-exact RGB H.264 for FFmpeg-based analysis (video only; some players display it incorrectly)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace the output if it already exists")
    parser.add_argument("--quiet", action="store_true", help="hide progress output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    is_image = args.input.suffix.lower() in IMAGE_EXTENSIONS
    extension = ".png" if is_image else ".mp4"
    output = args.output or args.input.with_name(f"{args.input.stem}_resized_{args.width}x{args.height}{extension}")
    try:
        if is_image:
            if args.no_audio or args.lossless_rgb or args.batch_size != 16:
                raise ValueError("--no-audio, --lossless-rgb and --batch-size apply only to videos")
            saved_path = process_image(
                args.input,
                output,
                height=args.height,
                width=args.width,
                overwrite=args.overwrite,
            )
            if not args.quiet:
                print(f"Done: {saved_path} ({args.width}x{args.height})", file=sys.stderr)
        elif args.input.suffix.lower() == ".mp4":
            process_video(
                args.input,
                output,
                height=args.height,
                width=args.width,
                batch_size=args.batch_size,
                keep_audio=not args.no_audio,
                lossless_rgb=args.lossless_rgb,
                overwrite=args.overwrite,
                quiet=args.quiet,
            )
        else:
            raise ValueError("Input must be an MP4 video or a PNG/JPEG/WebP/BMP/TIFF image")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
