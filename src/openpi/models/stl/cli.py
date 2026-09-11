#!/usr/bin/env python3
"""Run LocateAnything on one image and save visualized results."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .locator import LocateAnythingWorker
from .tasks import TASKS, TASKS_REQUIRING_QUERY
from .tasks import run_task as dispatch_task
from .visualization import draw_predictions

ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test LocateAnything on a single image."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local model directory or Hugging Face model ID.",
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--task",
        choices=TASKS,
        default="detect",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Categories for --task detect, for example: person car bicycle",
    )
    parser.add_argument(
        "--query", help="Phrase or question used by non-detection tasks."
    )
    parser.add_argument(
        "--generation-mode",
        choices=["fast", "slow", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--device", default="cuda", help="For example: cuda, cuda:0, or cpu."
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--output",
        help="Output image path. Defaults to models/stl/artifacts/outputs/<image-stem>_<task>.png.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print model timing statistics."
    )
    parser.add_argument(
        "--seed", type=int, help="Seed sampling immediately before inference."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--text-attention", choices=["sdpa", "magi"], default="sdpa")
    parser.add_argument(
        "--vision-attention",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        default="auto",
    )
    parser.add_argument(
        "--revision", help="Checkpoint revision when using a Hub model ID."
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.task == "detect" and not args.categories:
        parser.error("--categories is required when --task=detect")
    if args.task in TASKS_REQUIRING_QUERY and not args.query:
        parser.error(f"--query is required when --task={args.task}")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than zero")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.repetition_penalty <= 0:
        parser.error("Invalid sampling parameters")

    return args


def run_task(
    worker: LocateAnythingWorker,
    image: Image.Image,
    args: argparse.Namespace,
) -> dict:
    generation_args = {
        "generation_mode": args.generation_mode,
        "max_new_tokens": args.max_new_tokens,
        "verbose": args.verbose,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }
    return dispatch_task(
        worker,
        image,
        args.task,
        query=args.query,
        categories=args.categories,
        **generation_args,
    )


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check PyTorch/CUDA, or use --device cpu."
        )

    dtype = getattr(torch, args.dtype)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    worker = LocateAnythingWorker(
        args.model_path,
        device=args.device,
        dtype=dtype,
        text_attention=args.text_attention,
        vision_attn=args.vision_attention,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)
    result = run_task(worker, image, args)

    answer = result["answer"]
    boxes = worker.parse_boxes(answer, *image.size)
    points = worker.parse_points(answer, *image.size)

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else ARTIFACT_ROOT / "outputs" / f"{image_path.stem}_{args.task}.png"
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_predictions(image, boxes, points).save(output_path)

    json_path = output_path.with_suffix(".json")
    payload = {
        "model": args.model_path,
        "image": str(image_path),
        "task": args.task,
        "generation_mode": args.generation_mode,
        "answer": answer,
        "boxes": boxes,
        "points": points,
    }
    if "stats" in result:
        payload["stats"] = result["stats"]
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Model answer:\n{answer}")
    print(f"Boxes: {boxes}")
    print(f"Points: {points}")
    print(f"Visualization: {output_path}")
    print(f"JSON result: {json_path}")


if __name__ == "__main__":
    main()
