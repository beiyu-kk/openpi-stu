"""Reproducible GPU comparison against the original single-image script.

This file can also run directly in the original environment. External source
is imported only for --implementation original, never by the STL runtime.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image


def fingerprint(value):
    if isinstance(value, np.ndarray):
        return fingerprint(torch.from_numpy(value))
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(
                tensor.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    if isinstance(value, (tuple, list)):
        return [fingerprint(item) for item in value]
    return value


def compare_reports(reference, actual):
    differences = []
    for key in ("image_sha256", "seed", "max_new_tokens", "image_size"):
        if reference[key] != actual[key]:
            differences.append(key)
    if reference["cases"].keys() != actual["cases"].keys():
        differences.append("case_names")
    for name, result in actual["cases"].items():
        expected = reference["cases"].get(name, {})
        for field in ("inputs", "answer", "boxes", "points", "rendered_pixels_sha256"):
            if expected.get(field) != result.get(field):
                differences.append(f"{name}.{field}")
    return {"exact_match": not differences, "differences": differences}


def prepare_asset_directory(source, destination):
    """Expose only checkpoint data, with no executable model source."""
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError("--assets-only requires a local checkpoint directory")
    destination.mkdir(parents=True, exist_ok=True)
    for asset in source.iterdir():
        if asset.is_file() and asset.suffix in {
            ".json",
            ".jinja",
            ".safetensors",
            ".model",
            ".txt",
            ".tiktoken",
        }:
            target = destination / asset.name
            if target.is_symlink() and target.resolve() == asset:
                continue
            target.symlink_to(asset)
    if any(path.suffix == ".py" for path in destination.rglob("*")):
        raise AssertionError("Asset-only checkpoint contains Python source")
    return destination.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation", choices=["original", "local"], default="local"
    )
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--reference-attention-compat", action="store_true")
    parser.add_argument(
        "--assets-only",
        action="store_true",
        help="Test local loading from data-only asset symlinks.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--compare-report", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.assets_only:
        if args.implementation != "local":
            parser.error("--assets-only is for local source independence verification")
        args.model_path = str(
            prepare_asset_directory(args.model_path, args.report.parent / "assets_only")
        )

    if args.implementation == "original":
        if args.reference_root is None:
            parser.error("--reference-root is required for the original implementation")
        path = args.reference_root / "scripts" / "test_single_image.py"
        spec = importlib.util.spec_from_file_location("stl_reference_cli", path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        if args.reference_attention_compat:
            # Match the original 4.57 SDPA choice when running its loader on 4.53.
            from transformers import AutoConfig, AutoModel
            import locateanything_worker

            class CompatibleAutoModel:
                @staticmethod
                def from_pretrained(model_path, **kwargs):
                    config = AutoConfig.from_pretrained(
                        model_path, trust_remote_code=True
                    )
                    config._attn_implementation = "sdpa"
                    config.text_config._attn_implementation = "sdpa"
                    config.vision_config._attn_implementation = "sdpa"
                    return AutoModel.from_pretrained(
                        model_path, config=config, **kwargs
                    )

            locateanything_worker.AutoModel = CompatibleAutoModel
    else:
        from openpi.models.stl import cli

    worker = cli.LocateAnythingWorker(
        args.model_path, device="cuda", dtype=torch.bfloat16
    )
    with Image.open(args.image) as source:
        image = source.convert("RGB")

    report = {
        "implementation": args.implementation,
        "reference_attention_compat": args.reference_attention_compat,
        "model_path": args.model_path,
        "assets_only": args.assets_only,
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "image_size": list(image.size),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "tokenizers",
                "numpy",
                "Pillow",
                "peft",
                "accelerate",
            )
        },
        "source_files": {
            name: inspect.getfile(type(getattr(worker, name)))
            for name in ("model", "processor", "tokenizer")
        },
        "attention": {
            "text": worker.model.config.text_config._attn_implementation,
            "vision": worker.model.config.vision_config._attn_implementation,
        },
        "loading_info": getattr(worker, "loading_info", None),
        "cases": {},
    }
    if args.implementation == "local":
        root = Path(__file__).resolve().parent
        for name in ("model", "processor"):
            if not Path(report["source_files"][name]).is_relative_to(root):
                raise AssertionError(f"{name} was loaded from outside STL")

    captured = {}
    generate = worker.model.generate

    def record_generate(**kwargs):
        captured.clear()
        captured.update(
            {
                name: fingerprint(kwargs.get(name))
                for name in (
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "image_grid_hws",
                )
            }
        )
        return generate(**kwargs)

    worker.model.generate = record_generate
    text_query = "\u738b\u56fd\u7ef4\u6587\u9009"
    cases = [
        ("detect", "hybrid", None, ["book"]),
        ("ground-single", "hybrid", "the rightmost book", None),
        ("ground-multi", "hybrid", "books", None),
        ("detect-text", "hybrid", None, None),
        ("ground-text", "hybrid", text_query, None),
        ("gui-box", "hybrid", "the rightmost book", None),
        ("gui-point", "hybrid", "the rightmost book", None),
        ("point", "hybrid", "the rightmost book", None),
        ("custom", "hybrid", f"Please locate the text referred as {text_query}.", None),
        ("ground-text", "fast", text_query, None),
        ("ground-text", "slow", text_query, None),
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    for task, mode, query, categories in cases:
        torch.manual_seed(args.seed)
        task_args = argparse.Namespace(
            task=task,
            generation_mode=mode,
            query=query,
            categories=categories,
            max_new_tokens=args.max_new_tokens,
            verbose=False,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.1,
        )
        result = cli.run_task(worker, image, task_args)
        answer = result["answer"]
        boxes = worker.parse_boxes(answer, *image.size)
        points = worker.parse_points(answer, *image.size)
        rendered = cli.draw_predictions(image, boxes, points)
        name = f"{task}_{mode}"
        rendered.save(args.report.parent / f"{args.report.stem}_{name}.png")
        report["cases"][name] = {
            "inputs": dict(captured),
            "answer": answer,
            "boxes": boxes,
            "points": points,
            "rendered_pixels_sha256": hashlib.sha256(rendered.tobytes()).hexdigest(),
            "ended": "<|im_end|>" in answer,
        }
        print(f"{name}: {answer}", flush=True)

    report["dynamic_modules"] = sorted(
        name for name in sys.modules if name.startswith("transformers_modules.")
    )
    if args.implementation == "local" and report["dynamic_modules"]:
        raise AssertionError(
            "Local inference imported dynamic checkpoint Python modules"
        )
    if args.compare_report:
        report["comparison"] = compare_reports(
            json.loads(args.compare_report.read_text()), report
        )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Report: {args.report}", flush=True)
    if args.compare_report:
        print(json.dumps(report["comparison"], ensure_ascii=False), flush=True)
        if not report["comparison"]["exact_match"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
