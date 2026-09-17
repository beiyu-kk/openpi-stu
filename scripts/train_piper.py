"""Reusable entry point for pi0.5 fine-tuning on a local Piper LeRobot dataset."""

import argparse
import dataclasses
import logging
import pathlib

from openpi.models.region_guidance import RegionGuidanceConfig
from openpi.training import config as _config
from openpi.training import weight_loaders

if __package__:
    from scripts import compute_norm_stats
    from scripts import train
else:
    import compute_norm_stats
    import train


CONFIGS = (
    "pi05_piper_full_finetune",
    "pi05_piper_lora_finetune",
    "pi05_piper_full_finetune_336",
    "pi05_piper_lora_finetune_336",
    "pi05_piper_full_finetune_448",
    "pi05_piper_lora_finetune_448",
)
DEFAULT_BASE_MODEL = "gs://openpi-assets/checkpoints/pi05_base"


def _resolve_base_params_path(value: str) -> str:
    value = value.rstrip("/")
    if "://" in value:
        return value if value.endswith("/params") else f"{value}/params"

    model_path = pathlib.Path(value).expanduser().resolve()
    params_path = model_path if model_path.name == "params" else model_path / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"Base model params not found: {params_path}")
    return str(params_path)


def _validate_dataset_dir(value: str) -> pathlib.Path:
    dataset_dir = pathlib.Path(value).expanduser().resolve()
    required_paths = (dataset_dir / "meta" / "info.json", dataset_dir / "data")
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Invalid LeRobot dataset directory; missing: {', '.join(missing)}")
    return dataset_dir


def _build_config(args: argparse.Namespace) -> _config.TrainConfig:
    dataset_dir = _validate_dataset_dir(args.dataset_dir)
    norm_stats_dir = pathlib.Path(args.norm_stats_dir).expanduser().resolve() if args.norm_stats_dir else dataset_dir
    base_config = _config.get_config(args.config)
    if not isinstance(base_config.weight_loader, weight_loaders.CheckpointWeightLoader):
        raise TypeError("Piper fine-tuning requires a CheckpointWeightLoader")
    weight_loader = dataclasses.replace(
        base_config.weight_loader,
        params_path=_resolve_base_params_path(args.base_model_dir),
    )
    repo_id = args.dataset_repo_id

    data = dataclasses.replace(
        base_config.data,
        repo_id=repo_id,
        dataset_root=str(dataset_dir),
        norm_stats_dir=str(norm_stats_dir),
        assets=_config.AssetsConfig(asset_id=repo_id),
    )
    model = base_config.model
    if getattr(args, "region_guidance", False):
        if not isinstance(data, _config.LeRobotPiperDataConfig):
            raise ValueError("Region guidance requires the standard Piper data transforms")
        annotation_dir = pathlib.Path(args.region_annotations_dir or dataset_dir / "annotations").expanduser().resolve()
        if not (annotation_dir / "all_frames.parquet").is_file():
            raise FileNotFoundError(f"Region annotation table not found in {annotation_dir}")
        guidance = RegionGuidanceConfig(
            strength=args.region_bias_strength,
            keep_probability=args.region_keep_probability,
            decay_start=args.region_decay_start,
            decay_end=args.region_decay_end,
            layers=None if args.region_layers is None else tuple(args.region_layers),
            heads=None if args.region_heads is None else tuple(args.region_heads),
        )
        model = dataclasses.replace(model, region_guidance=guidance)
        data = dataclasses.replace(
            data,
            base_config=dataclasses.replace(
                data.base_config or _config.DataConfig(), region_annotations_dir=str(annotation_dir)
            ),
        )
    elif getattr(args, "region_annotations_dir", None) is not None:
        raise ValueError("--region-annotations-dir requires --region-guidance")
    updates = {
        "data": data,
        "model": model,
        "weight_loader": weight_loader,
        "checkpoint_base_dir": args.checkpoint_base_dir,
        "checkpoint_dir_override": args.checkpoint_dir,
        "exp_name": args.exp_name,
        "overwrite": args.overwrite,
        "resume": args.resume,
    }
    for arg_name in ("batch_size", "num_workers", "num_train_steps", "fsdp_devices"):
        if (value := getattr(args, arg_name)) is not None:
            updates[arg_name] = value
    if args.wandb_enabled is not None:
        updates["wandb_enabled"] = args.wandb_enabled
    return dataclasses.replace(base_config, **updates)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGS, default=CONFIGS[0])
    parser.add_argument("--dataset-dir", required=True, help="Root directory of the local LeRobot dataset.")
    parser.add_argument("--dataset-repo-id", required=True, help="Logical ID used for dataset assets in checkpoints.")
    parser.add_argument(
        "--base-model-dir",
        default=DEFAULT_BASE_MODEL,
        help="pi0.5 checkpoint root or its params directory; local paths and gs:// URLs are supported.",
    )
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument("--checkpoint-dir", help="Exact checkpoint output directory; overrides --checkpoint-base-dir.")
    parser.add_argument("--norm-stats-dir", help="Directory containing norm_stats.json; defaults to --dataset-dir.")
    parser.add_argument(
        "--compute-norm-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute missing stats, or reuse them when norm_stats.json already exists.",
    )
    parser.add_argument("--max-norm-frames", type=int)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to 32 from the Piper configs.")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-train-steps", type=int)
    parser.add_argument("--fsdp-devices", type=int)
    parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--region-guidance", action="store_true", help="Enable training-only annotation attention bias."
    )
    parser.add_argument(
        "--region-annotations-dir", help="Defaults to <dataset-dir>/annotations when guidance is enabled."
    )
    parser.add_argument("--region-bias-strength", type=float, default=0.5)
    parser.add_argument("--region-keep-probability", type=float, default=0.5)
    parser.add_argument(
        "--region-decay-start", type=float, default=0.3, help="Fraction of training before decay starts."
    )
    parser.add_argument("--region-decay-end", type=float, default=0.7, help="Fraction after which guidance is zero.")
    parser.add_argument(
        "--region-layers", type=int, nargs="+", help="Zero-based layers; defaults to two middle layers."
    )
    parser.add_argument(
        "--region-heads", type=int, nargs="+", help="Zero-based query heads; defaults to the first quarter."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    config = _build_config(args)
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats_dir = compute_norm_stats.get_norm_stats_dir(config, data_config)
    norm_stats_path = norm_stats_dir / "norm_stats.json"

    if norm_stats_path.exists():
        logging.info("Reusing normalization stats: %s", norm_stats_path)
    elif args.compute_norm_stats:
        logging.info("Normalization stats not found; computing: %s", norm_stats_path)
        compute_norm_stats.compute(config, max_frames=args.max_norm_frames)
    else:
        raise FileNotFoundError(
            f"Normalization stats not found: {norm_stats_path}. "
            "Run again with --compute-norm-stats to generate them automatically."
        )

    logging.info("Dataset: %s", data_config.dataset_root)
    logging.info("Base model params: %s", config.weight_loader.params_path)
    logging.info("Checkpoint output: %s", config.checkpoint_dir)
    logging.info("Fine-tuning config: %s (batch size %d)", config.name, config.batch_size)
    logging.info("Image resolution: %s", config.model.image_resolution)
    if config.model.region_guidance is not None:
        logging.info("Training-only region guidance: %s", config.model.region_guidance)
    train.main(config)


if __name__ == "__main__":
    main()
