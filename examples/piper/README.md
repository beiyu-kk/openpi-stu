# Piper pi0.5 fine-tuning

The `pi05_piper_full_finetune` and `pi05_piper_lora_finetune` configs define Piper transforms and training
hyperparameters without embedding a dataset location or ID. Both configs use a global batch size of 32 and an action
horizon of 30 frames (one second at 30 FPS). They convert all seven absolute action dimensions to values relative to
the current state. The seventh gripper dimension remains continuous; it is not binarized.

Use the reusable wrapper to select the local dataset, pi0.5 base checkpoint, and exact checkpoint output path:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
  --config pi05_piper_full_finetune \
  --dataset-dir /path/to/lerobot/dataset \
  --dataset-repo-id piper_dataset \
  --base-model-dir ./checkpoints/pi05_base \
  --checkpoint-dir ./checkpoints/piper/full \
  --exp-name piper_full \
  --compute-norm-stats \
  --overwrite
```

For LoRA, change the config and output directory:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
  --config pi05_piper_lora_finetune \
  --dataset-dir /path/to/lerobot/dataset \
  --dataset-repo-id piper_dataset \
  --base-model-dir ./checkpoints/pi05_base \
  --checkpoint-dir ./checkpoints/piper/lora \
  --exp-name piper_lora \
  --compute-norm-stats \
  --overwrite
```

`--dataset-dir` and `--dataset-repo-id` are required. The ID is a stable logical name used to store normalization
assets inside checkpoints; it does not need to be a Hugging Face Hub repository.

`--base-model-dir` accepts either a checkpoint root containing `params` or the `params` directory itself. It also
accepts a `gs://` checkpoint URL; when omitted it uses `gs://openpi-assets/checkpoints/pi05_base`.

By default, OpenPI normalization statistics are read from `<dataset-dir>/norm_stats.json`. With
`--compute-norm-stats`, the wrapper computes that file only when it is missing and reuses it on subsequent runs.
Use `--norm-stats-dir` to keep it elsewhere. Without `--compute-norm-stats`, a missing file is reported before
model initialization.

When `--checkpoint-dir` is omitted, the standard OpenPI layout is used:
`<checkpoint-base-dir>/<config>/<exp-name>`. Use `--resume` instead of `--overwrite` to continue an existing run.

## Image resolution (JAX)

Image resolution is set in `src/openpi/training/config.py`, in the selected `TrainConfig.model`:

```python
model=pi0_config.Pi0Config(
    pi05=True,
    action_horizon=30,
    image_resolution=(336, 336),
),
```

Training transforms, model initialization, training augmentation and action inference all read this field.
Training and policy-serving entry points do not expose an image-size override. Choose the same config for training
and deployment. The default is `224x224`; `224x224`, `336x336` and `448x448` have dedicated regression coverage.
Other square sizes must also be positive multiples of the SigLIP patch size, 14.

The following presets define all three resolutions in the training config:

| Resolution | Full fine-tuning | LoRA fine-tuning |
| --- | --- | --- |
| 224 | `pi05_piper_full_finetune` | `pi05_piper_lora_finetune` |
| 336 | `pi05_piper_full_finetune_336` | `pi05_piper_lora_finetune_336` |
| 448 | `pi05_piper_full_finetune_448` | `pi05_piper_lora_finetune_448` |

For example, select the 448 configuration:

```bash
uv run scripts/train_piper.py \
  --config pi05_piper_full_finetune_448 \
  --dataset-dir /path/to/lerobot/dataset \
  --dataset-repo-id piper_dataset \
  --base-model-dir ./checkpoints/pi05_base \
  --checkpoint-dir ./checkpoints/piper/full_448 \
  --exp-name piper_full_448 \
  --batch-size 1
```

The presets inherit the original training hyperparameters, including batch size 32. Start with a small batch
and measure memory use before increasing it: 448 produces 1024 visual tokens per camera, compared with 256 at 224.
The example explicitly selects batch size 1 for this reason; it is not a tuned training recommendation.

All Piper configs enable position embedding adaptation. When initializing from the 224 base checkpoint, the loader
interpolates only `PaliGemma/img/pos_embedding` from a `16x16` grid to `24x24` for 336 or `32x32` for 448,
using bicubic interpolation. The interpolated embedding remains trainable.
Other parameter shapes are still checked strictly, and matching-resolution embeddings are loaded unchanged.
The base checkpoint is not modified. When constructing a custom training config directly, enable
`CheckpointWeightLoader(..., resize_siglip_posemb=True)` to adapt pretrained position embeddings.

Changing resolution starts a new fine-tuning run from model weights. Do not use `--resume` to convert an old
224 training state: resume requires the original resolution, including optimizer state shapes. State/action
normalization statistics can be reused when the dataset and action transforms are unchanged.

Serve the trained checkpoint with the matching preset:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_piper_full_finetune_448 \
  --policy.dir ./checkpoints/piper/full_448/5000
```

If editing an existing config's `model.image_resolution`, deploy with that same edited config and a matching checkpoint.
Resolution is read from the training config, not automatically inferred from checkpoint parameters.

The Piper camera client sends original-resolution RGB frames. Resizing happens on the policy server after
Piper preprocessing, using the selected training config; no separate client image-size setting is needed.
This also preserves source detail for text detection. Network payloads contain the original camera frames.
This configuration is currently supported by the JAX model; the PyTorch model rejects non-224 resolutions explicitly.
