# Piper fine-tuning

The current Piper recipes support both pi0.5 and autoregressive pi0-FAST.

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

## Image resolution

The fixed-resolution Piper TrainConfigs use `224x224` by default, `_image448` configs use `448x448`,
and `_image640` configs use `640x640`. For a `640x480` camera frame, the 640 config preserves the original
pixels and adds 80 rows of padding above and below.
Use `pi0_fast_piper_full_finetune_image640` / `pi0_fast_piper_lora_finetune_image640` for pi0-FAST, or
`pi05_piper_full_finetune_image640` / `pi05_piper_lora_finetune_image640` for pi0.5.
For strict no-scaling preprocessing, use the `_image640_pad_only` variants. They only center-pad smaller images;
they reject an input if either side exceeds 640 pixels. Examples are
`pi0_fast_piper_full_finetune_image640_pad_only` and `pi05_piper_full_finetune_image640_pad_only`.
Standard image640 variants keep resize-with-pad behavior.
The pad-only canvas is fixed by the TrainConfig: a `320x240` image is placed unchanged in a `640x640` canvas too.
To use a larger canvas, register a new config with `image_resolution=(side, side)` and
`image_resize_mode="pad_only"`. Choose a side at least as large as every source image dimension.
This JAX mode disables crop, resize and rotation augmentations during training; color jitter is retained.
SigLIP additionally pads the right and bottom edges to a patch boundary (640 to 644), without scaling.

For native-size inputs, select a `_pad_to_square` config, such as
`pi0_fast_piper_full_finetune_pad_to_square` or `pi05_piper_lora_finetune_pad_to_square`.
These configs set `pad_to_square=True` directly on `Pi0FASTConfig` / `Pi0Config`, next to
`image_resolution`. The switch defaults to `False`, which uses the fixed canvas and `image_resize_mode`.
When `True`, dynamic padding takes precedence over `image_resize_mode` and `image_resolution` only
defines the position embedding reference and initialization size.
Each image is center-padded to its own longest side: `640x480` becomes `640x640`, while
`1280x720` becomes `1280x1280`. This JAX mode preserves pixels without resizing or cropping,
including during training; color jitter remains enabled. Cameras and batches can have different sizes.
Within a batch, each camera is padded again to that camera's largest image so it can be stacked;
region masks receive the same padding. SigLIP pads to patch boundaries and interpolates position embeddings
at runtime from a fixed reference grid (`image_resolution`), keeping parameter and optimizer shapes stable.
New input shapes trigger JAX compilation, and larger images increase memory use. Spatial token merging
is currently unsupported in this mode. TT-RTC variants use `_finetune_rtc_pad_to_square`;
both full and LoRA fine-tuning are available for pi0.5 and pi0-FAST.
Image processing is selected exclusively through the named TrainConfig.

For example, start pi0-FAST LoRA fine-tuning at `448x448`:

```bash
uv run scripts/train_piper.py \
  --config=pi0_fast_piper_lora_finetune_image448 \
  --dataset-dir /path/to/lerobot/dataset \
  --dataset-repo-id piper_dataset \
  --base-model-dir ./checkpoints/pi0_fast_base \
  --checkpoint-dir ./checkpoints/piper/fast_lora_448 \
  --exp-name piper_fast_lora_448 \
  --batch-size 1
```

Higher resolution increases visual token count and memory use. Start with a smaller batch size and measure
memory use before increasing it.

All Piper configs enable SigLIP position embedding adaptation. When initializing from a 224 base checkpoint, the loader
interpolates only `PaliGemma/img/pos_embedding` from a `16x16` grid to `24x24` for 336 or `32x32` for 448,
using bicubic interpolation. The interpolated embedding remains trainable.
Other parameter shapes are still checked strictly, and matching-resolution embeddings are loaded unchanged.
The base checkpoint is not modified. When constructing a custom training config directly, enable
`CheckpointWeightLoader(..., resize_siglip_posemb=True)` to adapt pretrained position embeddings.

Changing the fixed resolution or image processing mode starts a new fine-tuning run from model weights.
Do not use `--resume` to convert an old 224 training state: resume requires the original resolution,
including optimizer state shapes. State/action
normalization statistics can be reused when the dataset and action transforms are unchanged.

Serve the trained checkpoint with the same TrainConfig:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_fast_piper_lora_finetune_image448 \
  --policy.dir ./checkpoints/piper/fast_lora_448/9999
```

The selected TrainConfig defines the image processing used for training and policy preprocessing.

The Piper camera client sends original-resolution RGB frames. Image processing happens on the policy server after
Piper preprocessing; no separate client image-size setting is needed.
This also preserves source detail for text detection. Network payloads contain the original camera frames.
