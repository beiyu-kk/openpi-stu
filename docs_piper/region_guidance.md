# Training-Only Region Attention Guidance

This option uses the existing black-label annotations to bias configured
query-to-image attention during training. It adds no auxiliary loss, model parameters, image
views, detector, or inference inputs. Ordinary Piper configs keep their original
behavior unless `--region-guidance` is specified. The dedicated
`pi05_piper_full_finetune_rtc_attention_bias` config enables guidance directly.

## Launch

Run from the repository root. The dataset already contains the two bbox fields
in its episode parquet files.

For the current `recognize_book_name` RTC run, use the dedicated config and
`--dataset-dir .../success/origin`; see
[`rtc_attention_bias_training.md`](rtc_attention_bias_training.md) for the exact
command. That config does not require an additional `--region-guidance` flag.
The command below demonstrates enabling the same mechanism explicitly on an
ordinary LoRA config.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync scripts/train_piper.py \
  --config pi05_piper_lora_finetune \
  --image-size 448 \
  --dataset-dir /media/ubun/16T/Dataset/piper_eva_data/recognition/recognize_book_name/success/origin \
  --dataset-repo-id recognize_book_label_color_1_region_bias \
  --base-model-dir /media/ubun/16T/checkpoints/openpi/openpi-assets/checkpoints/pi05_base/params \
  --checkpoint-dir /media/ubun/16T/checkpoints/openpi/recognize_book_label_color_1_region_bias \
  --exp-name recognize_book_label_color_1_region_bias \
  --batch-size 1 \
  --num-train-steps 10000 \
  --compute-norm-stats \
  --region-guidance
```

Batch size 1 is intended as a conservative startup setting; increase it based on
available GPU memory. Use the same batch size for the baseline comparison.
`--no-sync` uses the existing environment without resolving unrelated packages.

For a baseline run, remove `--region-guidance` and use a different checkpoint
directory and experiment name. All image, state, action, normalization and
prompt transforms are the same. `--config` selects the TrainConfig; `--image-size`
sets the square input resolution independently. Both Piper configs default to
224. Replace former `_336`/`_448` config names with the base name and
`--image-size 336`/`--image-size 448`.

There is no sidecar or separate annotation path. Startup validates the embedded
feature declarations, parquet column types, episode count, and total frame
count. Field locations, conversion, and inspection commands are documented in
[`annotation_adapter.md`](annotation_adapter.md).

## Defaults

| Setting | Value |
| --- | --- |
| Region source | Embedded book/title xyxy bbox from the right-wrist image |
| Eligibility | A valid positive-area bbox; `[-1, -1, -1, -1]` means unavailable |
| Initial attention bias multiplier | `attention_bias_scale=0.5` |
| Initial keep probability | 0.5 per example and camera |
| Constant phase | First 30% of optimizer steps |
| Linear decay | 30% through 70% of optimizer steps |
| Unguided phase | Final 30% of optimizer steps |
| Layers | Two middle layers; indices 8 and 9 for the 18-layer pi0.5 |
| Query heads | First quarter; indices 0 and 1 for pi0.5 |
| Attention path | Configured query types (`action`, `image`, `prompt`) to annotated image keys |

At 10,000 steps, attention-bias-scale/probability are 0.5/0.5 through step 3000,
0.25/0.25 at step 5000, and exactly zero from step 7000 onward.
The decay uses the checkpoint's optimizer step, not the data-loader position.

The multiplier is explicit in the model TrainConfig:

```python
region_guidance=RegionGuidanceConfig(attention_bias_scale=1.5)
```

The query scope is explicit in the model config. The original behavior is
`query_types=("action",)`. The current bbox-guidance presets enable all three
query types:

```python
region_guidance=RegionGuidanceConfig(
    attention_bias_scale=0.5,
    query_types=("action", "image", "prompt"),
)
```

This applies the same annealed bias when image or prompt queries attend to
annotated image patches. The key side remains the selected image patches; no
bias is added to language or action keys.

At the start of training, `attention_bias_scale=1.5` adds `+1.5` to a fully
covered patch's pre-softmax attention logit; 40% coverage adds `+0.6`. The
configured scale is then reduced by the decay schedule.

Optional command-line overrides:

```text
--attention-bias-scale 1.5
--region-keep-probability 0.5
--region-decay-start 0.3
--region-decay-end 0.7
--region-layers 8 9
--region-heads 0 1
```

The defaults are experimental starting points, not proven optimal values.

## Data and Attention

The dataset wrapper reads the bbox from the same LeRobot sample, then attaches
`region_masks` and `region_valid`. Invalid regions and the missing left camera
receive zero masks. No action samples are dropped.

Boxes are rasterized into masks in the original right-wrist image coordinates.
Masks use the same resize/padding geometry
as the original images. The existing augmentation chain jointly transforms
images and masks: masks share cropping/rotation but skip color jitter. Fractional
coverage is averaged over each 14-by-14 image patch. At 448 pixels, each camera
has a 32-by-32 patch grid.

For the selected layers and heads, each configured query type receives:

```text
logits' = logits + attention_bias_scale(step) * keep_gate * patch_coverage
```

The original hard attention mask is applied afterward. Language keys, action
keys and invalid camera keys receive no region bias. Query types that are not
listed in `query_types` receive no region bias. Gates are sampled once per
example/camera and shared across selected layers and heads.
The baseline's image augmentation, action noise and flow timestep RNG streams
are preserved. The original flow-matching action loss is unchanged.

This implementation is supported by the JAX standard Piper LeRobot pipeline.
Custom geometric preprocessing, PyTorch training and RLDS require their own
alignment/integration work and are not enabled by this switch.

## Resume and Deployment

Use the same launch options, including `--image-size`, plus `--resume` to continue a guided run. The output
directory records `region_guidance.json`; resume checks the guidance settings,
total training steps, input resolution and dataset paths. Changing the total
steps would change the annealing schedule, so it is rejected. Start a separate
run to change the method or schedule.

No model parameters are added. Guided checkpoints can be loaded using the
matching pi0.5 configuration and image size: for the example above, use
`--policy.config=pi05_piper_lora_finetune --policy.image-size=448` when serving.
`sample_actions` never applies the annotation bias and needs no annotation files.
Likewise, `compute_loss(..., train=False)` measures the unguided model.

Training logs include `region/strength`, `region/keep_probability` and
`region/valid_fraction/<camera>`. The latter reports annotation eligibility,
not the randomly realized gate frequency or annotation accuracy.

## Evaluation

Compare the baseline and guided training with identical data splits, batch size,
initial weights, augmentation, optimizer and step budget. Split by complete
episodes or scene layouts, never adjacent frames. Evaluate with no annotation
bias, and measure target-selection, grasp and complete pick/place success.

This is a training intervention, not an internal text detector. Current labels
identify a black sticker; they do not establish OCR or scene-text understanding.
An improvement while annotation bias is enabled does not demonstrate an
improvement in autonomous deployment.

Focused regression checks:

The CPU checkpoint integration test disables persistent JAX compilation caching
locally: this environment's JAX 0.5.3 cached CPU executables can crash on the
first donated update after restore. Production GPU cache settings are unchanged.

```bash
JAX_PLATFORMS=cpu uv run --no-sync python -m pytest -q \
  src/openpi/models/region_guidance_test.py \
  src/openpi/training/region_annotations_test.py \
  scripts/train_region_test.py \
  scripts/train_piper_test.py \
  src/openpi/training/config_test.py \
  src/openpi/policies/piper_policy_test.py \
  src/openpi/transforms_test.py
```
