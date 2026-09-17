# Training-Only Region Attention Guidance

This option uses the existing black-label annotations to bias action-to-image
attention during training. It adds no auxiliary loss, model parameters, image
views, detector, or inference inputs. The existing Piper commands and configs
keep their original behavior unless `--region-guidance` is specified.

## Launch

Run from the repository root. The dataset below is the annotated independent
copy; neither training mode needs to modify the original dataset.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync scripts/train_piper.py \
  --config pi05_piper_lora_finetune_448 \
  --dataset-dir /media/ubun/16T/Dataset/piper_data/recognition_book/recognize_book_label_color_1_lerobot_1_v2.1_spine_annotated \
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
prompt transforms are the same. Original launch commands also continue to work.

`--region-annotations-dir` can point to a separate `annotations/` directory;
it defaults to `<dataset-dir>/annotations`. Startup verifies the source manifest
against the selected dataset, including every source file's SHA256. A mismatch
aborts the run rather than using another dataset's boxes.

## Defaults

| Setting | Value |
| --- | --- |
| Region source | Visible black-label polygon, `label_polygon_xy` |
| Eligibility | `use_for_region_loss=true`; used as a validity flag, not a new loss |
| Initial bias | 0.5 |
| Initial keep probability | 0.5 per example and camera |
| Constant phase | First 30% of optimizer steps |
| Linear decay | 30% through 70% of optimizer steps |
| Unguided phase | Final 30% of optimizer steps |
| Layers | Two middle layers; indices 8 and 9 for the 18-layer pi0.5 |
| Query heads | First quarter; indices 0 and 1 for pi0.5 |
| Attention path | Action queries to image keys only |

At 10,000 steps, strength/probability are 0.5/0.5 through step 3000,
0.25/0.25 at step 5000, and exactly zero from step 7000 onward.
The decay uses the checkpoint's optimizer step, not the data-loader position.

Optional overrides:

```text
--region-bias-strength 0.5
--region-keep-probability 0.5
--region-decay-start 0.3
--region-decay-end 0.7
--region-layers 8 9
--region-heads 0 1
```

The defaults are experimental starting points, not proven optimal values.

## Data and Attention

The original transforms run first. A read-only dataset wrapper attaches two
optional fields, `region_masks` and `region_valid`, afterward. Invalid regions
and the missing left camera receive zero masks. No action samples are dropped.
Episode, frame, global index, timestamp and original image dimensions are checked.

Verified polygons reconstruct the label masks without decompressing a whole
video's NPZ for every shuffled sample. Masks use the same resize/padding geometry
as the original images. The existing augmentation chain jointly transforms
images and masks: masks share cropping/rotation but skip color jitter. Fractional
coverage is averaged over each 14-by-14 image patch. At 448 pixels, each camera
has a 32-by-32 patch grid.

For the selected layers and heads, action queries receive:

```text
logits' = logits + strength(step) * keep_gate * patch_coverage
```

The original hard attention mask is applied afterward. Language keys, action
keys, prefix queries and invalid camera keys receive no region bias. Gates are
sampled once per example/camera and shared across selected layers and heads.
The baseline's image augmentation, action noise and flow timestep RNG streams
are preserved. The original flow-matching action loss is unchanged.

This implementation is supported by the JAX standard Piper LeRobot pipeline.
Custom geometric preprocessing, PyTorch training and RLDS require their own
alignment/integration work and are not enabled by this switch.

## Resume and Deployment

Use the same launch options plus `--resume` to continue a guided run. The output
directory records `region_guidance.json`; resume checks the guidance settings,
total training steps, input resolution and dataset paths. Changing the total
steps would change the annealing schedule, so it is rejected. Start a separate
run to change the method or schedule.

No model parameters are added. Guided checkpoints can be loaded using the
original matching pi0.5 configuration (same resolution and LoRA variants).
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
