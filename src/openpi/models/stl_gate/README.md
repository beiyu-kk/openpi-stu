# STL Gate Text Detector

`PP-OCRv6_tiny_det`, CPU Paddle Inference, batch size 1. This package provides
text polygons and scores for a future gate; it does not select a target book,
schedule STL calls, or alter OpenPI training/inference automatically.

Only the CPU runtime and the required DB preprocessing/postprocessing are used.
PaddleOCR, PaddleX, recognition, orientation classification, GPU Paddle,
services and external source imports are not required. The core implementation
is `detector.py` plus `db_postprocess.py`.

## OpenPI Environment

Use the existing repository `.venv`, not the separate PaddleOCR environment:

```bash
uv --no-config pip install --python .venv/bin/python \
  'https://paddle-whl.cdn.bcebos.com/stable/cpu/paddlepaddle/paddlepaddle-3.2.0-cp311-cp311-linux_x86_64.whl' \
  -r src/openpi/models/stl_gate/requirements.txt
```

The wheel above is for Linux x86_64 / Python 3.11. The root `pyproject.toml` and
`uv.lock` are not changed. Paddle requires `opt-einsum==3.3.0` and
`safetensors>=0.6.0`; the optional requirements pin the versions used here.
NumPy, OpenCV, PyTorch, JAX and Transformers retain their OpenPI versions.
The existing STL environment checker pins `safetensors==0.5.3`, so it reports
that one version mismatch after this installation. Its requirements file is
outside this change's scope. Actual STL localization with safetensors 0.6.2
was verified in the same process as this detector.

## Model

Supply a local inference directory containing `inference.json`,
`inference.pdiparams` and `inference.yml`. The default is the existing cache at
`~/.paddlex/official_models/PP-OCRv6_tiny_det`. There are no implicit downloads.

Only this model is supported. Its normalization uses BGR, scale 1/255,
mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225). DB thresholds are read
from `inference.yml` (normally 0.2 / 0.4 / 1.4 for threshold / score / unclip).

## Test and Call

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/test_stl_gate.py \
  --image test_data/pic/four_books.png
```

The script writes JSON and an annotated PNG under `stl_gate/artifacts/` and
reports warm mean/P50/P95 latency, including preprocessing and postprocessing.
File decoding, drawing and model initialization are excluded from warm timing.
Use `--model-dir`, `--limit-side-len`, `--cpu-threads`, `--box-thresh`,
`--warmup`, `--runs`, or `--output-dir` to override defaults.

```python
import numpy as np
from PIL import Image
from openpi.models.stl_gate import DBTextDetector

with DBTextDetector() as detector:
    rgb = np.asarray(Image.open("frame.png").convert("RGB"))
    result = detector.detect(rgb)
    print(result.polygons, result.scores, result.has_text)
```

Create one detector per sequential stream/worker and reuse it across frames.
Inputs are HWC RGB uint8 arrays, including non-contiguous arrays. Outputs are
original-image pixel coordinates: `(N, 4, 2)` int32 polygons and `(N,) float32
scores. Empty detections preserve these shapes. `has_text` indicates any detected
text, not a valid target or a keyframe decision. Instances are not thread-safe.

The resize rule is Paddle's `limit_type=max`, followed by rounding dimensions
to multiples of 32. Increasing the limit from 640 to 960 does not upscale a
640x480 frame. Keep the original RGB image for downstream STL and ROI cropping.

```bash
mkdir -p src/openpi/models/stl_gate/artifacts
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/openpi/models/stl_gate/test_detector.py \
  -o cache_dir=src/openpi/models/stl_gate/artifacts/pytest-cache \
  --basetemp=src/openpi/models/stl_gate/artifacts/pytest-tmp -q
```

## Source

Resize, normalization and the quad/fast DB decoding path are adapted from
PaddleX 3.7.2, `paddlex/inference/models/text_detection/processors.py`, under
Apache-2.0 (included in `LICENSE`). Source SHA256:
`b23f2349e9e3439d748548730443459d7ed43a4f5e5beb04903f3e5ad58be7c1`.
Unused resize modes, dilation, slow scoring, polygon output, dependency checks
and benchmarking wrappers are omitted. Empty results have stable shapes,
degenerate contours are skipped, and coordinates use int32 instead of int16.
No model network or learned parameters are reimplemented.

## Verification (2026-09-10)

- OpenPI Python 3.11.16; CPU Paddle 3.2.0; NumPy 1.26.4; OpenCV 4.11.0.
- Nine focused tests and Ruff checks pass.
- Sixteen cases cover books, a rotated image, an enlarged image, three video
  frames, a blank image and a tiny image at limits 640 and 960.
- In the original Paddle environment, input tensors, polygons and float32
  scores match `TextDetection` exactly in all sixteen cases.
- In OpenPI, the adapted decoder matches the original DB postprocessing in
  all sixteen cases. Cross-environment input tensors and polygons also match;
  scores differ by at most 0.00584. The original environment has OpenCV 4.10.0
  and NumPy 2.3.5; this integration preserves OpenPI's versions.
- Both import orders work with OpenPI's JAX/PyTorch GPU computations. Actual
  STL inference also succeeds while the CPU detector is alive, with unchanged
  DB results before and after STL inference. No full policy training was run.
- On i9-14900K, two Paddle threads, a 628x423 book image, five warmups and
  thirty measured calls: mean 20.13 ms, P50 19.82 ms, P95 21.38 ms.

Local output and comparison reports are under the ignored `artifacts/` folder.
These checks verify integration and implementation parity, not general text
detection accuracy or speed during concurrent OpenPI training.
