# stl_gate 启动

输出包含文字的视频

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/test_stl_gate.py \
  --image test_data/pic/four_books.png
```

输出包含文字的视频，以及视频中的关键帧

```bash
python scripts/stl_gate/filter_text_video.py \
  --input test_data/videos/episode_0.mp4 \
  --output test_data/videos/episode_0_text_only.mp4 \
  --save-keyframes \
  --overwrite
```

## codex 输出
已接入到 [stl_gate](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl_gate)，并新增测试脚本 [scripts/test_stl_gate.py](/home/ubun/project/stu_vla/openpi-piper/scripts/test_stl_gate.py)。**本次源码改动仅发生在该目录和这个脚本中**，没有修改现有模型、策略、训练代码或根依赖文件。

采用的是 **`PP-OCRv6_tiny_det + CPU Paddle Inference`**。为减少依赖，直接调用 Paddle 静态图推理接口，保留官方单图检测所需的前后处理，无需安装 PaddleOCR/PaddleX。

核心实现共两个文件，约 200 行：

- [detector.py](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl_gate/detector.py)：模型加载与复用、RGB 输入转换、预处理、CPU 推理和结果返回。
- [db_postprocess.py](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl_gate/db_postprocess.py)：从概率图提取文字四边形、检测分数，并还原到原图坐标。

输入为 `HWC / RGB / uint8` 的 NumPy 图像，返回 `polygons`、`scores` 和 `has_text`，供后续门控判断使用。模型实例可以重复处理多帧。

现在直接使用 **OpenPI 当前 `.venv`** 测试：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/test_stl_gate.py \
  --image test_data/pic/four_books.png
```

默认读取已有权重目录 `~/.paddlex/official_models/PP-OCRv6_tiny_det`，也可以用 `--model-dir` 指定。测试输出已生成：

- [检测框图片](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl_gate/artifacts/four_books.png)
- [坐标、分数和测速 JSON](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl_gate/artifacts/four_books.json)

本机 i9-14900K、CPU 两线程、书本图片 628×423，预热 5 次、测量 30 次的结果为：

| 指标 | 结果 |
|---|---:|
| 检测文字区域 | 5 个 |
| 平均耗时 | **20.13 ms** |
| P50 | 19.82 ms |
| P95 | 21.38 ms |

耗时包含预处理、推理和后处理，不包含图片读取、模型初始化和绘图。

验证已完成：9 项单元测试及 Ruff 检查通过；16 组样本完成官方实现对照；JAX、PyTorch 与检测器同进程运行通过；还实际加载了原 STL，成功定位“王国维文选”，且 STL 调用前后 DB 检测结果一致。详细记录见 [README](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl_gate/README.md)。

**环境变更需要说明：**当前 `.venv` 新增了 CPU Paddle、pyclipper 及必要依赖；按 Paddle 要求，`safetensors` 从 `0.5.3` 更新到 `0.6.2`，`opt-einsum` 从 `3.4.0` 调整到 `3.3.0`。NumPy、OpenCV、PyTorch、JAX、Transformers 保持原版本。原 STL 的精确版本检查会因此报告一项 `safetensors` 差异，其目录按你的限制未修改；实际联合定位已经验证通过。