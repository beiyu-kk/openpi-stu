# data_resize_visual

把单张照片或 MP4 的每一帧按 OpenPI 送入 SigLIP 之前的 `resize_with_pad` 阶段处理，输出缩放后的图片或完整 MP4 视频。

默认行为与 OpenPI 正常的 Policy 推理路径一致：

- 目标分辨率为 `224x224`。
- 输入按 RGB、`uint8` 处理。
- 保持原始宽高比，不裁剪、不拉伸。
- 缩放尺寸使用 OpenPI 相同的 `int(source / ratio)` 截断规则。
- 每帧使用 `PIL.Image.fromarray` 转成图像。
- 插值使用与 OpenPI Client 相同的 `PIL.Image.BILINEAR`。
- 使用 `PIL.Image.new(..., 0)` 创建目标画布，再按 OpenPI 相同的位置 `paste`。
- 空白区域补黑；遇到奇数 padding 时，多出的 1 像素放在右侧或底部。
- 保留每一帧、原视频帧率和音轨（如果有）。
- 默认输出兼容性最好的 H.264 `yuv420p`，显式使用 BT.709 色彩空间，避免播放器出现偏绿、偏紫等颜色错误。

## 环境

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。处理视频还需要系统 FFmpeg（`ffmpeg`、`ffprobe`），处理照片不需要 FFmpeg。项目独立维护自己的 Python 依赖，不依赖父目录的 `openpi` 包。

```bash
cd data_resize_visual
uv sync
```

## 使用

### 单张照片

支持 PNG、JPEG（`.jpg`/`.jpeg`）、WebP、BMP、TIFF 单帧图片，自动识别输入扩展名。
读取照片时先按 EXIF 方向转正，再转换为 RGB 并执行与视频帧相同的缩放、补黑边处理。

```bash
uv run data-resize-visual photo.jpg
```

默认在输入文件旁生成 `photo_resized_224x224.png`。默认使用 PNG 无损保存，便于检查 resize 后的像素。

指定输出位置和尺寸：

```bash
uv run data-resize-visual photo.jpg -o outputs/photo.png --width 224 --height 224
```

支持 `--overwrite` 覆盖已有输出、`--quiet` 隐藏完成提示，输出路径不能与原图相同。
图片宽高可以为奇数；`--batch-size`、`--no-audio`、`--lossless-rgb` 仅用于视频。
保存为 JPEG 或默认 WebP 会有有损编码，不适合逐像素核验；需要核验时使用 PNG。

从 OpenPI 仓库根目录也可以直接使用子项目环境：

```bash
data_resize_visual/.venv/bin/data-resize-visual /path/to/photo.jpg -o data_resize_visual/outputs/photo.png
```

### MP4 视频

```bash
uv run data-resize-visual /home/ubun/project/stu_vla/openpi-piper/test_data/videos/episode_0.mp4
```

默认在输入文件旁生成：

```text
input_resized_224x224.mp4
```

指定输出位置：

```bash
uv run data-resize-visual input.mp4 -o output.mp4
```

常用参数：

```bash
uv run data-resize-visual input.mp4 \
  --height 224 \
  --width 224 \
  --batch-size 16 \
  --overwrite
```

- `--no-audio`：不复制输入音轨。
- `--lossless-rgb`：输出逐像素无损的 RGB H.264，仅用于 FFmpeg 等工具核验；部分播放器会错误显示其颜色。
- `--quiet`：隐藏处理进度。
- `--overwrite`：允许覆盖已有输出文件。

也可以从 OpenPI 仓库根目录运行：

```bash
uv run --project data_resize_visual data-resize-visual input.mp4 -o output.mp4
```

## 测试

```bash
uv run --group dev pytest
```

`src/data_resize_visual/resize.py` 是从当前 OpenPI 的 `packages/openpi-client/src/openpi_client/image_tools.py::resize_with_pad` 独立抽出的核心实现。OpenPI 的 `transforms.ResizeImages` 和 Piper 输入路径实际使用的就是这套 PIL 处理。视频解码得到 RGB 帧后直接把批次交给该函数，没有使用 OpenCV 的近似 resize。

独立项目固定使用 Pillow `11.2.1`，与当前 OpenPI 的 `uv.lock` 实际解析版本一致，避免不同 Pillow 版本在插值边界上产生差异。MP4 解码和重新编码不属于 OpenPI 图像预处理；默认编码仅用于兼容播放器，`--lossless-rgb` 用于需要逐像素检查编码前 resize 结果的场景。
