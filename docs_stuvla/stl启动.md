# stl single image启动

## 任务类型

| `--task` 参数 | 功能 | 额外参数 |
|---|---|---|
| `detect` | 检测指定类别的所有目标 | `--categories 类别1 类别2` |
| `ground-single` | 根据描述定位单个目标 | `--query "目标描述"` |
| `ground-multi` | 根据描述定位所有匹配目标 | `--query "目标描述"` |
| `detect-text` | 检测图片中的所有文字区域 | 不需要额外参数 |
| `ground-text` | 定位指定文字 | `--query "文字内容"` |
| `gui-box` | 用框定位 GUI 元素 | `--query "元素描述"` |
| `gui-point` | 返回 GUI 元素的点击位置 | `--query "元素描述"` |
| `point` | 返回指定目标的点坐标 | `--query "目标描述"` |
| `custom` | 直接向模型发送自定义问题 | `--query "完整问题"` |

## 1. ground-text 启动

文本定位功能

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
python scripts/stl/test_stl_single_image.py \
    --model-path /media/ubun/16T/checkpoints/stuvla/locateanything \
    --image /home/ubun/project/stu_vla/locate-anything/test_data/pic/four_books.png \
    --task ground-text \
    --query '王国维文选' \
    --generation-mode fast \
    --seed 42 \
    --local-files-only
```

### 2. ground-single 启动

根据描述定位单个目标

```bash
python scripts/stl/test_stl_single_image.py  \
    --model-path /media/ubun/16T/checkpoints/stuvla/locateanything \
    --image /home/ubun/project/stu_vla/openpi-piper/test_data/pic/book_with_black_label.png \
    --task ground-single \
    --query "the book with black label" \
    --generation-mode fast \
    --verbose
```


## codex输出

已在 `src/openpi/models/stl/` 完成 LocateAnything 的**源码级接入**，可以直接使用 OpenPI 的 `.venv` 在同一进程内调用，不需要另启 LocateAnything 服务。

本次改动全部位于 `stl/`，外部仅新增了 3 个 `scripts/` 调用入口。你已有的 `piper_policy.py`、训练配置、`pyproject.toml` 和 `uv.lock` 均保持开始时的内容，已核对 SHA256。

目前接入的内容包括：

- **完整推理源码**：LocateAnything 主模型、Qwen2 解码器、MoonViT 视觉编码器、预处理、MTP/NTP/hybrid 解码、注意力与批处理实现，位于 [locate_anything/](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl/locate_anything)。
- **本地加载机制**：[loader.py](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl/loader.py:90) 显式加载项目内的模型类。权重目录只需提供权重、配置和 tokenizer 数据，不再执行其中的 Python 源码。
- **进程内 STL 接口**：[SceneTextLocator](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl/locator.py:551) 支持 PIL 图像或 HWC/RGB/uint8 数组，返回原始答案、原图像素框和点坐标；模型初始化一次后可重复调用。
- **原脚本功能**：保留全部 9 类任务、3 种解码模式、原始 prompt、坐标解析、绘图和 JSON 输出规则。

**复现结果已经做了实际 GPU 对照，但需要明确“完全一致”的条件：**

| 对照条件 | 实测结果 |
| --- | --- |
| 原实现与迁入实现都运行在 OpenPI 环境 | 11/11 完全一致 |
| 原实现与迁入实现都运行在原 LocateAnything 环境 | 11/11 完全一致 |
| 原环境结果直接与 OpenPI 环境结果比较 | 输入全部一致，8/11 输出一致 |

11 组测试包括全部 9 类任务的 hybrid 模式，以及文字定位的 fast、slow 模式。比较了输入张量、生成文本、坐标和绘图像素，报告见 [openpi_verified.json](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl/artifacts/parity/openpi_verified.json)。

跨环境的 `detect`、`detect-text`、`gui-box` 存在输出差异，因此不能承诺跨 PyTorch/CUDA 版本逐 token 一致。**同环境内的源码迁移等价性已经验证通过**，并且验证了权重目录不含 Python 文件时仍可独立运行。另外，19 项接口测试、静态检查及 SDPA 批处理冒烟测试均通过。

你现在可以在项目根目录直接运行：

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
.venv/bin/python scripts/test_stl_single_image.py \
  --model-path /media/ubun/16T/checkpoints/stuvla/locateanything \
  --image /home/ubun/project/stu_vla/locate-anything/test_data/pic/four_books.png \
  --task ground-text \
  --query '王国维文选' \
  --generation-mode hybrid \
  --seed 42 \
  --local-files-only
```

已实际得到：

```text
<ref>王国维文选.</ref><box><820><516><875><662></box><|im_end|>
```

生成文件：[标注图片](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl/artifacts/outputs/four_books_ground-text.png) · [JSON 结果](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl/artifacts/outputs/four_books_ground-text.json)。

依赖沿用 OpenPI 的 **PyTorch 2.7.1、Transformers 4.53.2、NumPy 1.26.4**，固定版本清单在 [requirements-openpi.txt](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl/requirements-openpi.txt)。单图不要求 FlashAttention、Magi、Decord 或 LMDB；STL 与 Pi05 的 JAX/PyTorch 模型共同导入检查已通过。

当前工作区已有的根 TOML 格式错误和 Decord wheel 平台标签问题，按你的修改范围限制没有处理，因此上述命令直接使用 `.venv/bin/python`。具体兼容说明、API 用法和对照命令均写在 [STL README](/home/ubun/project/stu_vla/openpi-piper/src/openpi/models/stl/README.md)。

本次交付的是可运行的 STL 模块及接口。由于没有修改训练或 policy 文件，关键帧分支尚未自动调用 STL，Pi05 与 STL 同时驻留 GPU 的完整训练流程也尚未验证。