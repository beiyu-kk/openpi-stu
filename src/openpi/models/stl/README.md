# STL: Scene Text Locator

LocateAnything 的模型源码已内置到 `openpi.models.stl`。STL 与 OpenPI 共用一个 Python
环境，在调用进程内加载模型；不需要单独启动终端服务。

本次范围是 STL 模块及独立调用脚本。没有修改 Piper policy、Pi05 模型、训练配置、
数据流、根 `pyproject.toml` 或 `uv.lock`。因此，当前不会自动触发关键帧定位，也不会改变训练行为。

## 目录

```text
src/openpi/models/stl/
  config.py                 STL 运行配置
  locator.py                LocateAnythingWorker / SceneTextLocator
  loader.py                 仅使用本地源码加载模型与预处理器
  types.py                  STLResult，原始答案与原图像素坐标
  tasks.py                  与原脚本一致的 9 类任务
  parsing.py                原始 box / point 解析规则
  visualization.py          原始绘图规则
  cli.py                    单图命令行实现
  locate_anything/           完整推理源码，包括 batch_utils / kernel_utils
  requirements-openpi.txt    已验证的统一环境依赖版本
  environment.py            依赖与 OpenPI 共同导入检查
  verify.py                 GPU 原实现对照工具
  tests/                    无需权重的接口回归测试
  UPSTREAM.md               源码来源、适配说明、许可证位置
  upstream_manifest.json    原始源码 SHA256
  artifacts/                输出、对照报告、测试缓存；不纳入 Git
```

`scripts/` 下只新增 `test_stl_single_image.py`、`check_stl_environment.py`、`verify_stl.py`，
这些入口的实际实现都在 STL 内部。

## 统一环境

已在当前 OpenPI `.venv` 中验证：Python 3.11.16、PyTorch 2.7.1/CUDA 12.6、
torchvision 0.22.1、Transformers 4.53.2、tokenizers 0.21.1、NumPy 1.26.4、
Pillow 11.2.1、PEFT 0.12.0、Accelerate 1.5.2。

在仓库根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/check_stl_environment.py
```

当前环境已具备单图所需依赖。其他机器需先准备 OpenPI 环境，再安装 STL 的固定版本清单：

```bash
uv --no-config pip install --python .venv/bin/python \
  -r src/openpi/models/stl/requirements-openpi.txt
```

此命令直接操作指定解释器，不修改根依赖文件；不要为 STL 升级 OpenPI 的 Transformers
或覆盖 OpenPI 已使用的 Transformers 适配文件。清单固定直接依赖，不代替完整 CUDA 系统环境锁定。
本次只验证现有环境，没有执行上述重装命令。

单图默认采用文本 SDPA，视觉 `auto` 在没有 FlashAttention 时采用 SDPA。
不要求安装 `flash-attn`、`magi_attention`、`decord` 或 `lmdb`。
LMDB 输入和视频读取需要各自后端，见依赖清单注释。可选加速后端尚未做 GPU 验证。

当前工作区另有两个已有环境问题：根 `pyproject.toml` 的 OpenCV 依赖行末包含多余字符，
会使 `uv run`/`uv sync` 的 TOML 解析失败；`uv pip check` 会报告已安装 Decord wheel
的平台标签不匹配。两者均未在本次范围外修改。直接使用 `.venv/bin/python` 的 STL 单图入口
已通过验证；`check_stl_environment.py` 检查的是 STL 所需版本及共同导入，不代表全部已安装包均通过检查。

## 单图复现

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

已得到与原样例一致的答案：

```text
<ref>王国维文选.</ref><box><820><516><875><662></box><|im_end|>
```

图像大小为 `628 x 423`，转换后的像素框约为
`(514.96, 218.268, 549.5, 280.026)`。模型输出坐标按 0 到 1000 归一化，
API/JSON 中的坐标是原始输入图像的像素值，保留原算法的浮点表示，不做取整、裁剪、NMS 或额外置信度推断。

默认输出：

```text
src/openpi/models/stl/artifacts/outputs/four_books_ground-text.png
src/openpi/models/stl/artifacts/outputs/four_books_ground-text.json
```

支持 `--output` 指定位置，`--verbose` 输出原模型统计。
原有参数、JSON 字段、绘图颜色与坐标规则保持一致。
`--seed` 在模型加载完成后、推理开始前设置 PyTorch 随机种子；默认不固定种子，沿用原脚本采样行为。

| task | 必需参数 | 功能 |
| --- | --- | --- |
| `detect` | `--categories book cup` | 类别检测 |
| `ground-single` | `--query` | 单目标短语定位 |
| `ground-multi` | `--query` | 多目标短语定位 |
| `detect-text` | 无 | 场景文字检测 |
| `ground-text` | `--query` | 指定文字定位，STL 的主要任务 |
| `gui-box` | `--query` | GUI 框定位 |
| `gui-point` | `--query` | GUI 点定位 |
| `point` | `--query` | 点定位 |
| `custom` | `--query` | 原始自定义 prompt |

三种原解码模式 `fast`、`slow`、`hybrid` 均保留。默认 `hybrid`、2048 new tokens、
bfloat16、temperature 0.7、top-p 0.9、top-k 0、repetition penalty 1.1，与原脚本/Worker 相同。

## 进程内调用

```python
from PIL import Image
from openpi.models.stl import STLConfig, SceneTextLocator

stl = SceneTextLocator(STLConfig(
    model_path="/media/ubun/16T/checkpoints/stuvla/locateanything",
    device="cuda:0",
    local_files_only=True,
))
image = Image.open("/path/to/frame.png").convert("RGB")
result = stl.locate(image, query="王国维文选")
print(result.answer)
print(result.boxes)
```

模型在构造时加载一次，后续帧复用同一实例。输入也可以是 HWC/RGB/uint8 NumPy 数组。
不要把 OpenPI 中已归一化的 float 图像、CHW 张量或 OpenCV BGR 数组直接传入；
STL 应获取对应原始 RGB 帧，以保持文字细节和坐标语义。
没有匹配结果时 `boxes`/`points` 为空，原始 `answer` 仍保留。

目前默认将 STL 权重设为 eval、禁用梯度，适合作为 Pi05 的冻结定位模块。
未来接入关键帧分支时应在进程初始化阶段创建实例，再对关键帧调用 `locate`；
是否将框转为 ROI、mask 或视觉 token 属于后续训练接口工作，本次未实现。
实际 Pi05 训练与 STL 同时驻留 GPU 的显存、吞吐和训练梯度路径尚未测试。

高级接口 `LocateAnythingWorker` 保留原任务方法、视觉提示参数和 `predict_batch`。
默认批量调用按单图串行执行；`use_batch_runtime=True, attn="sdpa", vision_attn="sdpa"`
启用内置 hybrid 批处理。该运行时每个进程只支持一种模型配置，不应跨线程并发调用。
视觉提示能力还依赖具体权重；上游说明现有公开 3B 权重并不保证视觉提示效果。

## 对照验证

2026-09-09 在本机 NVIDIA RTX 4090 上对 `four_books.png` 运行 11 组 GPU 测试：
9 类任务的 hybrid 模式，加 ground-text 的 fast 与 slow 模式；seed 42，512 new tokens，
每组均生成结束标记。对比真实输入张量、原始答案、box/point 和绘图像素 SHA256。

| 对照条件 | 结果 |
| --- | --- |
| 原实现与迁入实现在 OpenPI 环境中运行 | 11/11 全项完全一致 |
| 原实现与迁入实现在原 LocateAnything 环境中运行 | 11/11 全项完全一致 |
| 原环境输出与 OpenPI 环境输出直接比较 | 输入全部一致，8/11 输出一致 |

原环境是 Python 3.10、PyTorch 2.14.0、Transformers 4.57.1、Pillow 12.3.0。
跨环境的 `detect`、`detect-text`、`gui-box` 有输出差异。
因此“完全一致”的验证结论限定于相同权重、输入、软件环境、attention 和采样设置；
不承诺跨 PyTorch/CUDA 版本、硬件或随机状态逐 token 一致。

本次完整报告位于 `artifacts/parity/openpi_verified.json` 和
`artifacts/parity/local_conda.json`，两者均为 `exact_match: true`；
跨环境差异保存在 `artifacts/parity/local.json`。
另已通过 19 项无权重接口测试，以及 `cuda:0`/SDPA 下文字框与目标点的双请求批处理冒烟测试。
批处理只验证可运行性，未将其输出等同于单图解码结果。

OpenPI 环境中的原实现对照仅在测试进程内显式设置子模型 SDPA，以匹配原环境的实际后端；
没有改动外部源码文件。正式 STL loader 自身完成这一兼容。

验证工具用法：

```bash
# 用原始源码在 OpenPI 环境生成参考报告。
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
HF_MODULES_CACHE="$PWD/src/openpi/models/stl/artifacts/reference_modules" \
.venv/bin/python src/openpi/models/stl/verify.py \
  --implementation original --reference-attention-compat \
  --reference-root /home/ubun/project/stu_vla/locate-anything \
  --model-path /media/ubun/16T/checkpoints/stuvla/locateanything \
  --image /home/ubun/project/stu_vla/locate-anything/test_data/pic/four_books.png \
  --report src/openpi/models/stl/artifacts/verify/reference.json

# 仅使用本地源码，并让权重目录不含任何 Python 文件，逐项比较。
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
.venv/bin/python scripts/verify_stl.py --assets-only \
  --model-path /media/ubun/16T/checkpoints/stuvla/locateanything \
  --image /home/ubun/project/stu_vla/locate-anything/test_data/pic/four_books.png \
  --report src/openpi/models/stl/artifacts/verify/local.json \
  --compare-report src/openpi/models/stl/artifacts/verify/reference.json
```

`--assets-only` 在报告旁创建仅含模型数据文件的符号链接目录，不复制数 GB 权重。
本地实现会检查模型/processor 的源码路径并拒绝动态 checkpoint Python 模块；
存在差异时保存完整报告并以非零状态退出，不放宽坐标容差或自动忽略不同结果。

无权重回归测试：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -c /dev/null --rootdir=src/openpi/models/stl \
  -o cache_dir=artifacts/pytest-cache \
  --basetemp=src/openpi/models/stl/artifacts/pytest-tmp \
  src/openpi/models/stl/tests -q
```

测试覆盖任务 prompt、坐标规则、空结果、输入格式、模型复用、checkpoint 预处理配置、
导入隔离和批处理配置保护。单张图的等价测试不是通用 OCR/定位质量评测。
