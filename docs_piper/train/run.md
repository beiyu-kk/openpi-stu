## 环境安装

1. uv方式

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

2. docker 方式

设置路径，需要 docker compose v2

```bash
export OPENPI_DATA_HOME=~/.cache/openpi
```

使用以下命令构建 Docker 镜像并启动容器：

```bash
docker compose -f scripts/docker/compose.yml up --build
```

## 模型下载

下载 base 模型到本地

```bash
OPENPI_DATA_HOME=/media/ubun/16T/checkpoints/openpi \
uv run python -c '
from openpi.shared.download import maybe_download
print(maybe_download("gs://openpi-assets/checkpoints/pi05_base"))
'
```

## 启动训练

全量微调和 LoRA 微调的动作块长度统一为 30 帧；当前数据集为 30 FPS，因此每个动作块覆盖约 1 秒。

### 保留原始像素，按最长边补成方形

选择后缀为 `_pad_to_square` 的命名配置启用动态尺寸。每张图以自身最长边为边长，
居中补黑边，不缩放、不裁剪：640×480 → 640×640，1280×720 → 1280×1280。
不同相机、不同请求、不同训练 batch 可以使用不同尺寸。奇数补边多出的 1 像素放在右侧或底部。

| 模型 | 普通全量训练 | TT-RTC 全量训练 |
| --- | --- | --- |
| π0.5 | `pi05_piper_full_finetune_pad_to_square` | `pi05_piper_full_finetune_rtc_pad_to_square` |
| π0-FAST | `pi0_fast_piper_full_finetune_pad_to_square` | `pi0_fast_piper_full_finetune_rtc_pad_to_square` |

名称中的 `full` 换成 `lora` 即可选择对应的 LoRA 配置。将下方训练命令的 `--config`
替换为所需名称；部署也使用相同名称。该行为由 train config 的 `model.pad_to_square=True`
决定，启动脚本不提供 `--image-size` 或图像策略覆盖选项。

新增配置时，直接在 `model` 中设置，与 `image_resolution` 并列：

```python
model=pi0_fast.Pi0FASTConfig(
    image_resolution=(224, 224),
    pad_to_square=True,
)
```

π0／π0.5 的 `Pi0Config` 也使用同名参数。`pad_to_square` 默认 `False`：
此时 `image_resolution` 是固定画布，`image_resize_mode` 选择缩放补边或仅补边；
设为 `True` 时优先按原图最长边动态补方形，`image_resize_mode` 不生效，
`image_resolution` 仅用于位置编码参考网格和模型初始化。

组 batch 时，同一相机的图片还会补到该 batch 内的最大方形边长以便堆叠，
区域标注同步补边。例如同 batch 的 640×640 和 1280×1280 会组成
`[batch,1280,1280,3]`；其他相机可以有自己的尺寸。所有额外处理均为 padding。
训练关闭几何缩放增强，保留颜色增强。

此模式目前支持 JAX π0／π0.5 和 π0-FAST，支持 RTC，暂不支持 `spatial_2x2` token merger。
SigLIP 仍使用 14×14 patch，非整除边长会继续在右侧／底部补到 patch 边界，保留全部像素。
模型会按实际 patch 网格插值位置编码；配置中的 `image_resolution=(224,224)`
仅定义位置参数的参考网格和初始化样本尺寸，不限制输入画布。
参数与优化器状态形状保持固定，不同输入尺寸触发相应的 JAX 编译。
显存和计算量随实际尺寸增加；同 batch 混入大图也会增大该相机的整个 batch 画布。

固定画布策略仍可选 `_image640_pad_only`：它始终补到 640×640，超过画布的输入报错。


全量微调：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
    --config pi05_piper_full_finetune \
    --dataset-dir /media/ubun/16T/Dataset/piper_data/recognize_book_label_color_lerobot_v2.1 \
    --dataset-repo-id recognize_book_label_color \
    --base-model-dir /media/ubun/16T/checkpoints/openpi/openpi-assets/checkpoints/pi05_base/params \
    --checkpoint-dir /media/ubun/16T/checkpoints/openpi/piper_book_color \
    --exp-name piper_full_book_color \
    --compute-norm-stats \
    --overwrite
```

`dataset-dir` 和 `dataset-repo-id` 是必须传递的参数；不传 `base-model-dir` 时使用所选 TrainConfig 的基础权重路径。


LoRA 微调：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
    --config=pi05_piper_lora_finetune_image448 \
    --dataset-dir /media/ubun/16T/Dataset/piper_data/recognize_book_label_color_lerobot_v2.1 \
    --dataset-repo-id recognize_book_label_color \
    --base-model-dir /media/ubun/16T/checkpoints/openpi/openpi-assets/checkpoints/pi05_base/params \
    --checkpoint-dir /media/ubun/16T/checkpoints/openpi/piper_book_label_color \
    --exp-name piper_lora_book_label_color \
    --compute-norm-stats \
    --overwrite
```
