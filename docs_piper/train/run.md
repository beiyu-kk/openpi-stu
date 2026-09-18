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

`--config` 直接选择 TrainConfig，`--image-size` 独立指定图像边长（14 的正整数倍，例如 224、336、448）。两个 Piper 配置默认均为 224；恢复训练和部署时需保持与训练一致的尺寸。

全量微调：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
    --config pi05_piper_full_finetune \
    --image-size 224 \
    --dataset-dir /media/ubun/16T/Dataset/piper_data/recognize_book_label_color_lerobot_v2.1 \
    --dataset-repo-id recognize_book_label_color \
    --base-model-dir /media/ubun/16T/checkpoints/openpi/openpi-assets/checkpoints/pi05_base/params \
    --checkpoint-dir /media/ubun/16T/checkpoints/openpi/piper_book_color \
    --exp-name piper_full_book_color  \
    --compute-norm-stats \
    --overwrite
```

`dataset-dir` 和 `dataset-repo-id` 是必须传递的参数；不传 `base-model-dir` 时使用所选 TrainConfig 的基础权重路径。


LoRA 微调：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
    --config pi05_piper_lora_finetune \
    --image-size 448 \
    --dataset-dir /media/ubun/16T/Dataset/piper_data/recognize_book_label_color_lerobot_v2.1 \
    --dataset-repo-id recognize_book_label_color \
    --base-model-dir /media/ubun/16T/checkpoints/openpi/openpi-assets/checkpoints/pi05_base/params \
    --checkpoint-dir /media/ubun/16T/checkpoints/openpi/piper_book_label_color \
    --exp-name piper_lora_book_label_color  \
    --compute-norm-stats \
    --overwrite
```
