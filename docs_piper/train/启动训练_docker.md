# openpi piper 训练

## docker 启动容器

使用以下命令构建 Docker 镜像并启动容器：

```bash
docker compose -f scripts/docker/compose.yml up --build
```

```bash
docker compose -f scripts/docker/compose.yml build \
    --build-arg GITHUB_PROXY_PREFIX=https://ghfast.top/ \
    --progress=plain \
    openpi_server
```

## 模型下载

下载 base 模型到本地

```bash
OPENPI_DATA_HOME=/path/to/download/pi05_base \
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
docker exec -it docker-openpi_server-1 bash -c "
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
    --config pi05_piper_full_finetune \
    --image-size 224 \
    --dataset-dir /path/to/lerobot/dataset \
    --dataset-repo-id piper_dataset_name \
    --base-model-dir /path/to/pi05_base \
    --checkpoint-dir /path/to/output/piper_full \
    --exp-name piper_full \
    --compute-norm-stats
"
```

`dataset-dir` 和 `base-model-dir` 是必须传递的参数


LoRA 微调：

```bash
docker exec -it docker-openpi_server-1 bash -c "
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
    --config pi05_piper_lora_finetune \
    --image-size 448 \
    --dataset-dir /path/to/lerobot/dataset \
    --dataset-repo-id piper_dataset_name \
    --base-model-dir /path/to/pi05_base \
    --checkpoint-dir /path/to/output/piper_lora \
    --exp-name piper_lora \
    --compute-norm-stats
"
```

恢复/覆盖 训练:

```bash
docker exec -it docker-openpi_server-1 bash -c "
uv run scripts/train_piper.py \
    --config pi05_piper_lora_finetune \
    --image-size 448 \
    --dataset-dir /path/to/lerobot/dataset \
    --dataset-repo-id piper_dataset_name \
    --base-model-dir /path/to/pi05_base \
    --checkpoint-dir /path/to/output/piper_lora \
    --exp-name piper_lora \
    --compute-norm-stats \
    --resume
"
```

--overwrite 会清除同一输出目录里的已有训练结果，只应在确认不需要旧 checkpoint 时使用。
