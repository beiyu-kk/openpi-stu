# Piper训练

## 环境安装

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
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

`--config` 直接选择 `src/openpi/training/config.py` 中的 TrainConfig。图像尺寸通过 `--image-size` 独立设置，必须为 14 的正整数倍，例如 224、336、448。两个 Piper 配置默认均为 224；不再使用带 `_336`、`_448` 后缀的配置名。恢复训练时保持尺寸一致，部署时传对应的 `--policy.image-size`。已有 448 分辨率的 LoRA 实验需要显式传 `--image-size 448`。

全量微调：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
    --config pi05_piper_full_finetune \
    --image-size 224 \
    --dataset-dir /path/to/lerobot/dataset \
    --dataset-repo-id piper_dataset_name \
    --base-model-dir /path/to/pi05_base \
    --checkpoint-dir /path/to/output/piper_full \
    --exp-name piper_full \
    --compute-norm-stats
```

`dataset-dir` 和 `dataset-repo-id` 是必须传递的参数；不传 `base-model-dir` 时使用所选 TrainConfig 的基础权重路径。


LoRA 微调：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
    --config pi05_piper_lora_finetune \
    --image-size 448 \
    --dataset-dir /path/to/lerobot/dataset \
    --dataset-repo-id piper_dataset_name \
    --base-model-dir /path/to/pi05_base \
    --checkpoint-dir /path/to/output/piper_lora \
    --exp-name piper_lora \
    --compute-norm-stats
```

恢复/覆盖 训练:

```bash
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
```

--overwrite 会清除同一输出目录里的已有训练结果，只应在确认不需要旧 checkpoint 时使用。


## 参数行为说明

--dataset-dir 和 --dataset-repo-id 都是必填参数。前者是本地 LeRobot 数据集根目录，后者是写入 checkpoint
时用于标识数据集 assets 的稳定逻辑名称；缺少任意一个参数，训练脚本都会直接报错。

--base-model-dir 可以传以下任意形式：

```text
  /path/to/pi05_base
  /path/to/pi05_base/params
  gs://openpi-assets/checkpoints/pi05_base
  gs://openpi-assets/checkpoints/pi05_base/params
```

Norm Stats 行为

使用 --compute-norm-stats 时：

1. 默认检查 <dataset-dir>/norm_stats.json。
2. 文件已经存在则直接复用，不重复计算。
3. 文件不存在则自动遍历数据集并计算。
4. 计算完成后再启动训练。
5. stats 会随 checkpoint 保存，供后续推理使用。

指定其他目录：

--norm-stats-dir /path/to/piper_norm_stats

程序将检查或生成：

/path/to/piper_norm_stats/norm_stats.json

不传 --compute-norm-stats 时，如果统计文件不存在，脚本会在模型初始化之前明确报错。也可以
显式传：

--no-compute-norm-stats

限制统计计算帧数可使用：

--compute-norm-stats --max-norm-frames 10000

正常训练建议不设置 --max-norm-frames，使用完整数据集计算。
