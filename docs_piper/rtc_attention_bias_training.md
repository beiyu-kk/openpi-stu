# Pi0.5 RTC bbox attention-bias 训练说明

`pi05_piper_full_finetune_rtc_attention_bias` 是在现有 Pi0.5 全量 RTC
微调配置上的可选扩展。它只在训练阶段读取 LeRobot 中嵌入的 LocateAnything bbox，并将
目标区域对应图像 patch 的 attention logits 增加一个正偏置；动作损失、Pi0.5
模型主体、RTC 时延训练和推理接口保持既有实现。

普通 LeRobot 数据训练继续使用已有配置，例如
`pi05_piper_full_finetune` 或 `pi05_piper_full_finetune_rtc`。这些配置没有
`region_guidance`，不会读取任何标注文件，也不会产生区域 attention bias。

## 工作方式

每个 LeRobot sample 直接携带右腕书本框和书名框。对通过审核的框：

```text
bbox (640 x 480 原图)
→ 二值区域 mask
→ 与 ResizeImages 相同的 resize-with-pad
→ 14 x 14 patch 覆盖率
→ action query 到 right_wrist image key 的 attention logit bias
```

patch 覆盖率位于 `[0, 1]`，表示该 patch 中属于 bbox 区域的像素比例。对每个
有效 sample，bias 是：

```text
attention_logit_new = attention_logit + attention_bias_scale * patch_coverage
```

bias 只作用在指定 Transformer 层和 query head 的 action token 到右腕图像 token
连接上。默认随机保留一半有效 sample，并在训练进度 30% 到 70% 间从初始强度退火到
零。因此模型不会在部署时依赖 bbox；`train=False` 和策略推理时该逻辑不执行。

## 1. 将精简标注写入 LeRobot

在工程根目录运行一次：

```bash
uv run --frozen scripts/embed_annotation_v2.py \
  --dataset /media/ubun/16T/Dataset/piper_eva_data/recognition/recognize_book_name/success/origin \
  --annotations /media/ubun/16T/Dataset/piper_eva_data/recognition/recognize_book_name/success/annotations/locateanything_vlm_full_v2_final
```

每个 episode parquet 只增加：

```text
observation.region.hand_right.book_bbox_xyxy
observation.region.hand_right.title_bbox_xyxy
```

脚本会逐帧校验标注与原数据身份，并在替换前生成轻量备份。这份数据已经完成写入，
正常启动训练时不要重复执行转换。字段声明可在 `origin/meta/info.json` 查看，逐帧坐标
位于 `origin/data/chunk-*/episode_*.parquet`；具体查看命令见
[标注适配层说明](annotation_adapter.md#在哪里查看新增字段)。

## 2. 启动训练

当前数据的 LeRobot 根目录是 `success/origin`。使用新的 RTC attention-bias
配置时会自动读取嵌入字段，不再需要额外标注路径。

```bash
uv run --frozen scripts/train_piper.py \
  --config pi05_piper_full_finetune_rtc_attention_bias \
  --dataset-dir /media/ubun/16T/Dataset/piper_eva_data/recognition/recognize_book_name/success/origin \
  --dataset-repo-id recognize_book_name \
  --region-annotation-mode book \
  --exp-name recognize_book_book_bias \
  --checkpoint-dir /media/ubun/16T/checkpoints/openpi/recognize_book_book_bias \
  --compute-norm-stats
```

上面的配置本身已经启用 region guidance，不需要额外传 `--region-guidance`。训练启动时
会检查 `--dataset-dir` 内是否存在两个 bbox 字段；没有 sidecar 参数，也不会从
`success/annotations` 读取训练数据。

首次 RTC 训练应使用 `--compute-norm-stats`，它会按 RTC 训练 episode split
生成归一化统计。之后相同数据和 split 可复用该统计文件。

如果基础权重不在默认位置，附加：

```bash
--base-model-dir /media/ubun/16T/checkpoints/openpi/openpi-assets/checkpoints/pi05_base/params
```

## 3. bbox 选择与推荐对照实验

`--region-annotation-mode` 决定从同一数据样本中取哪一类框：

| 值 | 训练区域 |
| --- | --- |
| `book` | 整本书 bbox |
| `title` | 书名 bbox |
| `both` | 两类 bbox 的并集 |
| `none` | 不产生有效区域，用于检查数据链路 |

仅 `status=matched` 且 `review_status=accepted` 的框会产生 bias。无标注、
`needs_review`、`skipped_unconfirmed` 的帧仍参与正常动作训练，`region_valid`
为 false。建议以一致的初始权重、seed、训练步数和 RTC split 分别训练 baseline、
`book`、`title`、`both`，再比较策略评估结果。

## 4. 调整 bias 超参数

新配置的默认值为：`attention_bias_scale=0.5`、`keep_probability=0.5`、从 30% 训练进度开始
退火、70% 时变为零；默认选择中间两层和前四分之一 attention heads。

若需要覆盖这些默认值，在命令中加入 `--region-guidance`，再按需传入：

```bash
--attention-bias-scale 0.5 \
--region-keep-probability 0.5 \
--region-decay-start 0.3 \
--region-decay-end 0.7 \
--region-layers 8 9 \
--region-heads 0 1
```

同一 checkpoint 用 `--resume` 继续训练时，区域模式、bias 超参数、图像尺寸、
数据根目录和总训练步数必须与首次运行一致。训练目录中的
`region_guidance.json` 保存这些设置，用于阻止不一致的续训。

## 相关文件

- [标注适配层说明](annotation_adapter.md)：精简字段和迁移说明。
- `scripts/embed_annotation_v2.py`：嵌入式迁移工具。
- `scripts/train_piper.py`：本地 Piper 训练入口。
- `src/openpi/training/config.py`：命名训练配置。
