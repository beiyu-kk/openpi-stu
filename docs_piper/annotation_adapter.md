# LocateAnything 标注适配层

这个适配层把 `annotation.v2` 的逐帧书本/书名框精简后直接写入原始 LeRobot
episode parquet。训练时直接读取这个原始数据集，不生成也不读取 sidecar。原始
LocateAnything 标注保持只读；修改前的 parquet 和 `meta/info.json` 会自动备份到
数据集旁边。

## 输入与输出

输入目录需要包含：

```text
<annotation-v2>/
├── episodes/
│   ├── episode_000000/
│   │   ├── manifest.json
│   │   └── annotations.jsonl
│   └── ...
```

每帧从 `object_boxes` 读取整本书框，从 `text_boxes` 读取书名框。只新增两个字段：

```text
observation.region.hand_right.book_bbox_xyxy   float32[4]
observation.region.hand_right.title_bbox_xyxy  float32[4]
```

坐标顺序为 `[x1, y1, x2, y2]`，单位是右腕原图像素。只有
`status=matched` 且 `review_status=accepted` 的框会写入；无框、拒绝或待复核帧写为
`[-1, -1, -1, -1]`。有效性直接由坐标推导，不重复保存布尔值、状态、文本、query、
相机名、尺寸、frame/index/timestamp 等已有或仅供审计的信息。

## 写入原始数据集

在 `openpi-stu` 根目录运行：

```bash
uv run --frozen scripts/embed_annotation_v2.py \
  --dataset /media/ubun/16T/Dataset/piper_eva_data/recognition/recognize_book_name/success/origin \
  --annotations /media/ubun/16T/Dataset/piper_eva_data/recognition/recognize_book_name/success/annotations/locateanything_vlm_full_v2_final
```

这份 `recognize_book_name` 数据已经完成写入，不需要在每次训练前重复执行。只有重新
生成标注或恢复了未嵌入字段的备份后，才需要再次运行转换。

脚本先暂存并验证全部 episode，再备份和替换。备份默认位于
`success/origin.region_backup_<UTC时间>`，只含被修改的约 3 MB parquet 与
`meta/info.json`，不复制视频。脚本只读取每个 episode 的 `annotations.jsonl`。

## 在哪里查看新增字段

字段声明位于：

```text
/media/ubun/16T/Dataset/piper_eva_data/recognition/recognize_book_name/success/origin/meta/info.json
```

逐帧字段值位于各 episode 的 parquet，例如：

```text
/media/ubun/16T/Dataset/piper_eva_data/recognition/recognize_book_name/success/origin/data/chunk-000/episode_000000.parquet
```

查看 schema 声明：

```bash
jq '.features
  | with_entries(select(.key | startswith("observation.region.")))' \
  /media/ubun/16T/Dataset/piper_eva_data/recognition/recognize_book_name/success/origin/meta/info.json
```

查看第一个 episode 的前五帧实际值：

```bash
uv run --frozen python -c "import pyarrow.parquet as pq; p='/media/ubun/16T/Dataset/piper_eva_data/recognition/recognize_book_name/success/origin/data/chunk-000/episode_000000.parquet'; keys=['observation.region.hand_right.book_bbox_xyxy','observation.region.hand_right.title_bbox_xyxy']; print(pq.read_table(p, columns=keys).slice(0, 5).to_pandas().to_string(index=False))"
```

`meta/info.json` 只记录字段类型，真正逐帧变化的 bbox 坐标在 parquet 中。它们不是
单独的 JSON 文件，也不在原标注目录中。

详细的 Pi0.5 RTC attention-bias 训练命令、机制和实验建议见
[`rtc_attention_bias_training.md`](rtc_attention_bias_training.md)。

## 训练模式

`--region-annotation-mode` 支持四种选择：

```text
book   只使用整本书框
title  只使用书名框
both   两类框做并集
none   不生成有效区域
```

例如使用整本书框：

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

这个命名配置已启用 region guidance，不需要再传 `--region-guidance`。只有需要覆盖
默认 bias 超参数时才同时传该开关和对应的 `--region-*` 参数。如果不使用配置中的
默认基础权重，再增加 `--base-model-dir /实际路径/pi05_base/params`。

把 `book` 换成 `title` 即可进行书名框实验。建议分别训练不同实验，不要在同一个
checkpoint 的续训过程中切换模式。

## 与模型的连接

训练 loader 直接从当前 LeRobot sample 读取 bbox。当前标注相机是
`observation.images.hand_right`，因此有效 mask 只写入
`right_wrist_0_rgb`；base 和 left wrist mask 为零且无效。

bbox 先在原始 `640×480` 坐标中栅格化，再按照 Pi0.5 的 resize-with-pad 几何映射到
训练图像。模型随后按 14×14 patch 求覆盖率，并在训练期将 coverage 作为配置中
`action`、`image`、`prompt` query 到图像 key 的 attention bias。推理和 `train=False`
不使用 bias。query 范围由 `RegionGuidanceConfig.query_types` 显式控制；默认值仍是
`("action",)`。

## 验证

转换完成后可验证：

```bash
uv run --frozen python -c "from openpi.training.region_annotations import validate_embedded_annotations as v; print(v('/media/ubun/16T/Dataset/piper_eva_data/recognition/recognize_book_name/success/origin'))"
```

这份数据预期为 100 个 episode、73,264 帧，其中 31,184 帧有 accepted book/title
框。迁移时会逐帧校验 episode、frame、全局 index 和 timestamp；训练启动时再次校验
feature schema、episode 数和总帧数。
