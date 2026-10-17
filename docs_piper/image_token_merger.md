# 视觉 patch 合并配置

448×448 输入经过 SigLIP/14 后仍然先得到 32×32 个原始 patch。启用
`image_token_merger="spatial_2x2"` 后，模型在 SigLIP 编码器输出之后把相邻的
2×2 patch 拼接、归一化并投影回原始 embedding 宽度，因此每路相机得到
16×16=256 个视觉 token。新 Piper 配置同时将永久无效的左腕相机槽位从
Transformer 输入中排除，因此两路有效相机合计 512 个视觉 token。

合并不改变 SigLIP 的 patch 提取和位置编码，所以 448 分辨率的基础 checkpoint
仍然可以通过 `resize_siglip_posemb=True` 加载。merger 是新增的可训练模块；从旧
checkpoint 开始训练时，loader 会保留它的随机初始化参数，其他同名参数照常加载。

区域 attention bias 使用同一个 2×2 网格：先计算 14×14 原始 patch 的像素覆盖率，
再对每个相邻四 patch 取平均，生成 256 个 bias key。这样 bias 长度和合并后的图像
token 数量始终一致。

原有配置默认 `image_token_merger="none"` 且保留三个相机槽位，token 数量和旧
checkpoint 行为不变。
可直接使用新实验配置：

```bash
uv run scripts/train_piper.py \
  --config pi05_piper_full_finetune_rtc_attention_bias_448_d9_scale_0_5_merge2 \
  --dataset-dir /path/to/lerobot_dataset \
  --dataset-repo-id recognize_book_name \
  --region-annotation-mode book
```

该配置每路 448×448 图像从 1024 个 token 降为 256 个；语言 token、动作 token、
RTC 延迟和区域 bias 的训练逻辑保持原有实现。
