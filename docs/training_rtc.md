# Piper Training RTC

Training RTC 由 `src/openpi/training/config.py` 中的 `TrainConfig.training_rtc` 统一管理。
`None` 表示普通训练；`TrainingRTCConfig(...)` 表示启用 RTC。训练入口仍是
`scripts/train_piper.py`，通过 `--config` 选择，不需要另加 RTC 开关。

| 配置 | 训练方式 |
| --- | --- |
| `pi05_piper_full_finetune` | 原有全量微调 |
| `pi05_piper_lora_finetune` | 原有 LoRA 微调 |
| `pi05_piper_full_finetune_rtc` | 全量 Training RTC |
| `pi05_piper_lora_finetune_rtc` | 动作侧 LoRA Training RTC |

当前实现支持 **JAX π0.5、Piper 单臂 7 维 all-delta 动作**。可以与 `--image-size`
和 `--region-guidance` 组合。PyTorch 训练／模型会拒绝 RTC 配置。

## 启动

```bash
uv run --no-sync scripts/train_piper.py \
    --config pi05_piper_lora_finetune_rtc \
    --dataset-dir /path/to/lerobot_dataset \
    --dataset-repo-id piper_task \
    --base-model-dir /path/to/pi05_base/params \
    --checkpoint-dir /path/to/checkpoints/piper_task_rtc_lora \
    --exp-name piper_task_rtc_lora \
    --compute-norm-stats \
    --image-size 224 \
    --batch-size 16 \
    --no-wandb-enabled
```

全量 RTC 将 `--config` 改为 `pi05_piper_full_finetune_rtc`，使用新的实验输出目录。
两个配置的默认 horizon 都是 30，模型动作维度是 32，训练 10,000 步。
全量默认 batch 32，LoRA 默认 batch 16；`--batch-size`、`--fsdp-devices` 等继续按机器容量设置。
没有为任何特定 GPU 硬编码设备数量。正式训练前可在独立输出目录使用
`--num-train-steps 2 --num-workers 0` 做短训练检查。

RTC 的算法和验证参数在配置文件中修改，例如：

```python
training_rtc=TrainingRTCConfig(
    finetune_mode="lora",  # 全量配置使用 "full"
    max_delay=4,
    delay_weights=None,
    loss_reduction="official",
    time_distribution="beta",
    inherited_lr_scale=0.2,
    validation_fraction=0.1,
    split_seed=42,
    validation_interval=1000,
    validation_batches=4,
    norm_source="train_split",
)
```

`max_delay` 包含上界：4 表示前缀长度 0–4，且必须小于 horizon。
默认 `P(d) ∝ exp(-d)`；自定义权重必须有 `max_delay+1` 项，有限、非负且总和为正。
如需固定 episode 划分，同时设置 `train_episodes` 和 `validation_episodes`，二者应互斥且覆盖全部 episode。
`validation_interval=0` 只关闭周期验证，不会取消训练／验证划分。

`resolve_training_config()` 在入口参数覆盖后派生模型运行参数、冻结规则及优化器。
用户修改 `training_rtc`，不需要同时修改 `model.simulated_delay`。
不要直接在普通模型配置中设置 `simulated_delay` 来绕过统一配置。

## LoRA 与全量的区别

RTC LoRA 的可训练范围与本仓库原有的普通 LoRA 不同：

| 模块 | RTC LoRA | RTC 全量 |
| --- | --- | --- |
| SigLIP、PaliGemma 主干 | 冻结 | 训练 |
| 已有 PaliGemma LoRA | 保留并冻结 | 不允许未经合并的 adapters |
| Action expert 原始 attention／FFN | 冻结 | 训练 |
| Action expert LoRA（rank 32） | 训练 | 不创建 |
| Action expert 的 adaRMS Dense | 训练 | 训练 |
| 时间 MLP、动作输入／输出投影 | 训练 | 训练 |

LoRA 使用真正的 AdamW 参数分组：LoRA 峰值学习率 `2.5e-5`，继承的可训练参数
使用 `0.2` 倍，即 `5e-6`。全量使用统一 AdamW，峰值 `1e-5`。
学习率计划位于两个配置的 `lr_schedule` 中。这些值是调参起点，不代表当前任务的最优值。
两个 RTC 配置都关闭 EMA，验证和推理参数保存使用当前模型。

初始化时先检查底座参数元数据。有 VLM adapters 时保留其结构并冻结；底座没有时不创建
随机 VLM adapters。专用权重加载器拒绝静默丢弃已有 adapters，并保留自定义图像尺寸所需的
SigLIP 位置编码插值能力。全量模式加载含 LoRA 的底座会报错，需要先单独合并 adapters。

## 动作与 loss

每个训练样本采样前缀长度 `d`，前 `d` 个真实动作保持干净，时间条件设为 0；其余动作
使用相同的噪声时间 `t`。沿用 OpenPI 的 `t=1` 噪声、`t=0` 干净动作约定。
前缀提供条件，不计入 loss。episode 尾部 padding 同时从 loss 和 attention 中排除。
图像增强开关不会关闭 RTC；验证使用相同 RTC 目标。

默认归约为：

```text
official_loss = 有效后缀位置的所有动作维平方误差之和 / 有效后缀 token 数
per_element_loss = official_loss / model_action_dim
```

对当前 32 维模型，两个 loss 相差 32 倍，不能直接拿 `official` 日志与旧训练逐元素 mean 比大小。
归约在全局 batch 上完成，不先对每个样本或每个设备单独归一化。没有有效目标的 batch
不更新参数或 Adam 状态；训练迭代计数仍推进。

每个动作窗口的坐标统一为 `action[t+i] - state[t]`，7 维均为 delta，包含夹爪。
归一化后 pad 到 32 维。日志的 `rtc/raw_mse` 是前 7 维的归一化 flow 速度误差，
不是机器人单位下的执行误差。

## 数据、归一化和验证

默认按 episode 固定划分 90% 训练、10% 验证，不按相邻帧随机拆分。
RTC 使用专门的数据集适配修复当前 LeRobot 版本在非连续 episode 子集上的窗口边界查询。
原始 episode/frame ID 保留，region guidance 标注仍按这些 ID 查找。

首次使用 `--compute-norm-stats` 会读取训练 episode 的 Parquet 数值列，计算 state 和
相对窗口起始 state 的 delta-action 统计量，不解码视频，并排除越界 padding。
默认保存到 `<dataset-dir>/rtc_norm_stats/<划分与horizon摘要>/`：

- `norm_stats.json`
- `norm_stats_manifest.json`

两个微调模式在相同划分和 horizon 下共享统计量。改变 delay 权重不需要重算统计量。
`--norm-stats-dir` 可以覆盖保存／读取位置；`train_split` 模式必须有匹配的统计 manifest，
旧的全数据统计不会仅因为文件存在就被复用。传入 `--compute-norm-stats` 时，入口也会
重新计算不匹配的 RTC 统计量；已有任务底座的统计量使用 `checkpoint` 模式继承。RTC 正式统计不接受 `--max-norm-frames`。

从已有任务 checkpoint 适配时，可以在配置中将 `norm_source` 改成 `"checkpoint"`，
复用底座 `assets/<asset_id>/norm_stats.json`。入口会尝试从底座 assets 中定位唯一的统计文件，
也可以用 `--norm-stats-dir` 显式指定其目录。此模式不重新拟合统计量；其训练／验证划分
只针对本次 RTC 适配，底座及继承统计量可能已经使用过这些验证 episode。
继承统计量必须符合相同的 7 维 all-delta 动作约定。

周期验证固定覆盖不同 episode／时间位置，对每个 delay 分别计算误差，跨 batch 累计
误差总和与有效 token 数后再相除。验证样本不足时缩小验证 batch，避免重复计算样本。
指标写入 W&B（启用时）及实验目录的 `rtc_validation.jsonl`；没有有效目标的 delay
指标记为 null，而不是零误差。

## 保存和恢复

实验目录保存 `training_rtc.json`；每个 checkpoint 的 `assets/rtc_manifest.json`
保存模型结构、RTC 参数、优化器／学习率设置、动作坐标、统计量摘要和 episode 划分。
参数、优化器状态和 step 沿用原 Orbax 保存机制，包括 step 0 的恢复。

相同实验在原命令追加 `--resume`。恢复时检查配置、数据及归一化一致性；数据迭代器的位置
沿用原训练框架的行为，不保证恢复到完全相同的下一批样本。

从普通模型开始 RTC，应使用原 checkpoint 的 `params` 作为 `--base-model-dir`，
并使用新的实验目录。不能直接对普通实验添加 RTC 配置并 `--resume`，也不能跨 full／LoRA 续训。

## 推理接口

现有 `scripts/serve_policy.py` 可以加载 RTC checkpoint：

```bash
uv run --no-sync scripts/serve_policy.py --port=8000 policy:checkpoint \
    --policy.config=pi05_piper_lora_finetune_rtc \
    --policy.dir=/path/to/checkpoints/piper_task_rtc_lora/9999
```

加载时从 checkpoint manifest 恢复模型实际设置，包括图像尺寸、adapters 和 delay 范围，
并使用 checkpoint 内的统计量。`infer(obs)` 支持额外的 RTC 字段：

```python
obs = {
    "observation/top_image": head_rgb,
    "observation/right_wrist_image": wrist_rgb,
    "observation/state": current_state,  # [7]
    "prompt": task,
    "rtc": {
        "prefix": absolute_commands,  # [delay, 7]，机器人原始单位
        "delay": delay,
        "start_index": observation_tick,
        "request_id": request_id,
    },
}
```

`prefix[0]` 对应观测时刻接下来要执行的命令。策略层先提取 RTC 字段，再用本次观测 state
将绝对前缀转换到训练的 delta／归一化坐标。每次去噪迭代均固定前缀，反归一化后还原原始
前缀命令，避免已承诺动作因数值误差改变。不提供 RTC 字段时按 delay 0 运行。

本次提供训练、模型采样及服务端策略接口。现有 Piper 客户端不会因此自动改为异步执行，
动作队列、控制 tick 对齐和端到端延迟测量需要在客户端另行接入。
离线 flow loss 也不等价于实机动作平滑度或任务成功率。

## 验证

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
    .venv/bin/python -m pytest tests/rtc -q
```

测试覆盖真实小模型的前向／反向／JIT、full 与 LoRA 冻结范围、region guidance、padding、
固定前缀、参数分组、统计量、checkpoint 保存／续训及策略加载。
双虚拟 CPU 设备测试用于检查分片与恢复，不代表完整 π0.5 的 GPU 显存或吞吐测试。

算法移植参考本机 `openpi-training-rtc/src/openpi` 实现，保留当前仓库的分辨率、区域偏置、
本地 dataset_root 和 Piper 输入格式；配置统一管理及回归测试在本仓库实现。


本机参数形状核对（224 分辨率）：全量模型总参数／可训练参数均为 3,353,433,872；
默认 RTC LoRA 总参数 3,375,552,272，可训练参数 140,789,792。
这是抽象参数树检查，没有分配完整模型训练权重。

当前象棋数据的实际预检：100 条 episode 按默认设置划分为 90 条训练、10 条验证；
训练集 29,216 帧，horizon 30 对应 837,330 个有效动作 token。
训练及验证 loader 均得到动作 `[2,30,32]` 和 padding `[2,30]`，
轨迹末帧有 29 个 padding token。统计预检只写入临时目录并在结束后清理。
尚未启动该数据上的完整模型 GPU 训练。
