# Knowledge Insulation 配置

项目现在支持论文《Knowledge Insulating Vision-Language-Action Models》中的双 action 训练结构：

```python
from openpi.models.pi0_config import Pi0Config

model = Pi0Config(
    pi05=True,
    knowledge_insulation=True,
    discrete_action_loss_weight=1.0,
)
```

`knowledge_insulation=False` 是默认值，保持原来的连续 flow matching 训练。设置为 `True` 后，训练数据自动使用 FAST tokenizer，同时计算：

```text
loss = flow_matching_loss + discrete_action_loss_weight * FAST_next_token_CE
```

FAST 的 prefix（图像、语言、state）使用双向 attention，FAST action token 使用因果 attention 并预测下一个 token。连续 action expert 读取图像、语言、state 和连续 action token，但连续分支会屏蔽 FAST action token；VLM 分支也不会读取连续 action expert。连续分支读取 VLM 的 K/V 时只对 VLM K/V 使用 `stop_gradient`，因此 flow loss 仍会更新 action expert，FAST CE 仍会更新 VLM backbone 和图像编码器。

`discrete_action_loss_weight` 必须是有限的非负数；启用 Knowledge Insulation 时必须大于零。论文在两个分支独立后使用 `alpha=1`，所以默认值为 `1.0`。

如果同时启用区域 attention bias，配置中的 `query_types` 会同时作用于连续分支和 FAST 离散分支。例如：

```python
from openpi.models.region_guidance import RegionGuidanceConfig

model = Pi0Config(
    pi05=True,
    knowledge_insulation=True,
    discrete_action_loss_weight=1.0,
    region_guidance=RegionGuidanceConfig(
        attention_bias_scale=0.5,
        query_types=("action", "image", "prompt"),
    ),
)
```

推理仍然调用连续 action expert 的 flow sampling，不需要生成 FAST action token。FAST token 只作为训练时的 representation-learning objective。

该选项当前对应 JAX 的 `scripts/train.py` 训练路径；PyTorch 训练入口会显式拒绝它，避免误以为只训练了 flow loss。
