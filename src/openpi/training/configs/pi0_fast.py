"""Training recipes for the autoregressive pi0-FAST model."""

import dataclasses

import openpi.models.pi0_fast as pi0_fast
from openpi.training import weight_loaders
import openpi.training.optimizer as _optimizer
from openpi.training.rtc_config import TrainingRTCConfig


def get_configs():
    """Build pi0-FAST recipes after shared config types are initialized."""
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotPiperDataConfig
    from openpi.training.config import TrainConfig

    _piper_fast_model = pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=30, max_token_len=250)
    _piper_fast_model_448 = dataclasses.replace(_piper_fast_model, image_resolution=(448, 448))
    _piper_fast_model_640 = dataclasses.replace(_piper_fast_model, image_resolution=(640, 640))
    _piper_fast_lora_model = pi0_fast.Pi0FASTConfig(
        action_dim=7,
        action_horizon=30,
        max_token_len=250,
        paligemma_variant="gemma_2b_lora",
    )
    _piper_fast_lora_model_448 = dataclasses.replace(_piper_fast_lora_model, image_resolution=(448, 448))
    _piper_fast_lora_model_640 = dataclasses.replace(_piper_fast_lora_model, image_resolution=(640, 640))

    configs = [
        TrainConfig(
            name="pi0_fast_piper_full_finetune",
            model=_piper_fast_model,
            data=LeRobotPiperDataConfig(base_config=DataConfig(prompt_from_task=True)),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_fast_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=1e-5,
                decay_steps=10000,
                decay_lr=1e-6,
            ),
            batch_size=16,
            num_train_steps=10000,
            log_interval=100,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0_fast_piper_full_finetune_image448",
            model=_piper_fast_model_448,
            data=LeRobotPiperDataConfig(base_config=DataConfig(prompt_from_task=True)),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_fast_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=1e-5,
                decay_steps=10000,
                decay_lr=1e-6,
            ),
            batch_size=16,
            num_train_steps=10000,
            log_interval=100,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0_fast_piper_full_finetune_image640",
            model=_piper_fast_model_640,
            data=LeRobotPiperDataConfig(base_config=DataConfig(prompt_from_task=True)),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_fast_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=1e-5,
                decay_steps=10000,
                decay_lr=1e-6,
            ),
            batch_size=16,
            num_train_steps=10000,
            log_interval=100,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0_fast_piper_lora_finetune",
            model=_piper_fast_lora_model,
            data=LeRobotPiperDataConfig(base_config=DataConfig(prompt_from_task=True)),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_fast_base/params", resize_siglip_posemb=True
            ),
            freeze_filter=_piper_fast_lora_model.get_freeze_filter(),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=2.5e-5,
                decay_steps=10000,
                decay_lr=2.5e-6,
            ),
            batch_size=16,
            num_train_steps=10000,
            log_interval=100,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0_fast_piper_lora_finetune_image448",
            model=_piper_fast_lora_model_448,
            data=LeRobotPiperDataConfig(base_config=DataConfig(prompt_from_task=True)),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_fast_base/params", resize_siglip_posemb=True
            ),
            freeze_filter=_piper_fast_lora_model_448.get_freeze_filter(),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=2.5e-5,
                decay_steps=10000,
                decay_lr=2.5e-6,
            ),
            batch_size=16,
            num_train_steps=10000,
            log_interval=100,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        TrainConfig(
            name="pi0_fast_piper_lora_finetune_image640",
            model=_piper_fast_lora_model_640,
            data=LeRobotPiperDataConfig(base_config=DataConfig(prompt_from_task=True)),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_fast_base/params", resize_siglip_posemb=True
            ),
            freeze_filter=_piper_fast_lora_model_640.get_freeze_filter(),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500,
                peak_lr=2.5e-5,
                decay_steps=10000,
                decay_lr=2.5e-6,
            ),
            batch_size=16,
            num_train_steps=10000,
            log_interval=100,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
    ]
    configs += [
        dataclasses.replace(
            config,
            name=f"{config.name}_pad_only",
            model=dataclasses.replace(config.model, image_resize_mode="pad_only"),
        )
        for config in configs
        if config.name in ("pi0_fast_piper_full_finetune_image640", "pi0_fast_piper_lora_finetune_image640")
    ]
    configs += [
        dataclasses.replace(
            config,
            name=f"{config.name}_pad_to_square",
            model=dataclasses.replace(config.model, pad_to_square=True),
        )
        for config in configs
        if config.name in ("pi0_fast_piper_full_finetune", "pi0_fast_piper_lora_finetune")
    ]
    return configs + [
        dataclasses.replace(
            config,
            name=config.name.replace("_finetune", "_finetune_rtc"),
            model=dataclasses.replace(config.model, max_token_len=512),
            training_rtc=TrainingRTCConfig(finetune_mode="lora" if "lora" in config.name else "full", max_delay=4),
        )
        for config in configs
    ]
