"""Training recipes for pi0 and pi0.5 models."""

import dataclasses

import openpi.models.pi0_config as pi0_config
from openpi.training import weight_loaders
import openpi.training.optimizer as _optimizer
from openpi.training.rtc_config import TrainingRTCConfig


def get_configs():
    """Build pi0 and pi0.5 training recipes after config types are initialized."""
    from openpi.training.config import DataConfig
    from openpi.training.config import FakeDataConfig
    from openpi.training.config import LeRobotPiperDataConfig
    from openpi.training.config import TrainConfig

    configs = [
        #
        ## Lora
        #
        TrainConfig(
            name="pi05_piper_lora_finetune_rtc",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(224, 224),
                action_expert_variant="gemma_300m_lora",
            ),
            training_rtc=TrainingRTCConfig(finetune_mode="lora", max_delay=4),
            data=LeRobotPiperDataConfig(base_config=DataConfig(prompt_from_task=True)),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=2.5e-5, decay_steps=10000, decay_lr=2.5e-6
            ),
            batch_size=16,
            num_train_steps=10000,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        #
        ## 原始 openpi
        #
        TrainConfig(
            name="pi05_piper_full_finetune",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=30, image_resolution=(224, 224)),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True),
                use_delta_actions=True,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            batch_size=32,
            num_train_steps=30_000,
            log_interval=100,
            save_interval=5000,
            keep_period=10_000,
            num_workers=2,
        ),
        TrainConfig(
            name="pi05_piper_full_finetune_image448",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=30, image_resolution=(448, 448)),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True),
                use_delta_actions=True,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            batch_size=32,
            num_train_steps=30_000,
            log_interval=100,
            save_interval=5000,
            keep_period=10_000,
            num_workers=2,
        ),
        TrainConfig(
            name="pi05_piper_full_finetune_image640",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=30, image_resolution=(640, 640)),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True),
                use_delta_actions=True,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            batch_size=32,
            num_train_steps=30_000,
            log_interval=100,
            save_interval=5000,
            keep_period=10_000,
            num_workers=2,
        ),
        TrainConfig(
            name="pi05_piper_lora_finetune",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(224, 224),
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True),
                use_delta_actions=True,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(224, 224),
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
            batch_size=16,
            num_train_steps=10_000,
            log_interval=100,
            save_interval=1000,
            keep_period=1_000,
        ),
        TrainConfig(
            name="pi05_piper_lora_finetune_image448",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(448, 448),
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True),
                use_delta_actions=True,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(448, 448),
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
            batch_size=16,
            num_train_steps=10_000,
            log_interval=100,
            save_interval=1000,
            keep_period=1_000,
        ),
        TrainConfig(
            name="pi05_piper_lora_finetune_image640",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(640, 640),
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True),
                use_delta_actions=True,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(640, 640),
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
            batch_size=16,
            num_train_steps=10_000,
            log_interval=100,
            save_interval=1000,
            keep_period=1_000,
        ),
        #
        ##  原始 openpi + image_448 + at3_s5
        #
        TrainConfig(
            name="pi05_piper_ft_image448_at3_s5",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(448, 448),
                region_guidance=pi0_config.RegionGuidanceConfig(
                    attention_bias_scale=5, query_types=("action", "image", "prompt")
                ),
            ),
            data=LeRobotPiperDataConfig(base_config=DataConfig(prompt_from_task=True)),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-5, decay_steps=10000, decay_lr=1e-6
            ),
            batch_size=32,
            num_train_steps=10000,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        #
        ## Action chunk ablation: image448 + at3_s5, with 10-step chunks.
        # All other settings match pi05_piper_ft_image448_at3_s5.
        #
        TrainConfig(
            name="pi05_piper_ft_image448_aclen10_at3_s5",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=10,
                image_resolution=(448, 448),
                region_guidance=pi0_config.RegionGuidanceConfig(
                    attention_bias_scale=5, query_types=("action", "image", "prompt")
                ),
            ),
            data=LeRobotPiperDataConfig(base_config=DataConfig(prompt_from_task=True)),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-5, decay_steps=10000, decay_lr=1e-6
            ),
            batch_size=32,
            num_train_steps=10000,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        #
        ##  原始 openpi + TT-RTC
        #
        # RTC is enabled only by training_rtc; the launcher needs no separate RTC flag.
        TrainConfig(
            name="pi05_piper_full_finetune_rtc",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=30, image_resolution=(224, 224)),
            training_rtc=TrainingRTCConfig(finetune_mode="full", max_delay=4),
            data=LeRobotPiperDataConfig(base_config=DataConfig(prompt_from_task=True)),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-5, decay_steps=10000, decay_lr=1e-6
            ),
            batch_size=32,
            num_train_steps=10000,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        #
        ## openpi + TT-RTC + attention bias (action) + image_resize 448
        #
        # Full Pi0.5 RTC fine-tuning with training-only bbox attention bias.
        # The launcher reads embedded annotations at runtime; ordinary Piper configs
        # remain free of annotation processing and attention modifications.
        TrainConfig(
            name="pi05_piper_full_finetune_rtc_attention_bias",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(224, 224),
                region_guidance=pi0_config.RegionGuidanceConfig(
                    attention_bias_scale=0.5, query_types=("action", "image", "prompt")
                ),
            ),
            training_rtc=TrainingRTCConfig(finetune_mode="full", max_delay=4),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True, region_annotation_mode="book"),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-5, decay_steps=10000, decay_lr=1e-6
            ),
            batch_size=32,
            num_train_steps=10000,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        # Separate delay-9 presets for 30 Hz deployments whose inference latency
        # at 448x448 exceeds the original four-tick RTC window. The names include
        # the attention-bias scale so experiments with different guidance
        # strengths cannot be confused with one another.
        TrainConfig(
            name="pi05_piper_full_finetune_rtc_attention_bias_448_d9_scale_0_5",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(448, 448),
                region_guidance=pi0_config.RegionGuidanceConfig(
                    attention_bias_scale=0.5, query_types=("action", "image", "prompt")
                ),
            ),
            training_rtc=TrainingRTCConfig(
                finetune_mode="full",
                max_delay=9,
                # At 30 Hz, the measured 223-232 ms round trip maps to delay 8
                # after the client's one-tick safety margin. Delay 9 covers tail
                # latency while delay 8 remains the most frequently sampled case.
                delay_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 12.0, 6.0),
            ),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True, region_annotation_mode="book"),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-5, decay_steps=10000, decay_lr=1e-6
            ),
            batch_size=32,
            num_train_steps=10000,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        TrainConfig(
            name="pi05_piper_full_finetune_rtc_attention_bias_448_d9_scale_2",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(448, 448),
                region_guidance=pi0_config.RegionGuidanceConfig(
                    attention_bias_scale=2, query_types=("action", "image", "prompt")
                ),
            ),
            training_rtc=TrainingRTCConfig(
                finetune_mode="full",
                max_delay=9,
                delay_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 12.0, 6.0),
            ),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True, region_annotation_mode="book"),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-5, decay_steps=10000, decay_lr=1e-6
            ),
            batch_size=32,
            num_train_steps=10000,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        TrainConfig(
            name="pi05_piper_full_finetune_rtc_attention_bias_448_d9_scale_5",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(448, 448),
                region_guidance=pi0_config.RegionGuidanceConfig(
                    attention_bias_scale=5, query_types=("action", "image", "prompt")
                ),
            ),
            training_rtc=TrainingRTCConfig(
                finetune_mode="full",
                max_delay=9,
                delay_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 12.0, 6.0),
            ),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True, region_annotation_mode="book"),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-5, decay_steps=10000, decay_lr=1e-6
            ),
            batch_size=32,
            num_train_steps=10000,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        #
        ## openpi + TT-RTC + at bias (action, image, prompt) + image_resize 448 + 2x2 merge
        #
        # 448x448 SigLIP with Qwen-style 2x2 spatial token merging.
        TrainConfig(
            name="pi05_piper_full_finetune_rtc_d9_r448_merge2_at_bias_s5",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=30,
                image_resolution=(448, 448),
                image_token_merger="spatial_2x2",
                image_token_merge_size=2,
                image_token_camera_keys=("base_0_rgb", "right_wrist_0_rgb"),
                region_guidance=pi0_config.RegionGuidanceConfig(
                    attention_bias_scale=5, query_types=("action", "image", "prompt")
                ),
            ),
            training_rtc=TrainingRTCConfig(
                finetune_mode="full",
                max_delay=9,
                delay_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 12.0, 6.0),
            ),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True, region_annotation_mode="book"),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-5, decay_steps=10000, decay_lr=1e-6
            ),
            batch_size=32,
            num_train_steps=10000,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        #
        ## openpi + TT-RTC + at bias (action, image, prompt) + image_resize 448 + 2x2 merge + KI + Fast token
        #
        TrainConfig(
            name="pi05_piper_full_finetune_rtc_attention_bias_448_d9_scale_0_5_merge2_KI",
            model=pi0_config.Pi0Config(
                knowledge_insulation=True,
                discrete_action_loss_weight=1.0,
                pi05=True,
                action_horizon=30,
                image_resolution=(448, 448),
                image_token_merger="spatial_2x2",
                image_token_merge_size=2,
                image_token_camera_keys=("base_0_rgb", "right_wrist_0_rgb"),
                region_guidance=pi0_config.RegionGuidanceConfig(
                    attention_bias_scale=0.5, query_types=("action", "image", "prompt")
                ),
            ),
            training_rtc=TrainingRTCConfig(
                finetune_mode="full",
                max_delay=9,
                delay_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 12.0, 6.0),
            ),
            data=LeRobotPiperDataConfig(
                base_config=DataConfig(prompt_from_task=True, region_annotation_mode="book"),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-5, decay_steps=10000, decay_lr=1e-6
            ),
            batch_size=32,
            num_train_steps=10000,
            save_interval=1000,
            keep_period=5000,
            ema_decay=None,
        ),
        #
        # Debugging configs.
        #
        TrainConfig(
            name="debug",
            data=FakeDataConfig(),
            batch_size=2,
            model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
            save_interval=100,
            overwrite=True,
            exp_name="debug",
            num_train_steps=10,
            wandb_enabled=False,
        ),
        TrainConfig(
            name="debug_restore",
            data=FakeDataConfig(),
            batch_size=2,
            model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
            weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
            overwrite=True,
            exp_name="debug",
            num_train_steps=10,
            wandb_enabled=False,
        ),
        TrainConfig(
            name="debug_pi05",
            model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
            data=FakeDataConfig(),
            batch_size=2,
            num_train_steps=10,
            overwrite=True,
            exp_name="debug_pi05",
            wandb_enabled=False,
        ),
    ]
    configs += [
        dataclasses.replace(
            config,
            name=f"{config.name}_pad_only",
            model=dataclasses.replace(config.model, image_resize_mode="pad_only"),
        )
        for config in configs
        if config.name in ("pi05_piper_full_finetune_image640", "pi05_piper_lora_finetune_image640")
    ]
    return configs + [
        dataclasses.replace(
            config,
            name=f"{config.name}_pad_to_square",
            model=dataclasses.replace(config.model, pad_to_square=True),
        )
        for config in configs
        if config.name
        in (
            "pi05_piper_full_finetune",
            "pi05_piper_lora_finetune",
            "pi05_piper_full_finetune_rtc",
            "pi05_piper_lora_finetune_rtc",
        )
    ]
