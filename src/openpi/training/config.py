"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.piper_policy as piper_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
from openpi.training.rtc_config import TrainingRTCConfig
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Optional root of a local LeRobot dataset. If unset, LeRobot resolves repo_id from its cache or the Hub.
    dataset_root: str | None = None
    # Opt-in training sidecars; the standard data flow never reads this directory.
    region_annotations_dir: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None
    # Derived from TrainConfig; users configure RTC only on TrainConfig.training_rtc.
    training_rtc: TrainingRTCConfig | None = None
    rtc_split: Literal["train", "validation"] = "train"
    episodes: tuple[int, ...] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(*model_config.image_resolution),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(*model_config.image_resolution),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Optional root of a local LeRobot dataset.
    dataset_root: str | None = None
    # Exact directory containing norm_stats.json. If set, this takes precedence over assets.
    norm_stats_dir: str | None = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            dataset_root=self.dataset_root,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = self.norm_stats_dir or str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(self.base_config or DataConfig(), repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPiperDataConfig(DataConfigFactory):
    """Data transforms for the single-arm Piper LeRobot dataset."""

    use_delta_actions: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/top_image": "observation.images.top_head",
                        "observation/right_wrist_image": "observation.images.hand_right",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        **(
                            {"action_is_pad": "action_is_pad"}
                            if getattr(model_config, "simulated_delay", None) is not None
                            else {}
                        ),
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[piper_policy.PiperInputs(model_type=model_config.model_type)],
            outputs=[piper_policy.PiperOutputs()],
        )
        if self.use_delta_actions:
            # All seven Piper dimensions are relative, including the continuous gripper dimension.
            delta_action_mask = _transforms.make_bool_mask(7)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class StuPiperDataConfig(DataConfigFactory):
    """Data transforms for the single-arm Piper LeRobot dataset."""

    use_delta_actions: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/top_image": "observation.images.top_head",
                        "observation/right_wrist_image": "observation.images.hand_right",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                piper_policy.PiperInputs(model_type=model_config.model_type),
                piper_policy.StuImagePreprocess(),
            ],
            outputs=[piper_policy.PiperOutputs()],
        )
        if self.use_delta_actions:
            # All seven Piper dimensions are relative, including the continuous gripper dimension.
            delta_action_mask = _transforms.make_bool_mask(7)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    # Single source of truth. None keeps the original training behavior.
    training_rtc: TrainingRTCConfig | None = None

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"
    # Exact checkpoint output directory. If set, this takes precedence over checkpoint_base_dir/name/exp_name.
    checkpoint_dir_override: str | None = None

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        if self.checkpoint_dir_override is not None:
            return pathlib.Path(self.checkpoint_dir_override).resolve()
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def resolve_training_config(config: TrainConfig) -> TrainConfig:
    """Derive RTC runtime settings after overrides, without doing I/O or mutating presets."""
    rtc = config.training_rtc
    if rtc is None:
        if getattr(config.model, "simulated_delay", None) is not None:
            raise ValueError("Configure RTC through TrainConfig.training_rtc, not model.simulated_delay")
        return config
    if not isinstance(config.model, pi0_config.Pi0Config) or not config.model.pi05:
        raise ValueError("Training RTC supports JAX pi0.5 only")
    if config.pytorch_weight_path is not None:
        raise ValueError("Training RTC supports JAX checkpoints only")
    if rtc.max_delay >= config.model.action_horizon:
        raise ValueError("RTC max_delay must be smaller than action_horizon")
    if not isinstance(config.data, LeRobotPiperDataConfig | FakeDataConfig):
        raise ValueError("RTC requires the standard Piper data transforms")
    if isinstance(config.data, LeRobotPiperDataConfig) and not config.data.use_delta_actions:
        raise ValueError("Piper RTC requires all seven delta action dimensions")
    if isinstance(config.data, LeRobotPiperDataConfig) and config.model.action_dim < 7:
        raise ValueError("Piper RTC model action_dim must be at least 7")
    model = dataclasses.replace(
        config.model,
        simulated_delay=rtc.max_delay + 1,
        simulated_delay_weights=rtc.delay_weights,
        rtc_loss_reduction=rtc.loss_reduction,
        rtc_time_distribution=rtc.time_distribution,
        rtc_finetune_mode=rtc.finetune_mode,
        paligemma_variant=(
            config.model.paligemma_variant.removesuffix("_lora")
            if rtc.finetune_mode == "full"
            else config.model.paligemma_variant
        ),
        action_expert_variant=(
            config.model.action_expert_variant.removesuffix("_lora")
            if rtc.finetune_mode == "full"
            else "gemma_300m_lora"
        ),
    )
    if not isinstance(config.optimizer, _optimizer.AdamW):
        raise ValueError("RTC requires AdamW")
    adam_settings = {f.name: getattr(config.optimizer, f.name) for f in dataclasses.fields(_optimizer.AdamW)}
    optimizer = (
        _optimizer.AdamW(**adam_settings)
        if rtc.finetune_mode == "full"
        else _optimizer.RTCAdamW(**adam_settings, inherited_lr_scale=rtc.inherited_lr_scale)
    )
    loader = config.weight_loader
    if isinstance(loader, weight_loaders.CheckpointWeightLoader):
        loader = weight_loaders.RTCCheckpointWeightLoader(**dataclasses.asdict(loader))
    data = dataclasses.replace(
        config.data,
        base_config=dataclasses.replace(config.data.base_config or DataConfig(), training_rtc=rtc),
    )
    config = dataclasses.replace(
        config,
        model=model,
        data=data,
        optimizer=optimizer,
        weight_loader=loader,
        freeze_filter=model.get_rtc_freeze_filter(),
        ema_decay=None,
    )
    if data.dataset_root is not None and data.norm_stats_dir is None and rtc.norm_source == "train_split":
        from openpi.training.rtc_data import default_norm_stats_dir

        config = dataclasses.replace(
            config, data=dataclasses.replace(data, norm_stats_dir=str(default_norm_stats_dir(config)))
        )
    return config


def prepare_rtc_base(config: TrainConfig) -> TrainConfig:
    """Inspect only checkpoint metadata before initializing a new RTC model."""
    if config.training_rtc is None or not isinstance(config.weight_loader, weight_loaders.RTCCheckpointWeightLoader):
        return config
    keys = config.weight_loader.parameter_keys()
    adapters = [k for k in keys if "lora" in k]
    if config.training_rtc.finetune_mode == "full" and adapters:
        raise ValueError("Full RTC cannot discard LoRA adapters; merge them into base weights before loading")
    if config.training_rtc.finetune_mode == "lora":
        vlm_lora = any(k.startswith("PaliGemma/llm/") and "_1/" not in k for k in adapters)
        model = dataclasses.replace(config.model, paligemma_variant="gemma_2b_lora" if vlm_lora else "gemma_2b")
        config = dataclasses.replace(config, model=model)
    return resolve_training_config(config)


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Fine-tuning Piper configs.
    #
    TrainConfig(
        name="pi05_piper_full_finetune",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
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
    # RTC is enabled only by training_rtc; the launcher needs no separate RTC flag.
    TrainConfig(
        name="pi05_piper_full_finetune_rtc",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
        training_rtc=TrainingRTCConfig(finetune_mode="full"),
        data=LeRobotPiperDataConfig(base_config=DataConfig(prompt_from_task=True)),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params", resize_siglip_posemb=True
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=1e-5, decay_steps=10000, decay_lr=1e-6),
        batch_size=32,
        num_train_steps=10000,
        save_interval=1000,
        keep_period=5000,
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_piper_lora_finetune_rtc",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=30, action_expert_variant="gemma_300m_lora"),
        training_rtc=TrainingRTCConfig(finetune_mode="lora"),
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

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return resolve_training_config(tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()}))


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return resolve_training_config(_CONFIGS_DICT[config_name])
