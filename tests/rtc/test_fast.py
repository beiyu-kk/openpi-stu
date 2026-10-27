# ruff: noqa: SLF001
import dataclasses
import json

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma_fast
from openpi.models import model as model_lib
from openpi.models import pi0_fast
from openpi.policies import policy_config
from openpi.policies.piper_policy import PiperInputs
from openpi.policies.rtc_policy import TrainingRTCPolicy
from openpi.shared.normalize import NormStats
from openpi.training import config
from openpi.training import fast_rtc
from openpi.training import rtc_manifest
from openpi.training import rtc_validation
from openpi.training import weight_loaders
from openpi.training.rtc_config import TrainingRTCConfig
from openpi.training.utils import TrainState
from scripts import train


class TinyVision(nn.Module):
    num_classes: int
    variant: str
    pool_type: str
    scan: bool
    dtype_mm: str
    posemb_grid: tuple[int, int] | None = None

    @nn.compact
    def __call__(self, image, *, train=False):
        del train
        x = nn.avg_pool(image, window_shape=(14, 14), strides=(14, 14))
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return nn.Dense(self.num_classes, dtype=self.dtype_mm)(x), {}


class FakeFASTTokenizer:
    def __init__(self, max_len=16):
        self.max_len = max_len

    def tokenize_rtc(self, prompt, state, prefix, suffix, *, delay, suffix_horizon):
        del prompt, state, prefix, suffix_horizon
        target = [] if suffix is None else [30 + delay] * len(suffix)
        tokens = [1, 10 + delay, *target]
        if len(tokens) > self.max_len:
            raise ValueError("overflow")
        padding = self.max_len - len(tokens)
        active = len(target)
        return (
            np.asarray(tokens + [0] * padding, np.int32),
            np.asarray([True] * len(tokens) + [False] * padding),
            np.asarray([0, 0] + [1] * active + [0] * padding, np.int32),
            np.asarray([False, False] + [True] * active + [False] * padding),
        )

    def extract_rtc_actions(self, tokens, horizon, action_dim):
        del tokens
        return np.full((horizon, action_dim), 0.25, np.float32)


@pytest.fixture
def tiny_fast(monkeypatch):
    original = gemma_fast.get_config

    def small(variant):
        cfg = original(variant)
        cfg.width = 32
        cfg.depth = 2
        cfg.mlp_dim = 64
        cfg.num_heads = 2
        cfg.num_kv_heads = 1
        cfg.head_dim = 16
        cfg.vocab_size = 128
        return cfg

    monkeypatch.setattr(gemma_fast, "get_config", small)
    monkeypatch.setattr(pi0_fast._siglip, "Module", TinyVision)
    cfg = pi0_fast.Pi0FASTConfig(
        action_dim=7,
        action_horizon=4,
        max_token_len=16,
        dtype="float32",
        image_resolution=(28, 28),
        paligemma_variant="gemma_2b_lora",
        simulated_delay=3,
    )
    return cfg, cfg.create(jax.random.key(0)), cfg.fake_obs(batch_size=2)


def fast_rtc_observation(base_observation, *, valid_frames=4):
    tokenizer = FakeFASTTokenizer()
    transform = fast_rtc.TokenizeFASTRTCInputs(tokenizer, action_horizon=4, max_delay=2)
    actions = np.ones((4, 7), np.float32)
    padding = np.arange(4) >= valid_frames
    encoded = transform(
        {"prompt": "task", "state": np.zeros(7, np.float32), "actions": actions, "action_is_pad": padding}
    )
    names = ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask")
    values = {name: jnp.asarray(np.stack([encoded[name]] * 2)) for name in names}
    bank = {name: jnp.asarray(np.stack([encoded["rtc_token_inputs"][name]] * 2)) for name in names}
    return dataclasses.replace(
        base_observation,
        **values,
        rtc_token_inputs=bank,
        action_is_pad=jnp.asarray(np.stack([padding] * 2)),
    )


def test_fast_rtc_candidates_follow_action_delay_and_padding():
    tokenizer = FakeFASTTokenizer()
    transform = fast_rtc.TokenizeFASTRTCInputs(tokenizer, action_horizon=4, max_delay=2)
    result = transform(
        {
            "prompt": "task",
            "state": np.zeros(7, np.float32),
            "actions": np.ones((4, 7), np.float32),
            "action_is_pad": np.array([False, False, True, True]),
        }
    )
    bank = result["rtc_token_inputs"]["tokenized_prompt"]
    np.testing.assert_array_equal(bank[0, :6], [1, 10, 30, 30, 0, 0])
    np.testing.assert_array_equal(bank[1, :5], [1, 11, 31, 0, 0])
    np.testing.assert_array_equal(bank[2, :4], [1, 12, 0, 0])
    assert result["rtc_token_inputs"]["token_loss_mask"][2].sum() == 0
    with pytest.raises(ValueError, match="contiguous episode tail"):
        transform(
            {
                "prompt": "task",
                "state": np.zeros(7, np.float32),
                "actions": np.ones((4, 7), np.float32),
                "action_is_pad": np.array([False, True, False, True]),
            }
        )


def test_fast_rtc_configs_resolve_full_lora_and_resize_modes():
    full = config.resolve_training_config(config.get_config("pi0_fast_piper_full_finetune_rtc"))
    lora = config.resolve_training_config(config.get_config("pi0_fast_piper_lora_finetune_rtc"))
    assert full.model.simulated_delay == lora.model.simulated_delay == 5
    assert full.model.max_token_len == lora.model.max_token_len == 512
    assert full.freeze_filter is nnx.Nothing
    assert lora.model.paligemma_variant == "gemma_2b_lora"
    assert not nnx.filterlib.to_predicate(lora.freeze_filter)(("PaliGemma", "llm", "lora_a"), nnx.Param(0))
    assert nnx.filterlib.to_predicate(lora.freeze_filter)(("PaliGemma", "img", "kernel"), nnx.Param(0))
    padded = config.get_config("pi0_fast_piper_full_finetune_rtc_image640_pad_only")
    assert padded.model.image_resolution == (640, 640)
    assert padded.model.image_resize_mode == "pad_only"
    with pytest.raises(ValueError, match="token cross-entropy"):
        config.resolve_training_config(
            dataclasses.replace(lora, training_rtc=dataclasses.replace(lora.training_rtc, loss_reduction="per_element"))
        )


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_fast_rtc_train_step_loss_validation_and_empty_suffix(tiny_fast, mode):
    model_cfg, _, base_observation = tiny_fast
    cfg = config.resolve_training_config(
        config.TrainConfig(
            name="test_fast_rtc",
            model=model_cfg,
            data=config.FakeDataConfig(),
            training_rtc=TrainingRTCConfig(finetune_mode=mode, max_delay=2),
            num_train_steps=4,
        )
    )
    model = cfg.model.create(jax.random.key(0))
    params = nnx.state(model)
    # FAST initializes the tied embedding/output table to zero for checkpoint loading.
    # A nonzero table is needed for gradients to reach the backbone, as in a real checkpoint.
    embedding = params["PaliGemma"]["llm"]["embedder"]["input_embedding"]
    embedding.value = jax.random.normal(jax.random.key(5), embedding.value.shape) * 0.1
    nnx.update(model, params)
    tx = cfg.optimizer.create(1e-3)
    state = TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(cfg.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )
    step = jax.jit(lambda rng, train_state, batch: train.train_step(cfg, rng, train_state, batch))
    actions = jnp.zeros((2, 4, 7))

    observation = fast_rtc_observation(base_observation)
    updated, metrics = step(jax.random.key(2), state, (observation, actions))
    assert float(metrics["rtc/active_tokens"]) > 0
    assert np.isfinite(float(metrics["rtc/token_ce"]))
    before, after = params.flat_state(), updated.params.flat_state()
    changed = {path for path in after if not np.array_equal(after[path].value, before[path].value)}
    assert changed
    assert changed <= set(params.filter(cfg.trainable_filter).flat_state())
    if mode == "lora":
        assert all("lora" in "/".join(path) for path in changed)
    else:
        assert any("/img/" in "/".join(path) for path in changed)
        assert any("input_embedding" in path for path in changed)

    empty = fast_rtc_observation(base_observation, valid_frames=0)
    skipped, info = step(jax.random.key(1), updated, (empty, actions))
    assert int(info["rtc/active_tokens"]) == 0
    for actual, expected in zip(jax.tree.leaves(skipped.params), jax.tree.leaves(updated.params), strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(jax.tree.leaves(skipped.opt_state), jax.tree.leaves(updated.opt_state), strict=True):
        np.testing.assert_array_equal(actual, expected)

    validate = jax.jit(
        lambda params, batch: rtc_validation.state_batch_sums(
            state.model_def, params, jax.random.key(3), batch, raw_action_dim=7
        )
    )
    sums = validate(updated.params, (observation, actions))
    assert sums["negative_log_likelihood"].shape == (3,)
    assert sums["active_tokens"].shape == (3,)
    summary = rtc_validation.summarize(
        {key: np.asarray(value) for key, value in sums.items()},
        raw_action_dim=7,
        model_action_dim=7,
        reduction="official",
    )
    assert "rtc_val/d0/token_accuracy" in summary


def test_fast_rtc_policy_returns_decoded_continuation_and_exact_prefix():
    class DummyModel(nnx.Module):
        action_horizon = 4
        action_dim = 7
        max_token_len = 16
        simulated_delay = 3
        rtc_action_tokens = True

        def sample_actions(self, rng, observation, **kwargs):
            del rng, observation, kwargs
            return jnp.zeros((1, 8), jnp.int32)

    tokenizer = FakeFASTTokenizer()
    policy = TrainingRTCPolicy(
        DummyModel(),
        raw_action_dim=7,
        transforms=[
            PiperInputs(model_type=model_lib.ModelType.PI0_FAST),
            fast_rtc.TokenizeFASTRTCInputs(tokenizer, action_horizon=4, max_delay=2),
        ],
        output_transforms=[fast_rtc.ExtractFASTRTCActions(tokenizer, action_horizon=4, action_dim=7)],
    )
    prefix = np.full((1, 7), 3.0, np.float32)
    result = policy.infer(
        {
            "observation/top_image": np.zeros((28, 28, 3), np.uint8),
            "observation/right_wrist_image": np.zeros((28, 28, 3), np.uint8),
            "observation/state": np.ones(7, np.float32),
            "prompt": "task",
            "rtc": {"prefix": prefix, "delay": 1, "start_index": 5, "request_id": 2},
        }
    )
    np.testing.assert_array_equal(result["actions"][0], prefix[0])
    np.testing.assert_array_equal(result["actions"][1:], 0.25)
    assert result["rtc"]["delay"] == 1
    assert policy._sample_kwargs["max_decoding_steps"] == 16


def test_fast_rtc_output_transform_checks_decoded_suffix_shape():
    class MalformedTokenizer(FakeFASTTokenizer):
        def extract_rtc_actions(self, tokens, horizon, action_dim):
            del tokens, horizon
            return np.zeros((1, action_dim), np.float32)

    transform = fast_rtc.ExtractFASTRTCActions(MalformedTokenizer(), action_horizon=4, action_dim=7)
    with pytest.raises(ValueError, match="invalid decoded suffix"):
        transform({"rtc_delay": 1, "rtc_prefix": np.zeros((1, 7)), "actions": np.zeros(8, np.int32)})


@pytest.mark.parametrize(("mode", "dynamic"), [("full", False), ("lora", False), ("lora", True)])
def test_fast_rtc_training_resume_validation_and_policy_loading(tiny_fast, tmp_path, monkeypatch, mode, dynamic):
    tiny, _, _ = tiny_fast
    suffix = "_pad_to_square" if dynamic else ""
    base = config.get_config(f"pi0_fast_piper_{mode}_finetune_rtc{suffix}")
    tiny = dataclasses.replace(tiny, pad_to_square=base.model.pad_to_square)
    cfg = config.resolve_training_config(
        dataclasses.replace(
            base,
            model=tiny,
            training_rtc=dataclasses.replace(
                base.training_rtc, max_delay=2, validation_interval=1, validation_batches=1
            ),
            data=dataclasses.replace(base.data, repo_id="fake", dataset_root=None),
            weight_loader=weight_loaders.NoOpWeightLoader(),
            batch_size=2,
            num_train_steps=3,
            save_interval=1,
            log_interval=1,
            checkpoint_dir_override=str(tmp_path / "run"),
            exp_name="test_fast",
            wandb_enabled=False,
        )
    )
    stats = {
        name: NormStats(mean=np.zeros(7), std=np.ones(7), q01=-np.ones(7), q99=np.ones(7))
        for name in ("state", "actions")
    }
    observation = fast_rtc_observation(cfg.model.fake_obs(batch_size=2))
    observations = [observation]
    if dynamic:
        observations = [
            dataclasses.replace(
                observation,
                images={
                    key: jnp.zeros((2, side + i * 7, side + 7, 3), jnp.float32)
                    for i, key in enumerate(observation.images)
                },
            )
            for side in (28, 42)
        ]
    batches = [(obs, cfg.model.fake_act(batch_size=2)) for obs in observations]

    class Loader:
        def __iter__(self):
            yield from (batches * 4)[:4]

        def data_config(self):
            return config.DataConfig(repo_id="fake", asset_id="task", norm_stats=stats, use_quantile_norm=True)

    monkeypatch.setattr(train._data_loader, "create_data_loader", lambda *a, **k: Loader())
    monkeypatch.setattr(train, "init_wandb", lambda *a, **k: None)
    monkeypatch.setattr(train.wandb, "log", lambda *a, **k: None)
    save = train._checkpoints.save_state

    def interrupt(manager, state, loader, step, **kwargs):
        save(manager, state, loader, step, **kwargs)
        if step == 1:
            manager.wait_until_finished()
            manager.close()
            raise InterruptedError("restart")

    monkeypatch.setattr(train._checkpoints, "save_state", interrupt)
    with pytest.raises(InterruptedError, match="restart"):
        train.main(cfg)
    monkeypatch.setattr(train._checkpoints, "save_state", save)
    train.main(dataclasses.replace(cfg, resume=True))
    checkpoint = tmp_path / "run/2"
    manifest = rtc_manifest.read(checkpoint)
    assert manifest["model_type"] == "pi0_fast"
    assert manifest["rtc_objective"] == "fast_prefix_suffix_v1"
    assert manifest["freeze_strategy"] == ("full" if mode == "full" else "fast_vlm_lora")
    metrics = [json.loads(line) for line in (tmp_path / "run/rtc_validation.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == [1, 2, 3]
    assert metrics[-1]["rtc_val/d2/active_tokens"] == 4
    assert metrics[-1]["rtc_val/d2/token_ce"] > 0

    with pytest.raises(ValueError, match="resume settings"):
        rtc_manifest.prepare_resume(
            dataclasses.replace(cfg, resume=True, training_rtc=dataclasses.replace(cfg.training_rtc, max_delay=1))
        )
    with pytest.raises(ValueError, match="model type"):
        rtc_manifest.restore_config(config.get_config("pi05_piper_full_finetune_rtc"), manifest, checkpoint)
    with pytest.raises(ValueError, match="tokenization objective"):
        rtc_manifest.restore_config(cfg, {**manifest, "rtc_objective": "unknown"}, checkpoint)

    monkeypatch.setattr(config._tokenizer, "FASTTokenizer", FakeFASTTokenizer)
    # Loading a baseline recipe must restore the saved FAST RTC model settings.
    policy = policy_config.create_trained_policy(config.get_config(f"pi0_fast_piper_{mode}_finetune"), checkpoint)
    assert isinstance(policy, TrainingRTCPolicy)
    assert policy.metadata["max_delay"] == 2
    assert policy._model.image_resolution == (28, 28)
    assert policy._model.pad_to_square == dynamic
    assert policy._model.image_resize_mode == "resize_with_pad"
    assert policy._model.max_token_len == 16
    prefix = np.full((2, 7), 0.3, np.float32)
    for side in (35, 49) if dynamic else (28,):
        result = policy.infer(
            {
                "observation/top_image": np.zeros((28, side, 3), np.uint8),
                "observation/right_wrist_image": np.zeros((side, 28, 3), np.uint8),
                "observation/state": np.ones(7, np.float32),
                "prompt": "task",
                "rtc": {"prefix": prefix, "delay": 2, "start_index": 10, "request_id": 1},
            }
        )
        np.testing.assert_array_equal(result["actions"][:2], prefix)
        np.testing.assert_allclose(result["actions"][2:], 1.25, atol=2e-6)
