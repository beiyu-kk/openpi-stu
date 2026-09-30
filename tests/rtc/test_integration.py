import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.policies import policy_config
from openpi.policies.rtc_policy import TrainingRTCPolicy
from openpi.shared.normalize import NormStats
from openpi.training import config
from openpi.training import rtc_manifest
from openpi.training import weight_loaders
from scripts import train


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_training_resume_validation_and_checkpoint_policy(tiny_model, tmp_path, monkeypatch, mode):
    tiny, _, _ = tiny_model
    base = config.get_config(f"pi05_piper_{mode}_finetune_rtc")
    cfg = config.resolve_training_config(
        dataclasses.replace(
            base,
            model=dataclasses.replace(
                tiny,
                action_dim=7,
                action_expert_variant="gemma_300m" if mode == "full" else "gemma_300m_lora",
                rtc_finetune_mode=mode,
            ),
            training_rtc=dataclasses.replace(base.training_rtc, validation_interval=1, validation_batches=1),
            batch_size=2,
            num_train_steps=3,
            save_interval=1,
            log_interval=1,
            checkpoint_dir_override=str(tmp_path / "run"),
            exp_name="test",
            wandb_enabled=False,
            weight_loader=weight_loaders.NoOpWeightLoader(),
            data=dataclasses.replace(base.data, repo_id="fake", dataset_root=None),
        )
    )
    stats = {
        name: NormStats(mean=np.zeros(7), std=np.ones(7), q01=-np.ones(7), q99=np.ones(7))
        for name in ("state", "actions")
    }
    obs = dataclasses.replace(cfg.model.fake_obs(batch_size=2), action_is_pad=jnp.zeros((2, 6), bool))
    batch = obs, cfg.model.fake_act(batch_size=2)

    class Loader:
        def __iter__(self):
            yield batch
            yield batch
            yield batch
            yield batch

        def data_config(self):
            return config.DataConfig(repo_id="fake", asset_id="task", norm_stats=stats, use_quantile_norm=True)

    monkeypatch.setattr(train._data_loader, "create_data_loader", lambda *a, **k: Loader())  # noqa: SLF001
    monkeypatch.setattr(train, "init_wandb", lambda *a, **k: None)
    monkeypatch.setattr(train.wandb, "log", lambda *a, **k: None)
    save = train._checkpoints.save_state  # noqa: SLF001

    def interrupt(manager, state, loader, step, **kwargs):
        save(manager, state, loader, step, **kwargs)
        if step == 1:
            manager.wait_until_finished()
            manager.close()
            raise InterruptedError("restart")

    monkeypatch.setattr(train._checkpoints, "save_state", interrupt)  # noqa: SLF001
    # Avoid an upstream CPU executable-cache issue with donated arrays after restore.
    with jax.disable_jit(disable=False), jax.enable_checks(new_val=True):
        cache_enabled = jax.config.jax_enable_compilation_cache
        jax.config.update("jax_enable_compilation_cache", False)  # noqa: FBT003
        try:
            with pytest.raises(InterruptedError, match="restart"):
                train.main(cfg)
            monkeypatch.setattr(train._checkpoints, "save_state", save)  # noqa: SLF001
            train.main(dataclasses.replace(cfg, resume=True))
        finally:
            jax.config.update("jax_enable_compilation_cache", cache_enabled)
    checkpoint = tmp_path / "run/2"
    manifest = rtc_manifest.read(checkpoint)
    assert manifest["training_rtc"]["finetune_mode"] == mode
    metrics = [json.loads(line) for line in (tmp_path / "run/rtc_validation.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == [1, 2, 3]
    assert metrics[-1]["rtc_val/d4/active_tokens"] == 4
    with pytest.raises(ValueError, match="resume settings"):
        rtc_manifest.prepare_resume(
            dataclasses.replace(cfg, resume=True, training_rtc=dataclasses.replace(cfg.training_rtc, max_delay=3))
        )
    with pytest.raises(ValueError, match="used training RTC"):
        rtc_manifest.prepare_resume(dataclasses.replace(cfg, resume=True, training_rtc=None))

    class Tokenizer:
        def tokenize(self, *a, **k):
            return np.ones(4, np.int32), np.ones(4, bool)

    monkeypatch.setattr(config._tokenizer, "PaligemmaTokenizer", lambda *_: Tokenizer())  # noqa: SLF001
    # Even an ordinary registry config must load the checkpoint's actual RTC semantics.
    policy = policy_config.create_trained_policy(config.get_config(f"pi05_piper_{mode}_finetune"), checkpoint)
    assert isinstance(policy, TrainingRTCPolicy)
    prefix = np.full((2, 7), 0.3, np.float32)
    result = policy.infer(
        {
            "observation/top_image": np.zeros((28, 28, 3), np.uint8),
            "observation/right_wrist_image": np.zeros((28, 28, 3), np.uint8),
            "observation/state": np.zeros(7, np.float32),
            "prompt": "task",
            "rtc": {"prefix": prefix, "delay": 2, "start_index": 10, "request_id": 1},
        }
    )
    np.testing.assert_array_equal(result["actions"][:2], prefix)
    assert np.isfinite(result["actions"]).all()
