# ruff: noqa: SLF001
import dataclasses

import flax.linen as nn
import jax
import pytest

from openpi.models import gemma
from openpi.models import pi0
from openpi.models import pi0_config


@pytest.fixture(autouse=True)
def fresh_cpu_cache():
    # JAX 0.5.3 CPU cached executables can corrupt donated updates after Orbax
    # restore. Include fixture/model initialization in the uncached scope.
    enabled = jax.config.jax_enable_compilation_cache
    if jax.default_backend() == "cpu":
        jax.config.update("jax_enable_compilation_cache", False)  # noqa: FBT003
    try:
        yield
    finally:
        jax.config.update("jax_enable_compilation_cache", enabled)


class TinyVision(nn.Module):
    num_classes: int
    variant: str
    pool_type: str
    scan: bool
    dtype_mm: str

    @nn.compact
    def __call__(self, image, *, train=False):
        x = nn.avg_pool(image, window_shape=(14, 14), strides=(14, 14))
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return nn.Dense(self.num_classes, dtype=self.dtype_mm)(x), {}


@pytest.fixture
def tiny_model(monkeypatch):
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    original = gemma.get_config

    def small(variant):
        c = original(variant)
        return dataclasses.replace(c, width=32, depth=2, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16)

    monkeypatch.setattr(gemma, "get_config", small)
    monkeypatch.setattr(pi0._siglip, "Module", TinyVision)
    cfg = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m_lora",
        dtype="float32",
        image_resolution=(28, 28),
        action_dim=4,
        action_horizon=6,
        max_token_len=4,
        simulated_delay=5,
    )
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(batch_size=2)
    return cfg, model, obs
