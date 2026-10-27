import dataclasses

import jax
import numpy as np
import pytest

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_dynamic_batches_preserve_pixels_and_region_alignment(dtype):
    items = []
    for size in (20, 31, 42, 17):
        pixels = np.random.default_rng(size).integers(0, 255, (size - 3, size, 3), dtype=np.uint8)
        if dtype == np.float32:
            pixels = pixels.astype(np.float32) / 127.5 - 1
        item = transforms.ResizeImages(224, 224, mode="pad_to_square")(
            {"image": {"head": pixels, "wrist": pixels[:8, :12]}, "state": np.zeros(7, np.float32)}
        )
        item["region_masks"] = {key: np.ones(image.shape[:2], np.float32) for key, image in item["image"].items()}
        items.append(item)
    loader = _data_loader.TorchDataLoader(items, local_batch_size=2, num_batches=2, pad_images_to_square=True)
    batches = list(loader)
    assert [batch["image"]["head"].shape for batch in batches] == [(2, 31, 31, 3), (2, 42, 42, 3)]
    assert all(batch["image"]["wrist"].shape == (2, 12, 12, 3) for batch in batches)
    for i, batch in enumerate(batches):
        for j, item in enumerate(items[2 * i : 2 * i + 2]):
            for key, image in item["image"].items():
                actual = np.asarray(batch["image"][key][j])
                side = actual.shape[0]
                offset = (side - image.shape[0]) // 2
                np.testing.assert_array_equal(actual[offset : offset + len(image), offset : offset + len(image)], image)
                expected_region = np.zeros((side, side), np.float32)
                expected_region[offset : offset + len(image), offset : offset + len(image)] = 1
                np.testing.assert_array_equal(batch["region_masks"][key][j], expected_region)
    assert items[0]["image"]["head"].shape == (20, 20, 3)


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
