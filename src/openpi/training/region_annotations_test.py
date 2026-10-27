import copy
import dataclasses

import jax
import numpy as np
import pytest

from openpi import transforms
from openpi.models import model
from openpi.policies import piper_policy
from openpi.training import config
from openpi.training import data_loader
from openpi.training import region_annotations as ra


@pytest.fixture
def annotated_data():
    sample = {
        "episode_index": np.array(3),
        "frame_index": np.array(7),
        "index": np.array(50),
        "timestamp": np.array(7 / 30),
        "observation.state": np.arange(7, dtype=np.float32),
        "action": np.arange(14, dtype=np.float32).reshape(2, 7),
        "prompt": "Pick the labeled book",
        "observation.images.top_head": np.full((3, 48, 64), 0.4, np.float32),
        "observation.images.hand_right": np.full((3, 48, 64), 0.7, np.float32),
        ra.BOOK_BBOX_KEY: np.array([10, 12, 19, 21], dtype=np.float32),
        ra.TITLE_BBOX_KEY: np.array([11, 13, 18, 18], dtype=np.float32),
    }
    cfg = config.DataConfig(
        repo_id="fake",
        repack_transforms=transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/top_image": "observation.images.top_head",
                        "observation/right_wrist_image": "observation.images.hand_right",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
        data_transforms=transforms.Group(
            inputs=[
                piper_policy.PiperInputs(model.ModelType.PI05),
                transforms.DeltaActions(transforms.make_bool_mask(7)),
            ]
        ),
        model_transforms=transforms.Group(inputs=[transforms.ResizeImages(28, 28)]),
    )
    return sample, cfg


def test_annotations_do_not_change_original_data_flow(annotated_data):
    sample, cfg = annotated_data
    original = data_loader.transform_dataset([copy.deepcopy(sample)], cfg)[0]
    guided_dataset = data_loader.transform_dataset(
        [copy.deepcopy(sample)], dataclasses.replace(cfg, region_annotations_embedded=True)
    )
    assert len(guided_dataset) == 1
    annotated = guided_dataset[0]
    for key in ("state", "actions", "prompt"):
        np.testing.assert_array_equal(annotated[key], original[key])
    for key in original["image"]:
        np.testing.assert_array_equal(annotated["image"][key], original["image"][key])
        assert annotated["region_masks"][key].shape == (28, 28)
    assert "region_masks" not in original
    assert not annotated["region_valid"]["base_0_rgb"]
    assert annotated["region_valid"]["right_wrist_0_rgb"]
    assert not annotated["region_valid"]["left_wrist_0_rgb"]
    assert annotated["region_masks"]["right_wrist_0_rgb"].sum() > 0
    np.testing.assert_array_equal(annotated["region_masks"]["base_0_rgb"], 0)


def test_embedded_annotations_create_right_wrist_mask(annotated_data):
    sample, cfg = annotated_data
    sample[ra.BOOK_BBOX_KEY] = np.array([10, 12, 19, 21], dtype=np.float32)
    sample[ra.TITLE_BBOX_KEY] = np.full(4, -1, dtype=np.float32)
    guided = data_loader.transform_dataset(
        [sample], dataclasses.replace(cfg, region_annotations_embedded=True, region_annotation_mode="book")
    )[0]

    assert guided["region_valid"]["right_wrist_0_rgb"]
    assert not guided["region_valid"]["base_0_rgb"]
    assert guided["region_masks"]["right_wrist_0_rgb"].sum() > 0


def test_pad_only_annotations_follow_image_offsets(annotated_data):
    sample, cfg = annotated_data
    cfg = dataclasses.replace(
        cfg,
        region_annotations_embedded=True,
        model_transforms=transforms.Group(inputs=[transforms.ResizeImages(84, 84, mode="pad_only")]),
    )
    annotated = data_loader.transform_dataset([copy.deepcopy(sample)], cfg)[0]
    expected = np.zeros((84, 84), dtype=np.float32)
    expected[18:66, 10:74] = ra.boxes_to_mask([sample[ra.BOOK_BBOX_KEY]], 48, 64)
    np.testing.assert_array_equal(annotated["region_masks"]["right_wrist_0_rgb"], expected)
    assert annotated["image"]["right_wrist_0_rgb"].shape == (84, 84, 3)


def test_dynamic_square_annotations_follow_native_image_offsets(annotated_data):
    sample, cfg = annotated_data
    cfg = dataclasses.replace(
        cfg,
        region_annotations_embedded=True,
        model_transforms=transforms.Group(inputs=[transforms.ResizeImages(28, 28, mode="pad_to_square")]),
    )
    annotated = data_loader.transform_dataset([copy.deepcopy(sample)], cfg)[0]
    expected = np.zeros((64, 64), dtype=np.float32)
    expected[8:56] = ra.boxes_to_mask([sample[ra.BOOK_BBOX_KEY]], 48, 64)
    np.testing.assert_array_equal(annotated["region_masks"]["right_wrist_0_rgb"], expected)
    assert annotated["image"]["right_wrist_0_rgb"].shape == (64, 64, 3)


def test_embedded_invalid_sentinel_disables_region(annotated_data):
    sample, cfg = annotated_data
    sample[ra.BOOK_BBOX_KEY] = np.full(4, -1, dtype=np.float32)
    sample[ra.TITLE_BBOX_KEY] = np.full(4, -1, dtype=np.float32)
    guided = data_loader.transform_dataset(
        [sample], dataclasses.replace(cfg, region_annotations_embedded=True, region_annotation_mode="both")
    )[0]

    assert not any(guided["region_valid"].values())
    assert all(not mask.any() for mask in guided["region_masks"].values())


def test_embedded_both_mode_unions_book_and_title(annotated_data):
    sample, cfg = annotated_data
    sample[ra.BOOK_BBOX_KEY] = np.array([2, 2, 8, 8], dtype=np.float32)
    sample[ra.TITLE_BBOX_KEY] = np.array([40, 30, 46, 36], dtype=np.float32)
    book = data_loader.transform_dataset(
        [copy.deepcopy(sample)],
        dataclasses.replace(cfg, region_annotations_embedded=True, region_annotation_mode="book"),
    )[0]
    both = data_loader.transform_dataset(
        [copy.deepcopy(sample)],
        dataclasses.replace(cfg, region_annotations_embedded=True, region_annotation_mode="both"),
    )[0]

    assert both["region_valid"]["right_wrist_0_rgb"]
    assert both["region_masks"]["right_wrist_0_rgb"].sum() > book["region_masks"]["right_wrist_0_rgb"].sum()


def test_guided_loader_with_spawn_workers(annotated_data):
    sample, cfg = annotated_data
    # The production tokenizer consumes prompt strings before batches enter JAX.
    cfg = dataclasses.replace(
        cfg,
        model_transforms=cfg.model_transforms.push(
            inputs=[
                transforms.RepackTransform(
                    {
                        "state": "state",
                        "actions": "actions",
                        "image": {key: f"image/{key}" for key in model.IMAGE_KEYS},
                        "image_mask": {key: f"image_mask/{key}" for key in model.IMAGE_KEYS},
                    }
                )
            ]
        ),
    )
    dataset = data_loader.transform_dataset(
        [copy.deepcopy(sample) for _ in range(4)],
        dataclasses.replace(cfg, region_annotations_embedded=True),
    )
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_batches=2,
        num_workers=2,
        sharding=jax.sharding.SingleDeviceSharding(jax.local_devices()[0]),
    )
    batches = list(loader)
    assert len(batches) == 2
    for batch in batches:
        assert batch["region_masks"]["right_wrist_0_rgb"].shape == (1, 28, 28)
        assert not np.asarray(batch["region_valid"]["base_0_rgb"]).any()
        assert np.asarray(batch["region_valid"]["right_wrist_0_rgb"]).all()


def test_missing_embedded_fields_fail(annotated_data):
    sample, cfg = annotated_data
    del sample[ra.TITLE_BBOX_KEY]
    annotated = data_loader.transform_dataset([sample], dataclasses.replace(cfg, region_annotations_embedded=True))
    with pytest.raises(ValueError, match="Missing embedded annotation fields"):
        annotated[0]
