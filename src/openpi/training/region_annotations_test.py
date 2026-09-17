import copy
import dataclasses
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi import transforms
from openpi.models import model
from openpi.policies import piper_policy
from openpi.training import config
from openpi.training import data_loader
from openpi.training import region_annotations as ra


@pytest.fixture
def annotated_data(tmp_path):
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
    }
    rows = [
        {
            "episode_index": 3,
            "frame_index": 7,
            "index": 50,
            "timestamp": 7 / 30,
            "camera": camera,
            "width": 64,
            "height": 48,
            "use_for_region_loss": camera.endswith("top_head"),
            "label_polygon_xy": [[10, 12], [19, 12], [19, 21], [10, 21]],
            "label_area_px": 100,
        }
        for camera in ra.CAMERAS.values()
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "all_frames.parquet")
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
    return sample, cfg, tmp_path


def test_annotations_do_not_change_original_data_flow(annotated_data):
    sample, cfg, root = annotated_data
    original = data_loader.transform_dataset([copy.deepcopy(sample)], cfg)[0]
    guided_dataset = data_loader.transform_dataset(
        [copy.deepcopy(sample)], dataclasses.replace(cfg, region_annotations_dir=str(root))
    )
    assert len(guided_dataset) == 1
    annotated = guided_dataset[0]
    for key in ("state", "actions", "prompt"):
        np.testing.assert_array_equal(annotated[key], original[key])
    for key in original["image"]:
        np.testing.assert_array_equal(annotated["image"][key], original["image"][key])
        assert annotated["region_masks"][key].shape == (28, 28)
    assert "region_masks" not in original
    assert annotated["region_valid"]["base_0_rgb"]
    assert not annotated["region_valid"]["right_wrist_0_rgb"]
    assert not annotated["region_valid"]["left_wrist_0_rgb"]
    assert annotated["region_masks"]["base_0_rgb"].sum() > 0
    np.testing.assert_array_equal(annotated["region_masks"]["right_wrist_0_rgb"], 0)
    np.testing.assert_array_equal(annotated["region_masks"]["base_0_rgb"][:3], 0)


def test_guided_loader_with_spawn_workers(annotated_data):
    sample, cfg, root = annotated_data
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
        dataclasses.replace(cfg, region_annotations_dir=str(root)),
    )
    loader = data_loader.TorchDataLoader(dataset, local_batch_size=1, num_batches=2, num_workers=2)
    batches = list(loader)
    assert len(batches) == 2
    for batch in batches:
        assert batch["region_masks"]["base_0_rgb"].shape == (1, 28, 28)
        assert np.asarray(batch["region_valid"]["base_0_rgb"]).all()
        assert not np.asarray(batch["region_valid"]["right_wrist_0_rgb"]).any()


@pytest.mark.parametrize("field", ["index", "timestamp", "frame_index"])
def test_mismatched_annotations_fail_instead_of_silently_guiding(annotated_data, field):
    sample, cfg, root = annotated_data
    sample[field] += 1
    annotated = data_loader.transform_dataset([sample], dataclasses.replace(cfg, region_annotations_dir=str(root)))
    with pytest.raises(ValueError, match="annotation|Annotation"):
        annotated[0]


def test_source_validation_is_read_only_and_rejects_another_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    annotations = tmp_path / "annotations"
    dataset.mkdir()
    annotations.mkdir()
    source = dataset / "frame.bin"
    source.write_bytes(b"original")
    (annotations / "all_frames.parquet").touch()
    (annotations / "source_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "frame.bin",
                        "size": 8,
                        "sha256": hashlib.sha256(b"original").hexdigest(),
                    }
                ]
            }
        )
    )
    original_mtime = source.stat().st_mtime_ns
    assert ra.validate_annotation_source(annotations, dataset) == 1
    assert source.stat().st_mtime_ns == original_mtime
    source.write_bytes(b"another!")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ra.validate_annotation_source(annotations, dataset)
