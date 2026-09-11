import numpy as np
import pytest

from examples.piper import main as piper_client
from openpi import transforms
from openpi.policies import piper_policy
from openpi.training import config as _config


@pytest.mark.parametrize("size", [224, 336, 448])
def test_original_camera_images_are_resized_by_train_config(monkeypatch, size):
    head = np.full((423, 628, 3), 80, dtype=np.uint8)
    wrist = np.full((480, 640, 3), 160, dtype=np.uint8)
    request = piper_client.build_policy_observation(head, wrist, np.zeros(7), "pick the book")
    np.testing.assert_array_equal(request["observation/top_image"], head)
    np.testing.assert_array_equal(request["observation/right_wrist_image"], wrist)

    name = "pi05_piper_full_finetune" if size == 224 else f"pi05_piper_full_finetune_{size}"
    config = _config.get_config(name)
    monkeypatch.setattr(_config._tokenizer, "PaligemmaTokenizer", lambda *_: None)  # noqa: SLF001
    group = _config.ModelTransformFactory()(config.model)
    resize = next(transform for transform in group.inputs if isinstance(transform, transforms.ResizeImages))
    observation = piper_policy.PiperInputs(model_type=config.model.model_type)(request)
    result = resize(observation)
    assert all(image.shape == (size, size, 3) for image in result["image"].values())
