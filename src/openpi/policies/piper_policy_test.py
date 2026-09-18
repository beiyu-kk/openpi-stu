import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import piper_policy


@pytest.mark.parametrize("base_image_key", ["observation/top_image", "observation/image"])
def test_piper_inputs_maps_cameras_and_keeps_continuous_actions(base_image_key):
    base_image = np.zeros((3, 8, 10), dtype=np.float32)
    wrist_image = np.ones((3, 8, 10), dtype=np.float32)
    actions = np.linspace(0.0, 1.0, 14, dtype=np.float32).reshape(2, 7)
    transform = piper_policy.PiperInputs(model_type=_model.ModelType.PI05)

    output = transform(
        {
            base_image_key: base_image,
            "observation/right_wrist_image": wrist_image,
            "observation/state": np.arange(7, dtype=np.float32),
            "actions": actions,
            "prompt": "pick up the book",
        }
    )

    assert output["image"]["base_0_rgb"].shape == (8, 10, 3)
    assert output["image"]["right_wrist_0_rgb"].shape == (8, 10, 3)
    np.testing.assert_array_equal(output["image"]["base_0_rgb"], np.zeros((8, 10, 3), dtype=np.uint8))
    np.testing.assert_array_equal(output["image"]["right_wrist_0_rgb"], np.full((8, 10, 3), 255, dtype=np.uint8))
    assert not output["image_mask"]["left_wrist_0_rgb"]
    np.testing.assert_array_equal(output["state"], np.arange(7, dtype=np.float32))
    assert output["prompt"] == "pick up the book"
    np.testing.assert_array_equal(output["actions"], actions)
    assert output["actions"][..., -1].tolist() == actions[..., -1].tolist()


def test_piper_inputs_prefers_top_image_when_both_keys_are_present():
    top_image = np.full((8, 10, 3), 42, dtype=np.uint8)
    output = piper_policy.PiperInputs(model_type=_model.ModelType.PI05)(
        {
            "observation/top_image": top_image,
            "observation/image": np.zeros_like(top_image),
            "observation/right_wrist_image": np.ones_like(top_image),
            "observation/state": np.zeros(7, dtype=np.float32),
        }
    )

    np.testing.assert_array_equal(output["image"]["base_0_rgb"], top_image)


def test_piper_inputs_reports_missing_base_image_keys():
    with pytest.raises(KeyError, match="observation/top_image.*observation/image"):
        piper_policy.PiperInputs(model_type=_model.ModelType.PI05)(
            {
                "observation/right_wrist_image": np.zeros((8, 10, 3), dtype=np.uint8),
                "observation/state": np.zeros(7, dtype=np.float32),
            }
        )


def test_piper_outputs_returns_seven_continuous_dimensions():
    actions = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
    output = piper_policy.PiperOutputs()({"actions": actions})
    np.testing.assert_array_equal(output["actions"], actions[..., :7])
