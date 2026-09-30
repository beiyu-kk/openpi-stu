import numpy as np
import pytest

from openpi.training.rtc_norm_stats import delta_windows
from openpi.transforms import DeltaActions


def test_delta_windows_match_dataset_queries_and_exclude_tail_padding():
    state = np.repeat(np.array([0, 10, 20], dtype=np.float32)[:, None], 7, axis=1)
    action = np.repeat(np.array([1, 12, 25], dtype=np.float32)[:, None], 7, axis=1)
    values = delta_windows(state, action, 3)
    expected = []
    for start in range(3):
        indices = np.minimum(np.arange(start, start + 3), 2)
        window = DeltaActions((True,) * 7)({"state": state[start], "actions": action[indices].copy()})["actions"]
        expected.extend(window[: 3 - start])
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(values[:, 0], [1, 12, 25, 2, 15, 5])
    # Gripper follows the same all-delta convention.
    np.testing.assert_array_equal(values[:, 6], values[:, 0])
    assert len(values) == 6


def test_nonfinite_data_is_rejected():
    state = np.ones((3, 7), dtype=np.float32)
    actions = state.copy()
    actions[1, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        delta_windows(state, actions, 30)
