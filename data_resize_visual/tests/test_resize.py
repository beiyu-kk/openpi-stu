import numpy as np

from data_resize_visual.resize import resize_with_pad


def test_uint8_landscape_resize_and_pad() -> None:
    image = np.full((1, 180, 320, 3), 255, dtype=np.uint8)

    actual = np.asarray(resize_with_pad(image, 224, 224))

    assert actual.shape == (1, 224, 224, 3)
    assert actual.dtype == np.uint8
    assert np.all(actual[:, :49] == 0)
    assert np.all(actual[:, 49:175] == 255)
    assert np.all(actual[:, 175:] == 0)


def test_odd_padding_remainder_goes_to_bottom_and_right() -> None:
    image = np.full((4, 5, 3), 255, dtype=np.uint8)

    actual = np.asarray(resize_with_pad(image, 6, 8))

    assert actual.shape == (6, 8, 3)
    assert np.all(actual[:, :7] == 255)
    assert np.all(actual[:, 7:] == 0)


def test_same_size_returns_input_unchanged() -> None:
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

    actual = np.asarray(resize_with_pad(image, 8, 8))

    assert np.array_equal(actual, image)
