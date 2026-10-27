import jax.numpy as jnp
import pytest

from openpi.shared import image_tools


def test_resize_with_pad_shapes():
    # Test case 1: Resize image with larger dimensions
    images = jnp.zeros((2, 10, 10, 3), dtype=jnp.uint8)  # Input images of shape (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 2: Resize image with smaller dimensions
    images = jnp.zeros((3, 30, 30, 3), dtype=jnp.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 3: Resize image with the same dimensions
    images = jnp.zeros((1, 50, 50, 3), dtype=jnp.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 4: Resize image with odd-numbered padding
    images = jnp.zeros((1, 256, 320, 3), dtype=jnp.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)


def test_640x480_image_is_padded_to_640_without_resizing():
    image = jnp.full((480, 640, 3), 127, dtype=jnp.uint8)

    padded = image_tools.resize_with_pad(image, 640, 640)

    assert padded.shape == (640, 640, 3)
    assert jnp.array_equal(padded[80:560], image)
    assert jnp.all(padded[:80] == 0)
    assert jnp.all(padded[560:] == 0)


def test_pad_to_size_preserves_pixels_and_rejects_oversized_images():
    image = jnp.arange(480 * 640 * 3, dtype=jnp.uint8).reshape(480, 640, 3)

    padded = image_tools.pad_to_size(image, 640, 640)

    assert padded.shape == (640, 640, 3)
    assert jnp.array_equal(padded[80:560], image)
    assert jnp.all(padded[:80] == 0)
    assert jnp.all(padded[560:] == 0)
    with pytest.raises(ValueError, match="never resizes"):
        image_tools.pad_to_size(jnp.zeros((720, 1280, 3), dtype=jnp.uint8), 640, 640)
