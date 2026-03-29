import numpy as np
import pytest

from backend.modules.input.preprocess import (
    InvalidImageError,
    normalize_image,
    resize_image,
    to_grayscale,
    validate_image,
)


def test_validate_image_extension():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    assert validate_image("test.jpg", img) is True

    with pytest.raises(InvalidImageError):
        validate_image("test.txt", img)


def test_validate_image_resolution():
    small_img = np.zeros((50, 50, 3), dtype=np.uint8)

    with pytest.raises(InvalidImageError):
        validate_image("small.jpg", small_img)


def test_resize_image():
    img = np.zeros((500, 300, 3), dtype=np.uint8)
    resized = resize_image(img, (256, 256))
    assert resized.shape == (256, 256, 3)


def test_normalize_image():
    img = np.ones((64, 64, 3), dtype=np.uint8) * 255
    normalized = normalize_image(img)

    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_to_grayscale():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    gray = to_grayscale(img)

    assert len(gray.shape) == 2
    assert gray.shape == (128, 128)
