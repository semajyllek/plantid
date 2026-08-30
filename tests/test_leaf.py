import cv2
import numpy as np

from plantid.features import leaf
from plantid.features.segmentation import center_mask


def _gray_bg_with_green_ellipse(size=200):
    img = np.full((size, size, 3), 128, dtype=np.uint8)  # gray, zero saturation
    cv2.ellipse(
        img,
        (size // 2, size // 2),
        (size // 4, size // 3),
        0,
        0,
        360,
        (0, 255, 0),  # green, BGR
        -1,
    )
    return img


def _flat_gray(size=200):
    return np.full((size, size, 3), 128, dtype=np.uint8)


def test_foreground_mask_isolates_saturated_region():
    img = _gray_bg_with_green_ellipse()
    mask = leaf.foreground_mask(img)
    assert mask.shape == img.shape[:2]
    assert set(np.unique(mask)) <= {0, 1}
    frac = mask.mean()
    assert 0.05 < frac < 0.6  # roughly the ellipse, not everything/nothing


def test_foreground_mask_falls_back_to_center_on_flat_image():
    img = _flat_gray()
    mask = leaf.foreground_mask(img)
    expected = center_mask(img.shape[:2])
    assert np.array_equal(mask, expected)


def test_shape_descriptor_shape_and_finite():
    img = _gray_bg_with_green_ellipse()
    desc = leaf.shape_descriptor(img)
    assert desc.shape == (7,)
    assert np.all(np.isfinite(desc))


def test_texture_descriptor_is_normalized_histogram():
    img = _gray_bg_with_green_ellipse()
    desc = leaf.texture_descriptor(img)
    assert desc.shape == (leaf.LBP_POINTS + 2,)
    assert np.all(desc >= 0)
    assert np.isclose(desc.sum(), 1.0, atol=1e-6)


def test_color_descriptor_is_normalized_histogram():
    img = _gray_bg_with_green_ellipse()
    desc = leaf.color_descriptor(img)
    assert desc.shape == (8 * 8 * 8,)
    assert np.all(desc >= 0)
    assert np.isclose(desc.sum(), 1.0, atol=1e-6)


def test_describe_concatenates_all_three():
    img = _gray_bg_with_green_ellipse()
    desc = leaf.describe(img)
    assert desc.shape == (7 + (leaf.LBP_POINTS + 2) + 8 * 8 * 8,)
