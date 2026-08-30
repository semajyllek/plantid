import numpy as np

from plantid.features import bark


def _checkerboard(size=200, square=10):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, square):
        for x in range(0, size, square):
            if (x // square + y // square) % 2 == 0:
                img[y : y + square, x : x + square] = 255
    return img


def _flat(size=200, value=128):
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_center_crop_shape():
    img = _flat(200)
    crop = bark.center_crop(img, frac=0.5)
    assert crop.shape == (100, 100, 3)


def test_texture_descriptor_finite_and_shape():
    expected_len = len(bark.GLCM_PROPS) * len(bark.GLCM_DISTANCES) * len(bark.GLCM_ANGLES)
    desc = bark.texture_descriptor(_checkerboard())
    assert desc.shape == (expected_len,)
    assert np.all(np.isfinite(desc))


def test_texture_descriptor_distinguishes_textured_from_flat():
    flat_desc = bark.texture_descriptor(_flat())
    checker_desc = bark.texture_descriptor(_checkerboard())
    # contrast (first GLCM prop block) should be ~0 for flat, > 0 for checkerboard
    n_per_prop = len(bark.GLCM_DISTANCES) * len(bark.GLCM_ANGLES)
    assert np.allclose(flat_desc[:n_per_prop], 0)
    assert checker_desc[:n_per_prop].sum() > 0


def test_lbp_descriptor_is_normalized_histogram():
    desc = bark.lbp_descriptor(_checkerboard())
    assert desc.shape == (bark.LBP_POINTS + 2,)
    assert np.all(desc >= 0)
    assert np.isclose(desc.sum(), 1.0, atol=1e-6)


def test_color_descriptor_is_normalized_histogram():
    desc = bark.color_descriptor(_checkerboard())
    assert desc.shape == (8 * 8 * 8,)
    assert np.all(desc >= 0)
    assert np.isclose(desc.sum(), 1.0, atol=1e-6)


def test_describe_concatenates_all_three():
    n_glcm = len(bark.GLCM_PROPS) * len(bark.GLCM_DISTANCES) * len(bark.GLCM_ANGLES)
    desc = bark.describe(_checkerboard())
    assert desc.shape == (n_glcm + (bark.LBP_POINTS + 2) + 8 * 8 * 8,)
