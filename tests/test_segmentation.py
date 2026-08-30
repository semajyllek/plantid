import numpy as np

from plantid.features.segmentation import center_mask, center_weight


def test_center_weight_peaks_at_center():
    w = center_weight((100, 100))
    assert w.shape == (100, 100)
    assert w[50, 50] == w.max()
    assert w[0, 0] < w[50, 50]
    assert (w > 0).all()
    assert w.max() <= 1.0


def test_center_weight_handles_non_square():
    w = center_weight((50, 200))
    assert w.shape == (50, 200)


def test_center_mask_is_binary_and_nonempty():
    m = center_mask((100, 100))
    assert set(np.unique(m)) <= {0, 1}
    assert m.sum() > 0
    assert m.sum() < m.size  # not everything is foreground
