"""Foreground localization shared by all per-organ descriptor pipelines.

GrabCut (seeded from a center rectangle) was tried first but proved unreliable
on this dataset: many images have low color/texture contrast between the
subject and background (e.g. bark vs. a stone wall, or a leaf against other
foliage), causing GrabCut to degenerate to "everything is foreground" or
"everything is background" depending on init mode.

Instead we use a soft, deterministic center-weighting prior: a 2D Gaussian
centered on the image, reflecting the assumption that the photographer framed
the subject roughly in the middle. This never degenerates and costs nothing
to compute. Descriptors that support weighted inputs (histograms, moments)
should use this weight map directly; descriptors that need a hard mask can
threshold it.
"""

import cv2
import numpy as np


def center_weight(shape: tuple[int, int], sigma_frac: float = 0.35) -> np.ndarray:
    """Return a float32 weight map of the given (h, w), peaking at 1.0 in the
    center and falling off as a Gaussian with sigma = sigma_frac * min(h, w).
    """
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sigma = sigma_frac * min(h, w)

    yy, xx = np.mgrid[0:h, 0:w]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)


def center_mask(shape: tuple[int, int], threshold: float = 0.2, sigma_frac: float = 0.35) -> np.ndarray:
    """Hard binary mask (uint8, 1=foreground) from thresholding `center_weight`."""
    weight = center_weight(shape, sigma_frac=sigma_frac)
    return (weight >= threshold).astype(np.uint8)
