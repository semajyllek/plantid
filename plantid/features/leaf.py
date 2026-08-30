"""Leaf descriptors: shape (Hu moments), venation texture (LBP), color (HSV
histogram).

Shape descriptors need a binary foreground region. We get one via Otsu
thresholding on HSV saturation (leaves are typically more saturated than
blurry/sky/dirt backgrounds), intersected with the center-weight prior from
segmentation.py so that green-on-green scenes fall back to the center region
rather than the whole frame or nothing.

Texture and color descriptors don't need a hard mask — they're computed over
the whole image, weighted by the center prior.
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from plantid.features.segmentation import center_mask, center_weight

LBP_POINTS = 8
LBP_RADIUS = 1
COLOR_BINS = (8, 8, 8)


def foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, otsu = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_mask = (otsu > 0).astype(np.uint8)

    cmask = center_mask(image_bgr.shape[:2])
    combined = otsu_mask & cmask

    if combined.sum() < 0.02 * combined.size:
        return cmask
    return combined


def shape_descriptor(image_bgr: np.ndarray) -> np.ndarray:
    """7 log-scaled Hu moments of the largest foreground contour."""
    mask = foreground_mask(image_bgr)
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros(7, dtype=np.float64)

    largest = max(contours, key=cv2.contourArea)
    hu = cv2.HuMoments(cv2.moments(largest)).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def texture_descriptor(image_bgr: np.ndarray) -> np.ndarray:
    """Center-weighted uniform-LBP histogram, capturing venation/surface texture."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    weight = center_weight(gray.shape)

    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), weights=weight)
    total = hist.sum()
    return hist / total if total > 0 else hist


def color_descriptor(image_bgr: np.ndarray, bins: tuple[int, int, int] = COLOR_BINS) -> np.ndarray:
    """Center-weighted 3D HSV color histogram, flattened."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    weight = center_weight(image_bgr.shape[:2])

    sample = hsv.reshape(-1, 3)
    hist, _ = np.histogramdd(
        sample, bins=bins, range=[(0, 180), (0, 256), (0, 256)], weights=weight.ravel()
    )
    total = hist.sum()
    hist = hist / total if total > 0 else hist
    return hist.flatten()


def describe(image_bgr: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [shape_descriptor(image_bgr), texture_descriptor(image_bgr), color_descriptor(image_bgr)]
    )
