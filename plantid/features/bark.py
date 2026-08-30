"""Bark/stem texture descriptors: GLCM (Haralick) features, LBP histogram,
and a color histogram, computed on a center crop.

Bark/stem images are dominated by a central vertical structure (trunk or
stem) running through the frame; rather than attempt segmentation, we take a
center crop (consistent with bark-texture literature such as BarkNet, which
crops fixed patches from bark photos) and compute texture descriptors there.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

CROP_FRAC = 0.5
GLCM_LEVELS = 32
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM")
LBP_POINTS = 8
LBP_RADIUS = 1
COLOR_BINS = (8, 8, 8)


def center_crop(image_bgr: np.ndarray, frac: float = CROP_FRAC) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return image_bgr[y0 : y0 + ch, x0 : x0 + cw]


def texture_descriptor(image_bgr: np.ndarray) -> np.ndarray:
    """Haralick/GLCM features at multiple distances and angles."""
    crop = center_crop(image_bgr)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    quantized = (gray.astype(np.float32) / 256 * GLCM_LEVELS).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    feats = [graycoprops(glcm, prop).flatten() for prop in GLCM_PROPS]
    return np.nan_to_num(np.concatenate(feats))


def lbp_descriptor(image_bgr: np.ndarray) -> np.ndarray:
    crop = center_crop(image_bgr)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")

    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    total = hist.sum()
    return hist / total if total > 0 else hist.astype(np.float64)


def color_descriptor(image_bgr: np.ndarray, bins: tuple[int, int, int] = COLOR_BINS) -> np.ndarray:
    crop = center_crop(image_bgr)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(bins), [0, 180, 0, 256, 0, 256])
    total = hist.sum()
    hist = hist / total if total > 0 else hist
    return hist.flatten()


def describe(image_bgr: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [texture_descriptor(image_bgr), lbp_descriptor(image_bgr), color_descriptor(image_bgr)]
    )
