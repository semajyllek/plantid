# Phase 2 findings: per-organ descriptors and outlier pass

## Descriptor pipelines

All three pipelines follow the same shape: localize the subject, then compute
shape/texture/color descriptors over that region (or a center-weighted view of
the whole image where a hard mask isn't appropriate).

- **Leaf** (`plantid/features/leaf.py`, 529-dim): foreground = Otsu threshold
  on HSV saturation, intersected with a Gaussian center-weight prior (falls
  back to the center prior alone on low-saturation/flat images). Shape = 7
  log-Hu moments of the largest contour. Texture = center-weighted uniform-LBP
  histogram (venation/surface). Color = center-weighted 8x8x8 HSV histogram.

- **Bark/stem** (`plantid/features/bark.py`, 594-dim): no segmentation —
  a 50% center crop (consistent with bark-texture literature, e.g. BarkNet) is
  used directly for GLCM/Haralick features (6 props x 3 distances x 4 angles =
  72-dim), an LBP histogram, and an HSV color histogram.

- **Flower** (`plantid/features/flower.py`, 529-dim): foreground = Excess
  Green Index (ExG = 2g - r - b on normalized RGB; Woebbecke et al. 1995) to
  separate non-green flower regions from green foliage background, intersected
  with the center-weight prior (same fallback as leaf). Same Hu-moment /
  LBP / HSV-histogram descriptors as leaf.

GrabCut was tried first for both leaf and flower segmentation and abandoned —
it degenerated to all-foreground or all-background on this dataset's
low-contrast field photos. The center-weight + threshold combination is
deterministic, never degenerates, and visually produces reasonable masks
across spot-checked species (umbels, hop cones, thistle buds, geranium
flowers, ferny leaves on rocky backgrounds).

Throughput: ~33 images/sec across all three organs combined (leaf ~32ms,
flower ~30ms, bark ~4ms per image) — full 10,777-image dataset takes ~5
minutes on CPU.

## Outlier / label-noise pass

`plantid/features/outliers.py` computes the per-organ descriptor for every
image, z-scores each dimension within its (species, organ) group, and takes
each image's Euclidean distance to the group centroid in that normalized
space. Distances are converted to a robust "modified z-score" via median/MAD
(Iglewicz & Hoaglin, 1993); images with modified z-score > 3.5 are flagged.
Groups with fewer than 5 images are left unscored (too few samples for stable
centroid/MAD).

Results (`data/processed/outlier_scores.parquet`, 10,777 rows):

| organ  | n     | flagged | %    |
|--------|-------|---------|------|
| leaf   | 5,040 | 184     | 3.7% |
| flower | 4,571 | 184     | 4.0% |
| bark   | 1,166 | 49      | 4.2% |
| **total** | **10,777** | **417** | **3.9%** |

The known *Sedum sediforme* "leaf" mislabeling case from Phase 1 is among the
flagged leaf images (modified z ~3.5).

Spot-checking the highest-scoring outliers shows the method is catching real
problems, mostly **organ mislabeling / off-target framing** rather than image
corruption:

- Several "bark" outliers (*Daphne laureola*, *Sedum rupestre*, *Phalaris
  arundinacea*) are whole-plant or close-up foliage/flower shots with no bark
  or stem texture visible at all.
- Several "flower" outliers (*Punica granatum*, *Mercurialis annua*, *Angelica*
  spp.) are wide whole-plant or habit shots where the flower is a tiny part of
  the frame (or the "flower" is an inconspicuous catkin/inflorescence next to
  dominant foliage) — consistent with Pl@ntNet's documented label ambiguity.

A handful of species/organ combos concentrate flagged images (e.g. *Punica
granatum* flower: 10/~60, *Althaea officinalis* flower: 8, *Daphne mezereum*
leaf: 7) — these are candidates for manual review or exclusion before building
matching indices in Phase 3.

## Implications for Phase 3

- `outlier_scores.parquet` gives a per-image `is_outlier` flag and continuous
  `modified_z` score that can be used to (a) exclude flagged images from
  index-building, (b) down-weight them, or (c) report them separately in the
  eval suite so a mislabeled test image doesn't masquerade as a hard negative.
- The ~4% flag rate is roughly uniform across organs, so no organ needs a
  fundamentally different descriptor before Phase 3 — but the concentrated
  species/organ combos above are worth a manual look before they're used as
  index "reference" images for those species.
