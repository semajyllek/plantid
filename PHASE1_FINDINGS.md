# Phase 1 findings: PlantNet-300K data audit

## Dataset

Source: [PlantNet-300K](https://github.com/plantnet/PlantNet-300K) metadata
(`data/raw/plantnet300k/`), images fetched via the Pl@ntNet image API
(`https://bs.plantnet.org/image/m/{image_id}`, "medium" size).

- 306,146 images, 1,081 species total. Severe long tail: top 10% of species
  hold 77% of images.
- Organ distribution: flower 57.7%, leaf 36.2%, fruit 3.7%, habit 1.5%,
  **bark 0.7%** (2,076 images). Bark is the bottleneck organ.
- `obs_id` is 1:1 with images — there is no native multi-organ "observation"
  to group on. Groups must be formed per-species.

## Working dataset (this repo)

`plantid/data/build_dataset.py` selects species with ≥5 images for each of
leaf/flower/bark, caps each (species, organ) at 60 images, and re-splits
train/val/test (15%/15%/70%, minimum 1 each) so every combination has
non-empty val and test sets.

Result: **87 species, 10,777 images** (leaf 5,040 / flower 4,571 / bark 1,166).
Manifest: `data/processed/plantnet_index.parquet`. Images:
`data/processed/images/{species_id}/{organ}/{image_id}.jpg`.

`plantid/data/groups.py` samples synthetic `{leaf_img, bark_img, flower_img,
plant_id}` groups per (species, split) for multi-organ fusion evaluation —
synthetic because no real co-occurring observations exist; images are reused
across groups when an organ (usually bark) is scarce.

## Image format

All images are exactly square (aspect ratio 1.0, std 0), max dimension 600px,
3-channel RGB. PlantNet's "m" size is a center-crop-and-resize to square —
some original content is lost at the edges.

## Visual / label-quality findings (from spot checks)

1. **Real field photos, not lab shots.** Cluttered, often green-on-green
   backgrounds (foliage behind foliage, stone walls, mulch). Any
   segmentation/shape-descriptor approach in Phase 2 needs to handle this —
   simple color thresholding (green vs. background) won't reliably isolate
   the subject.

2. **"bark" organ is semantically heterogeneous** (confirmed visually):
   - Woody species (*Liriodendron tulipifera*, *Acacia dealbata*): genuine
     bark/trunk texture — good for GLCM/LBP texture descriptors.
   - Herbaceous species (*Daucus carota*, *Lamium*, etc.): a hairy/smooth
     green stem close-up — texture descriptors alone may not transfer well;
     these look more like elongated-object shape problems.
   - Decision (per discussion): treat uniformly for now, let the evaluation
     suite reveal whether a woody/herbaceous split is needed.

3. **Label noise exists**, consistent with the Pl@ntNet-300K paper's own
   "high label ambiguity" framing. Example: *Sedum sediforme* (a succulent)
   has a "leaf" image that is a close-up of grass/wheat spikelets, and a
   "bark" image that's a whole potted-plant shot ("habit", not stem). This
   is not a one-off — expect a non-trivial fraction of mislabeled/
   off-target images in the working set.

## Implications for Phase 2

- Per-organ classical descriptors (leaf shape/venation, bark/stem texture,
  flower color/shape) need to be robust to background clutter — likely need
  a foreground-localization step (e.g., saliency, GrabCut, or center-weighted
  sampling) before descriptor extraction, not just whole-image features.
- Label noise means the evaluation suite (Phase 5) should report per-class
  and per-image outlier diagnostics, not just aggregate accuracy — a
  mislabeled training image will look like a hard negative/positive and can
  silently depress retrieval scores. Worth a lightweight outlier-detection
  pass (e.g., distance-to-class-centroid on a cheap descriptor) before
  building indices, but the scope of that is a Phase 2 decision.
