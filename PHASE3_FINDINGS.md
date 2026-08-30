# Phase 3 findings: matching & fusion

## Classical descriptor k-NN matcher

`plantid/matching/classical.py` builds a per-organ gallery from the Phase 2
descriptors (`features/store.py`), restricted to the train split with
flagged outliers (`outlier_scores.parquet`) excluded. Each descriptor
dimension is z-scored using gallery statistics (raw descriptors mix log-Hu
moments, LBP histograms, and color histograms on very different scales).
Species predictions are k-NN inverse-distance-weighted votes.

`plantid/eval/match_eval.py` sweeps k in {1,5,10,15,20,30} on the val split
to pick a per-organ k, then reports top-1/5/10 species accuracy on the test
split (87 species; random baselines: top1=1.1%, top5=5.7%, top10=11.5%).

| organ  | best_k | top1  | top5  | top10 |
|--------|--------|-------|-------|-------|
| leaf   | 15     | 0.065 | 0.183 | 0.283 |
| bark   | 30     | 0.098 | 0.257 | 0.383 |
| flower | 15     | 0.156 | 0.348 | 0.472 |

All three organs land well above the random baseline (6-14x at top-1, 3-6x
at top-5). **Flower is the strongest single-organ signal** — likely because
HSV color histograms capture a lot of discriminative signal for
flower-bearing species, where leaf shape/texture and bark texture are more
visually similar across species. Bark, despite being the smallest gallery
(800 train images vs. 3199-3530 for the others) and the organ with the most
heterogeneous label semantics (Phase 1), outperforms leaf — its GLCM/LBP
texture descriptors apparently transfer better than leaf shape/venation
across this species set.

Per-species worst performers (`per_species_top1`) are scattered across all
three organs with no single dominant failure mode — consistent with the
~4% outlier rate from Phase 2 being a contributing but not sole factor.

## imret (ORB + FAISS binary IVF) comparison

The pre-built `imret` wheel (`dist/imret-0.1.0-cp314-cp314-macosx_15_0_arm64.whl`)
installs cleanly and its `Vault` API (`add_batch`, `build`, `search`) works
out of the box. Two small-scale 1-NN smoke tests (10 species each, full
train/test split, `confidence_threshold=0.0` to bypass the fallback gate):

| organ | train | test | accuracy | random (1/10 species) |
|-------|-------|------|----------|------------------------|
| leaf  | 420   | 50   | 0.16     | 0.10                   |
| bark  | 292   | 62   | 0.11     | 0.10                   |

Both are essentially at chance. This confirms the Phase 1 hypothesis: imret's
ORB-keypoint + Hamming-distance design is built for **instance-level**
near-duplicate matching (same physical object, different photo) and doesn't
transfer to **species-level** classification, where two photos of the same
species (different individual plants, lighting, angle) share almost no
matching local keypoints. The public API also only exposes 1-NN (`search()`
returns a single `MatchResult`), so there's no way to do k-NN voting as with
the classical descriptors. Given near-chance results at small scale and no
top-k API, a full 87-species imret evaluation wasn't pursued further —
**classical descriptors are the clear matcher of choice** for this task.

## Late fusion (reciprocal rank fusion across organs)

`plantid/matching/fusion.py` evaluates fusion on synthetic
`{leaf, bark, flower, plant_id}` groups (`data/groups.py`, 1740 test groups)
using reciprocal rank fusion (RRF, Cormack et al. 2009): each organ's ranked
species list contributes `1/(60 + rank)` to a fused score per species. A
weighted variant scales each organ's contribution by that organ's val-split
top-1 accuracy (leaf 0.069, bark 0.104, flower 0.162).

| top-k | leaf  | bark  | flower | fused (unweighted) | fused (weighted) |
|-------|-------|-------|--------|---------------------|-------------------|
| 1     | 0.064 | 0.053 | 0.132  | 0.094               | 0.078             |
| 5     | 0.166 | 0.139 | 0.309  | 0.280               | 0.314             |
| 10    | 0.248 | 0.233 | 0.435  | 0.438               | 0.451             |

**Fusion does not beat the best single organ (flower) at top-1** — even
weighted RRF (0.078) is below flower alone (0.132). At top-5/top-10,
confidence-weighted fusion gives a small edge over flower alone (0.314 vs
0.309, 0.451 vs 0.435), but unweighted fusion does not.

This is a real, somewhat counter-intuitive result: when one organ (flower) is
clearly more discriminative than the others, naive or weakly-weighted RRF
dilutes its rank-1 dominance rather than reinforcing it. Implication for
Phase 6 (the CLI tool): **if a flower image is available, querying with it
alone and falling back to fusion only when it's missing may be a better
default than always fusing** — though this should be re-checked once Phase 4
(synthetic data) and any descriptor improvements change the per-organ
baselines.

### Per-query confidence-weighted fusion (negative result)

`fuse_rankings`/`evaluate_fusion` (`matching/fusion.py`) gained a
`query_confidence` option: weight each organ's RRF contribution for a given
query by that organ's own top-1 inverse-distance vote share (how "peaked"
its k-NN vote is for this specific query), instead of (or combined with) the
fixed val-accuracy weight.

| top-k | flower | fused (unweighted) | fused (val-weighted) | fused (query-confidence) | fused (combined) |
|-------|--------|---------------------|------------------------|----------------------------|-------------------|
| 1     | 0.132  | 0.094               | 0.078                  | 0.094                      | 0.090             |
| 5     | 0.309  | 0.280               | 0.314                  | 0.297                      | 0.323             |
| 10    | 0.435  | 0.438               | 0.451                  | 0.434                      | 0.432             |

Per-query confidence weighting does **not** change the headline conclusion:
no fusion variant beats flower-alone at top-1. With `RRF_K=60` and
`k_match<=15`, `1/(RRF_K+rank)` varies only ~20% across all candidate ranks,
so RRF is closer to "count of organs that agree" than "trust the most
decisive organ" — a per-query confidence multiplier rescales an organ's
whole contribution but can't change the shape of that curve. Improving
fusion further would need a different combination rule entirely (e.g.
score-level fusion instead of rank-level), which is lower priority than
improving the underlying per-organ descriptors (see Phase 4 plan below).

## Implications for Phase 4 / 5

- The classical-descriptor pipeline (leaf/bark/flower descriptors + k-NN +
  RRF fusion) is the baseline to beat. Current numbers (flower top1=0.156,
  fused top10=0.451 on 87 species) are modest but far above chance — there's
  clear room for improvement via better descriptors, synthetic data
  augmentation (Phase 4), or smarter fusion weighting (e.g. per-query
  confidence rather than a fixed per-organ weight).
- imret is not part of the Phase 5 eval suite's matcher comparison going
  forward, given near-chance results and lack of a k-NN API — but the
  `OrbConfig`/`Vault` smoke-test code in this investigation could be revisited
  if imret gains a top-k search API.
- The eval harness (`match_eval.evaluate_organ`, `fusion.evaluate_fusion`)
  is reusable as-is for Phase 4's "measured against the eval suite, kept only
  if it helps" requirement — any new descriptor or augmentation can be
  dropped into `features/store.py`'s cache and re-evaluated with the same
  code.
