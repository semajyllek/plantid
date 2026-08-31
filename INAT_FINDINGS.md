# Real multi-organ observations: fusion helps accuracy, not rejection

The last big unvalidated claim in this repo. Every fusion number here rests on
**synthetic** groups — one leaf, one bark, one flower drawn independently from
the same species (`data/groups.py`) — which assumes the organ images are
conditionally independent given species. That is what the sampler enforces, not
what a user does: a real person photographs *one individual plant*, so the
photos are correlated.

iNaturalist observations bundle several photos of the same individual, which is
the structure PlantNet-300K cannot provide (`obs_id` is 1:1 with images).

## Setup

`plantid/data/inat_eval.py` pulls research-grade plant observations with ≥2
photos into three buckets against the 261-species catalog:

| bucket | meaning | n |
|---|---|---|
| `in_catalog` | species is in the catalog → should be accepted | 150 |
| `near_ood` | species not in catalog, but its genus is → hard reject | 78 |
| `distant_ood` | neither species nor genus in catalog → easy reject | 150 |

378 observations, 1,022 photos, mean 2.7 photos each.

iNat photos are not organ-tagged, so the per-organ heads are combined by
**marginalising over organ**: `P(class|x) = Σ_o P(organ=o|x) · P(class|x,o)`,
using a 3-way organ router trained on the catalog's organ labels (79.2% accurate;
its errors are overwhelmingly leaf↔flower, which is genuine ambiguity — a photo
of a flowering plant contains both).

## Result: fusion adds almost nothing

| input | species | genus | distant-OOD AUROC | near-OOD AUROC |
|---|---|---|---|---|
| single photo | 0.913 | 0.987 | 0.943 | 0.883 |
| **all photos fused** | 0.907 | **1.000** | 0.944 | 0.890 |

**Fusing real photos of the same plant is worth +0.001 AUROC on distant OOD and
+0.007 on near OOD.** Species accuracy is flat-to-slightly-down. Only genus
accuracy improves (0.987 → 1.000).

Compare the synthetic-group result (`OPENSET_FINDINGS.md`): single organ
0.832–0.903 → fused 0.941, a gain of +0.04 to +0.11. **That gain does not
replicate on real observations.**

### It isn't just that observers shoot the same organ twice

The obvious defence is that iNat users take three photos of the same flower,
whereas guided capture would force organ diversity. Splitting observations by
how many distinct organs the router assigns:

| photos of | n obs | single | fused | gain | |
|---|---|---|---|---|---|
| 1 organ only | 237 | 0.949 | 0.944 | −0.005 | distant-OOD |
| 2+ distinct organs | 141 | 0.931 | 0.944 | **+0.013** | distant-OOD |
| 1 organ only | 237 | 0.917 | 0.923 | +0.006 | near-OOD |
| 2+ distinct organs | 141 | 0.801 | 0.808 | +0.006 | near-OOD |

Organ diversity helps a little — +0.013 versus −0.005 on distant OOD — but it is
an order of magnitude short of the synthetic-group gain. **The conditional-
independence assumption was doing most of the work.**

## Round 2: more photos, forced organ diversity

Round 1 had mean 2.7 photos/observation and only 90 observations met "≥3 photos
and ≥2 organs" (14 near-OOD — unusable). Refetched with `min_photos=3`,
`MAX_PHOTOS` raised to 6, and near-OOD queried by catalog *genus* rather than
random sampling: **696 observations, 2,627 photos, mean 3.77 each** — near-OOD
went from 78 to 175.

| filter | n | input | species | genus | distant-OOD | near-OOD |
|---|---|---|---|---|---|---|
| all (≥2 photos) | 696 | single | 0.835 | 0.958 | 0.920 | 0.832 |
| | | **fused** | **0.877** | **0.977** | 0.924 | 0.840 |
| ≥3 photos, ≥2 organs | 405 | single | 0.853 | 0.968 | 0.939 | 0.861 |
| | | **fused** | **0.885** | **0.981** | 0.945 | 0.860 |
| ≥3 photos, ≥2 organs, router conf ≥0.6 | 280 | single | 0.833 | 0.963 | 0.937 | 0.838 |
| | | **fused** | **0.889** | **0.981** | 0.940 | 0.830 |
| ≥4 photos, ≥2 organs | 199 | single | 0.849 | 0.973 | 0.955 | 0.872 |
| | | **fused** | **0.890** | **1.000** | 0.953 | 0.865 |

### The split: fusion helps accuracy, not rejection

**Rejection gain is ~zero and does not improve with better conditions.** Across
every filter, distant-OOD moves +0.006 / +0.003 / −0.002 and near-OOD moves
−0.001 / −0.008 / −0.007. Giving fusion its best case — more photos, confirmed
organ diversity — does not rescue it. The synthetic-group gain of +0.04 to +0.11
is not there.

**Accuracy gain is real and consistent: +3 to +6pp species, +2pp genus.**
0.853 → 0.885 at ≥3 photos/≥2 organs, 0.833 → 0.889 under the strict router
filter, and genus reaches 1.000 at ≥4 photos.

This **corrects round 1**, which reported species 0.913 → 0.907 (flat-to-down)
on 150 in-catalog observations. At 261 observations spanning more species, the
species gain from fusion is consistently positive. Round 1's absolute levels
were also inflated by sampling a smaller, easier species set — 0.835 here versus
0.913 there is the more representative number.

## What this means

1. **The multi-organ fusion premise fails for rejection.** The claim that
   "three views give three chances to notice nothing matches"
   (`OPENSET_FINDINGS.md`) was measured on groups constructed to be independent.
   On real correlated photos the effect is ~0, even with ≥4 photos and confirmed
   organ diversity.
2. **But multi-photo capture is justified on accuracy**: +3 to +6pp species and
   +2pp genus, consistently across filters. So guided capture should be built —
   the case for it is accuracy and photo quality, not rejection.
3. **Rejection is a single-photo capability, and it transfers.** Distant-OOD
   AUROC is 0.92–0.96 from one photo on a genuinely different source, and fusing
   adds nothing. The `__OTHER__` reject class, not the number of views, is what
   makes "I don't know" work.
4. **Near-OOD remains the hard case** (0.83–0.87 vs 0.92–0.96), consistent with
   every other experiment here, and reinforces the genus-level fallback
   (`HIERARCHY_FINDINGS.md`) as the right answer for it — genus accuracy is
   0.977–1.000 fused even where species is 0.88.

## Caveats — the accuracy levels are inflated

Round-2 species accuracy (0.835–0.890) still sits **above our own catalog test
set** (0.759–0.808), and round 1's 0.913 was higher still. Two reasons, both
pushing the same way:

- **Sampling bias**: in-catalog observations were found by querying species
  names, so they skew toward common, heavily-photographed, easy species.
- **Contamination**: iNaturalist feeds GBIF, which feeds TreeOfLife-200M, which
  BioCLIP-2 was trained on (`DATA_STRATEGY.md`). These exact photos may be in
  the encoder's training data.

So treat the levels as upper bounds. **The single-vs-fused comparison is the
reliable part** — both arms are affected identically by both biases, so the
*difference* between them is sound even though the levels are not.

Other limits: the router is 79% accurate, so "distinct organs" is noisy at
argmax — hence the `router-conf ≥0.6` row, which tells the same story on a
cleaner but smaller sample (n=280). The ≥4-photo row is n=199, so its −0.002
distant-OOD gain is well within noise; the point is that it is not *positive*.

## Follow-ups

- ~~Re-run with ≥3 photos and ≥2 confidently-distinct organs~~ — done, round 2.
- The synthetic-group fusion numbers in `CNN_FINDINGS.md` and
  `OPENSET_FINDINGS.md` should be annotated as optimistic wherever they are
  used to justify multi-organ capture.
- Fusion currently averages photo posteriors. Since the gain is in accuracy, a
  better combiner (confidence-weighted, or dropping low-quality photos) may
  extend the +3–6pp — worth testing before the capture UI is designed.
