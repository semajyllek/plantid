# Real multi-organ observations: fusion's benefit largely disappears

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

## What this means

1. **The multi-organ fusion premise is much weaker than this repo believed.**
   The claim that "three views give three chances to notice nothing matches"
   (`OPENSET_FINDINGS.md`) was measured on groups constructed to be independent.
   On real correlated photos the effect is ~+0.01, not ~+0.10.
2. **Guided capture is still defensible, but not on these grounds.** Its real
   value is photo *quality* control (framing, focus, subject isolation) and
   knowing which organ the user is photographing — not a large fusion gain in
   rejection. The case for it should be made on quality, not on fusion.
3. **Rejection itself holds up well on real data** — distant-OOD AUROC 0.943
   from a *single* photo, near-OOD 0.883. That is the encouraging half of this
   result: the `__OTHER__` mechanism transfers to a genuinely different source.
4. **Near-OOD remains the hard case** (0.883 vs 0.943), consistent with every
   other experiment here, and reinforces the genus-level fallback
   (`HIERARCHY_FINDINGS.md`) as the right answer for it — note genus accuracy
   reaches 1.000 fused.

## Caveats — the accuracy numbers are inflated

Species accuracy here (0.913) is **well above our own catalog test set**
(0.759–0.808). Two reasons, both pushing the same way:

- **Sampling bias**: in-catalog observations were found by querying species
  names, so they skew toward common, heavily-photographed, easy species.
- **Contamination**: iNaturalist feeds GBIF, which feeds TreeOfLife-200M, which
  BioCLIP-2 was trained on (`DATA_STRATEGY.md`). These exact photos may be in
  the encoder's training data.

So treat 0.913 / 1.000 as upper bounds. **The fusion comparison is the reliable
part** — single and fused are affected identically by both biases, so the
*difference* between them is sound even though the levels are not.

Other limits: 141 observations with 2+ distinct organs is a modest n; the router
is 79% accurate so "organ diversity" is noisy; mean 2.7 photos per observation
is fewer than a guided flow would collect.

## Follow-ups

- Re-run with observations filtered to ≥3 photos and ≥2 confidently-distinct
  organs, to give fusion its best case.
- The synthetic-group fusion numbers in `CNN_FINDINGS.md` and
  `OPENSET_FINDINGS.md` should be annotated as optimistic wherever they are
  used to justify multi-organ capture.
