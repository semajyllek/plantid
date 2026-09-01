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

## Round 3: is averaging the wrong combiner?

Averaging is the naive choice, and since photos of one plant are *correlated*
the right combiner should sit between mean and max. Eleven combiners tested
(`plantid/eval/combiners.py`) on the 405 observations with ≥3 photos and ≥2
organs — 156 of them in-catalog, which is the n behind species accuracy.

**Method note.** A dev/test split selected *geometric mean* as the winner on dev
(species 0.885) — and it then **lost on test** (0.859 vs the mean baseline's
0.897). Marginal accuracies at this n swing ±5pp between folds, so all
comparisons below are **paired bootstraps** over observations, which control for
which observations are hard.

### Species accuracy: nothing beats the mean

| combiner | species | Δ vs mean | 95% CI |
|---|---|---|---|
| single (1st photo) | 0.853 | −0.032 | [−0.071, +0.006] |
| **mean (baseline)** | **0.885** | — | — |
| geometric mean | 0.872 | −0.013 | [−0.051, +0.019] |
| median | 0.853 | −0.032 | [−0.064, −0.006] * |
| max-confidence | 0.840 | −0.045 | [−0.090, +0.000] |
| confidence-weighted | 0.885 | +0.000 | [+0.000, +0.000] |
| top-2 mean | 0.878 | −0.006 | [−0.032, +0.019] |
| **trimmed mean** | **0.885** | +0.000 | [−0.026, +0.026] |
| organ-best | 0.878 | −0.006 | [−0.038, +0.026] |
| power mean p=4 | 0.885 | +0.000 | [−0.019, +0.019] |

*(\* = CI excludes zero)*

Every alternative ties or loses. **The mean is already at the ceiling for
accuracy.**

This also **walks back the round-2 claim** that fusion gives "+3 to +6pp
species". The paired CI for single-photo vs mean is [−0.071, +0.006] — it
includes zero at n=156. The point estimate still favours fusion (+3.2pp) and it
was positive across every filter in round 2, but it is **not** statistically
established on this sample.

### Rejection: trimmed mean is a small, real win

| combiner | distant-OOD | Δ | near-OOD | Δ |
|---|---|---|---|---|
| single | 0.939 | −0.006 | 0.861 | +0.001 |
| mean (baseline) | 0.945 | — | 0.860 | — |
| **trimmed mean** | **0.953** | **+0.008** * | **0.875** | **+0.016** * |
| median | 0.947 | +0.002 | 0.874 | +0.015 * |
| max-confidence | 0.956 | +0.011 | 0.853 | −0.006 |
| top-2 mean | 0.956 | +0.011 | 0.867 | +0.008 |
| geometric mean | 0.942 | −0.003 | 0.858 | −0.001 |
| power mean p=2 | 0.940 | −0.005 * | 0.852 | −0.008 * |
| power mean p=4 | 0.933 | −0.012 * | 0.845 | −0.015 * |

**Trimmed mean — drop the least confident photo, average the rest — is
significantly better than the mean on both OOD metrics while tying it on
accuracy.** It also beat the mean on rejection in *both* folds of the dev/test
split independently, so it is not a selection artifact.

The gain is modest (+0.008 / +0.016) but it is free: one line, no extra model.

### Sharpening hurts — which confirms the correlation story

Geometric mean, power mean p=2 and p=4 all treat photos as more-independent
evidence, and all are **significantly worse** (p=4: −0.012 distant, −0.015
near). That is direct empirical confirmation of why synthetic groups
overestimated fusion: photos of one plant are correlated, and any combiner that
assumes otherwise degrades.

**Recommendation: trimmed mean.** Mean is a fine default; trimmed mean is
strictly better and costs nothing.

## What this means



1. **The multi-organ fusion premise fails for rejection.** The claim that
   "three views give three chances to notice nothing matches"
   (`OPENSET_FINDINGS.md`) was measured on groups constructed to be independent.
   On real correlated photos the effect is ~0, even with ≥4 photos and confirmed
   organ diversity.
2. **Multi-photo capture is probably justified on accuracy**, +3pp species
   (point estimate, CI includes zero at n=156) and +2pp genus, positive across
   every filter. Combined with photo-quality control, that is the case for
   guided capture — not rejection.
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

## Round 4: closing the hole in species coverage

247 of the catalogue's 497 species had **no real observation at all** — half the
catalogue was being scored only on the PlantNet test split, the corpus its own
head was fitted on. A targeted fetch (`--species-file`, querying those species
by name rather than waiting for them to appear) returned 931 observations
covering **161** of the 247. In-catalogue coverage went 250 → 411 species,
2,365 → 3,284 observations.

What that cost is recorded in [`REJECTION_FINDINGS.md`](REJECTION_FINDINGS.md):
species accuracy on the new cohort is 0.791 against 0.872 on the old, a gap of
+0.081 with CI [+0.025, +0.141], while genus accuracy is statistically flat.
**The caveat this document has carried since round 2 — that in-catalogue
observations "skew toward common, heavily-photographed, easy species" — is now
partly measured rather than only asserted, and it was real.**

### The other 86 are mostly a naming problem, not a data problem

All 247 species were in fact queried — the 86 that came back empty are spread
evenly across the fetch's shuffled query order (positions 0 to 246, 21/26/11/28
by quartile), so the loop did not truncate against `per_bucket`. Their emptiness
is a property of the query, not of the run.

They were then resolved against iNat's active taxonomy. Fuzzy search is not safe
here — `q=Anemone apennina` returns *Anemonoides blanda*, a different plant — so
matches were accepted only on an exact `matched_term` (iNat lists our name as a
synonym) or a surviving specific epithet across a genus transfer:

| outcome | n | why the fetch missed it |
|---|---|---|
| **renamed** | **49** | catalogue uses the pre-split name |
| **crowded out** | **8** | name is current; page 1 is a commoner congener |
| resolved but genuinely unobservable | 7 | *Peperomia*, *Sedum* houseplants |
| unresolved | 22 | |

**The 49 renames.** The catalogue carries PlantNet's pre-split names. Every one
of the 18 *Anemone* entries has moved: `Anemone nemorosa` → *Anemonoides
nemorosa* (84,623 research-grade observations), `Anemone pulsatilla` →
*Pulsatilla vulgaris*, `Anemone hepatica` → *Hepatica nobilis*. *Sedum* has
split into *Phedimus* and *Petrosedum*, `Perovskia atriplicifolia` → *Salvia
yangii*, `Hebe` → *Veronica*, `Schefflera` → *Heptapleurum*, `Duchesnea indica`
→ *Potentilla indica*. The fetch *does* retrieve these — `taxon_name=Anemone
nemorosa` returns 77 observations of *Anemonoides nemorosa* — and then `_rows`
discards every one of them, because the returned binomial is not in
`catalog_species`.

**The 8 crowded out** are the more interesting failure, because their names are
current and nothing is wrong with them. `taxon_name` on the observations
endpoint is a *fuzzy* match that returns related taxa, and the first page of 100
is ordered by nothing that favours the species asked for:

| query | total | what page 1 actually contains |
|---|---|---|
| `Lactuca sativa` | 56,684 | **100/100 *Lactuca serriola*** — zero of the species requested |
| `Acalypha virginica` | 6,615 | 92 *A. rhomboidea*, 8 *A. virginica* |

So a species with 330 usable research-grade observations returns nothing,
because a commoner congener fills the page. This is not rarity and not
synonymy — it is the query.

Both failures have the same one-line fix: **query by `taxon_id`, which is
exact, and accept the resolved name as the catalogue species.** Closing all 57
would take the evaluated catalogue from 411 to ~468 of 497 species.

The three groups that are *not* recoverable are each blocked for a different
reason, and none is about the plant being rare:

- **7 cultivated-only** — *Peperomia albovittata*, *Sedum burrito* and similar
  have 0–3 research-grade observations because iNat grades cultivated plants
  "casual". They are common; they are just not *wild*. No fetch fixes this, and
  it is a genuine gap for a catalogue built from a horticulture-heavy corpus.
- **~9 are not species** — `Anemone x`, `Fragaria ×`, `Freesia x`,
  `Hypericum x`, `Lupinus x`, `Pelargonium spp.`, `Pelargonium x`,
  `Pelargonium ×`, `Tradescantia x` are PlantNet hybrid and genus-level
  placeholders sitting in a species catalogue. They should not be labels.
- **6 *Ophrys* microspecies** — *incubacea*, *lupercalis*, *occidentalis*,
  *passionis*, *virescens*, *arachnitiformis* are contested segregates of the
  *O. sphegodes* complex that iNat does not recognise as distinct taxa. Not a
  lookup failure: a taxonomic disagreement the catalogue inherited.

So the honest accounting of the original 247: **161 closed, 57 cheaply closable
(49 renamed, 8 crowded out), ~29 blocked for reasons that are properties of the
catalogue rather than of iNaturalist.** Roughly 88% of the hole is reachable.

The same naming gap also leaks into bucketing, because bucket membership is a
binomial string match — measured at 0.34% of OOD rows and quantified in
`REJECTION_FINDINGS.md`. It is small now and grows with the catalogue.

## Follow-ups

- ~~Re-run with ≥3 photos and ≥2 confidently-distinct organs~~ — done, round 2.
- The synthetic-group fusion numbers in `CNN_FINDINGS.md` and
  `OPENSET_FINDINGS.md` should be annotated as optimistic wherever they are
  used to justify multi-organ capture.
- Fusion currently averages photo posteriors. Since the gain is in accuracy, a
  better combiner (confidence-weighted, or dropping low-quality photos) may
  extend the +3–6pp — worth testing before the capture UI is designed.
