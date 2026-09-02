# Location prior: real signal, but it does not pay

**Verdict against the pre-registered rule: do not ship.** The location gate's
expected-utility gain is +0.0069 with a 95% CI of [−0.0025, +0.0220] — the
interval includes zero, so Phase 8 closes as measured-and-rejected *in the form
that was tested*.

The signal underneath is real, and the reason it does not pay is specific and
informative. Both are recorded, because "geography does not help" would be the
wrong lesson.

## Setup

- **The prior**: 40,138 georeferenced iNaturalist observations across 214 of 248
  catalogue species (`plantid/data/species_ranges.py`), as an exponential
  distance kernel at 250 km bandwidth, pseudo-count smoothed so no species is
  ever ruled out.
- **The evaluation**: 2,600 of 2,601 observations now carry coordinates
  (`plantid/data/inat_locations.py`). Fitted on calibration, reported on test,
  cluster-bootstrapped.
- **Product constraint, decided before measuring**: location *gates* — it may
  withhold or generalise an answer — but never renames. A test asserts the gate
  can only reduce specificity, never change the label.

## The leak, closed

Range points were sampled from the same pool the evaluation came from.
**113 of 750 in-catalogue evaluation observations (15.1%) were verbatim in the
range sample** — for species with few global records, sampling 200 captures most
of them. `species_ranges.py` now records `obs_id` and `load_prior` drops any
range point that is an evaluation observation (120 dropped). Without this the
prior would have been partly built from the test point.

## Test A — there *is* identity information in location

Of the 32 species errors on the in-catalogue test split, **20 (62%) named a
species that is locally implausible while the true species is locally
plausible.** That is a ceiling of **5.4pp of species accuracy** available to
re-ranking.

We are not taking it: the product decision was that location must never change
the name. Recorded as what is being deliberately left on the table.

## Test B — the gate does not pay

| configuration | utility (test) | paired gain | 95% CI | verdict |
|---|---|---|---|---|
| baseline (no location) | +0.7731 | — | — | — |
| location gate, `t_loc` fitted alone | +0.7801 | **+0.0069** | [−0.0025, +0.0220] | fails |
| location gate, all three thresholds fitted jointly | +0.7662 | **−0.0069** | [−0.0536, +0.0347] | fails |

Joint fitting relaxes the vision thresholds (`t_species` 0.897 → 0.517) so
location can do more work — and it changes 45% of in-catalogue decisions — but
it loses on test. Both configurations fail; the rule was fixed in advance and is
honoured.

### Why it fails, precisely

The gate barely gets to act. At the fitted thresholds it changes **0.3–2.9%** of
decisions, because the vision thresholds are already conservative enough that
almost everything location would catch has been declined already. The AUROC gain
lands in a region of the curve the operating point never visits — exactly the
failure mode this project has recorded before, where fusion improved a ranking
metric and moved the decision not at all.

## But the signal is genuinely real

Three results say this is biology, not an artefact of where the buckets were
sampled:

**Permutation control.** Shuffling coordinates *within each bucket* preserves
each bucket's geography while destroying the species-location pairing. A
geographic artefact would survive; a real signal must collapse:

| | near-OOD | regional-OOD | distant-OOD |
|---|---|---|---|
| real | **0.715** | 0.766 | 0.803 |
| coordinates shuffled | 0.511 | 0.475 | 0.513 |

It collapses to chance. The signal is about the plant, not the place.

**It adds to the vision chain**, most where the system is weakest:

| bucket | genus confidence | with location | Δ |
|---|---|---|---|
| near-OOD (the hard case) | 0.792 | **0.815** | **+0.023** |
| regional-OOD | 0.978 | 0.983 | +0.005 |
| distant-OOD | 0.975 | 0.979 | +0.004 |

**It is partly independent** of the existing scores: Pearson r = +0.33 against
genus confidence, where the three vision scores are a nested chain and perfectly
ordered by construction.

## The score that would have faked a result

The obvious location feature — total prior mass near the user, i.e. how much of
the catalogue is plausible here — is **inadmissible by construction**, and is
kept in the code only as a control. It is a pure function of the coordinates. In
deployment a catalogue plant and an unknown one arrive from the same user at the
same coordinates, so it has *zero* discriminative power there. It scores at all
on this evaluation only because the buckets were sampled from different parts of
the world:

| | near-OOD | regional-OOD | distant-OOD |
|---|---|---|---|
| `loc_mass` (inadmissible) | 0.670 | **0.500** | 0.601 |

The 0.500 on regional-OOD gives it away: that bucket was drawn from the same
continents as the in-catalogue set, so the artefact vanishes exactly where the
geography matches. The shipped score (`loc_prq`) is a *within-location* rank
instead, invariant to how well-recorded a place is.

## Within-genus re-ranking — the narrow form, tested

Re-ranking was reopened in its narrow form: the genus stays decided by the
photograph, and location only chooses among **congeners** — where confusions
concentrate and where closely related species are separated more by range than
by appearance. 80% of in-catalogue test observations have a predicted genus with
≥2 catalogue species, so are eligible at all.

The exponent `w` in `posterior · prior**w` was fitted on calibration, with
`w = 0` in the grid so the fit could decline to use location. It chose
**w = 0.25**, and a sweep on test confirms that generalised (0.949 at 0.25,
0.944 at 0.5, 0.922 at 1.0, 0.810 at 2.0 — over-weighting the prior is harmful,
as it should be).

### At n=373 (first measurement)

| variant | species accuracy | | gain vs current | 95% CI | fixed / broke |
|---|---|---|---|---|---|
| current model | 0.914 | | — | — | — |
| genus-constrained, no location | 0.906 | | — | — | — |
| within-genus re-rank | **0.949** | | **+0.0349** | [−0.0053, +0.0808] | 20 / 7 |

### At n=1,150 — the effect largely evaporates

The in-catalogue bucket was expanded to 2,283 observations over 211 species
(from 750 over 131) specifically to settle this. Test split: **1,150
observations over 106 species**, roughly triple.

| variant | species accuracy | | gain vs current | 95% CI | fixed / broke |
|---|---|---|---|---|---|
| current model | 0.917 | | — | — | — |
| genus-constrained, no location | 0.916 | | — | — | — |
| within-genus re-rank | 0.923 | | **+0.0061** | [−0.0108, +0.0223] | **28 / 21** |

**+3.5pp became +0.6pp.** The confidence interval tightened as predicted — width
0.086 → 0.033, almost exactly the halving the power calculation implied — and it
still contains zero, now centred near it.

The fix-to-break ratio is the tell. At n=373 it was 20:7, which I cited as
evidence the effect was real. At n=1,150 it is **28:21**, close to parity: the
re-ranker breaks almost as many correct answers as it repairs.

A small consistent effect does survive — the exponent fits at `w = 0.25` on
calibration and the test sweep still peaks there (0.923 against 0.916 at
`w = 0`, falling to 0.885 at `w = 1.0`). But it is worth well under a point of
accuracy, not the three and a half points the small sample suggested.

**Verdict: within-genus re-ranking does not pay either.** Location is now
measured and rejected in both forms — as a gate, and as a constrained re-ranker.

### This is the third time in this project

A promising result at small n has now failed to replicate at larger n three
times: multi-organ fusion (round 1 → round 2), the combiner comparison (dev →
test), and this. Each had a plausible mechanism and an encouraging point
estimate. The common factor is a confidence interval that included zero and was
read optimistically because the story was good.

## What this changes

- **Phase 8 closes** as measured-and-rejected for gating. Location is not a free
  accuracy win the way the roadmap assumed.
- **Within-genus re-ranking is also rejected**, at +0.6pp with a CI containing
  zero on 1,150 observations. Location is measured and closed in both forms.
  What remains untested is unconstrained re-ranking (ruled out on product
  grounds), seasonality, and prevalence-weighting the prior.

## Scope: this result is a property of a 490-species regional catalogue

The conclusion above is stated too broadly, and iNaturalist's published numbers
show why. Their geographic prior is worth **+12pp of top-1 accuracy** — vision
alone 75%, vision + 1-degree grid 83%, vision + geomodel 87% — on a label space
of 108,124 taxa spanning the whole planet
([Introducing the iNaturalist Geomodel](https://www.inaturalist.org/blog/84677-introducing-the-inaturalist-geomodel)).
We measured +0.007 utility with an interval containing zero.

Both are correct, and the difference is not methodological:

- **Our label space is already narrow.** 490 species, all Europe/N. America.
  Location narrows it 2.7–9.5x, but genus accuracy is *already* 0.975, so there
  is almost nothing left for a prior to resolve. The AUROC gain lands where the
  operating point never looks.
- **Theirs is 220x larger and global.** At 108k taxa most of the label space is
  ruled out by geography alone, so the prior is doing enormous work before the
  photograph is consulted at all.

**The value of a geographic prior scales with label-space size and geographic
spread, and ours has neither.** So the honest statement is *"location does not
pay for a 490-species regional catalogue"* — not *"location does not pay"*. Any
expansion toward a global or much larger catalogue should re-open this, and
should expect the finding to reverse rather than replicate.

## Known limitations

- **Prevalence is not modelled.** Every species is capped at 200 sampled points,
  so the prior estimates P(location | species) and normalising across species
  imposes a uniform species prior — a globally rare plant with 20 nearby records
  outranks an abundant one with 5. A prevalence rescale was attempted and the
  fetch errored; it is untested and is the first thing to fix if this is
  revisited. Note it would likely make Test A look *worse* and the product
  better, since the evaluation set is itself prevalence-flat (~6 observations
  per species by construction).
- **34 species have no range data**, almost all *Anemone* reclassified by
  iNaturalist into other genera (*A. hepatica* → *Hepatica*). They receive a
  neutral rank rather than the floor, so they are not penalised — but the fix is
  taxon IDs rather than binomial strings.
- **Evaluation locations and the prior share a source.** Both are iNaturalist,
  so prior coverage on the evaluation set is optimistically high relative to
  deployment. 25% of in-catalogue observations already have no same-species
  record within 100 km, which is the sampling cap rather than biology and is why
  the kernel replaced a hard radius.
