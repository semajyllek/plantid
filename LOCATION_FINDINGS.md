# Location prior: real signal, but it does not pay as a gate

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

| variant | species accuracy (test) |
|---|---|
| current model (global argmax) | 0.914 |
| constrained to the predicted genus, no location | 0.906 |
| **within-genus re-rank with location** | **0.949** |

| comparison | gain | 95% CI | fixed / broke |
|---|---|---|---|
| vs current model | **+0.0349** | [−0.0053, +0.0808] | 20 / 7 |
| vs genus-constrained | +0.0429 | [+0.0056, +0.0861] ✓ | 20 / 4 |

**The honest reading is the first row: promising, not established.** The gain
against the actually-deployed model is +3.5pp with a confidence interval that
includes zero at n=373. The second row clears its interval, but it is the wrong
comparison for a shipping decision — it measures re-ranking against a
handicapped baseline rather than against what we run today.

Two things make it look real rather than noise: the fix-to-break ratio is 20:7,
and the exponent fitted on calibration landed at the test optimum rather than
somewhere the sweep punishes. One thing counts against: constraining to the
predicted genus costs −0.008 on its own (CI [−0.0216, +0.0000]), so location has
to earn that back before it profits.

**This is a power problem, not an effect problem.** The in-catalogue test split
holds 373 observations over ~65 species and only ~32 errors to fix. An expanded
in-catalogue bucket (~2,500 observations over the 215 species with range data)
is fetching; it should roughly halve the interval and settle this either way.

## What this changes

- **Phase 8 closes** as measured-and-rejected for gating. Location is not a free
  accuracy win the way the roadmap assumed.
- **Within-genus re-ranking is the live candidate**, at +3.5pp against the
  deployed model with a CI that includes zero. It preserves the property that
  motivated the gate-only decision — the genus a user sees is still decided by
  the photograph — while capturing most of Test A's headroom. Pending the
  better-powered measurement.

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
