# Three-way answer decision: species, genus, or decline

Replaces the single accept/reject threshold on `P(__OTHER__)`, which accepted
60% of same-genus and 39% of unrelated out-of-catalogue plants.

> **The numbers in the next three sections are superseded.** They were measured
> at `mu=2`, on a 261-species catalogue and 2,601 observations, before
> thresholds were anchored to a stated deployment prevalence. The rule, the
> nesting argument and the reasoning are unchanged; for current figures jump to
> [Current numbers](#current-numbers-497-species-5534-observations).

Evaluated on **2,601 real iNaturalist observations** (750 in-catalogue / 351
near-OOD / 750 global-OOD / 750 regional-OOD, 9,785 photos), split calibration
vs test **by cluster** — in-catalogue and both distant OOD buckets by species,
near-OOD by genus, because ~6 observations share a species and the near-OOD
decision is genus-level.
Thresholds are fitted on calibration only; every CI is a cluster bootstrap.

## The rule

```
decline            if genus confidence   < t_genus      # 0.744 -> now 0.486
report genus only  if species confidence < t_species    # 0.897 -> now 0.555
report species     otherwise
```

The three candidate scores are **nested exactly, by construction**:

```
max_c P(c)  ≤  max_g Σ_{c∈g} P(c)  ≤  Σ_{c≠OTHER} P(c)  =  1 − P(__OTHER__)
```

so the cascade is well-ordered — "species confident but genus not" is
unreachable — and they cannot be ensembled as independent evidence. There is an
invariant test for this (`tests/test_rejection.py::test_scores_are_nested`).

Fitted by maximising expected utility, declared before fitting: correct species
`+1.0`, correct genus `+0.5`, wrong `−2.0`, declining an out-of-catalogue plant
`+1.0`, declining an in-catalogue one `0.0`.

## Result

Fitted thresholds: `t_genus = 0.744`, `t_species = 0.897`.

| | mean utility (test) | 95% CI |
|---|---|---|
| baseline, single threshold | +0.396 | [+0.298, +0.489] |
| **three-way rule** | **+0.773** | [+0.734, +0.813] |
| **paired gain** | **+0.377** | **[+0.282, +0.480]** ✓ |

| bucket | n | species | genus | decline | answered wrong |
|---|---|---|---|---|---|
| in-catalogue | 373 | 0.094 | 0.643 | 0.263 | **0.000** |
| near-OOD | 175 | 0.034 | 0.251 | 0.714 | 0.034 |
| global-OOD | 366 | 0.003 | 0.016 | **0.981** | 0.019 |
| regional-OOD | 382 | 0.005 | 0.008 | **0.987** | 0.013 |

The baseline, having no genus level, answered confident species on 44.6% of
near-OOD, 15.3% of global-OOD and 21.5% of regional-OOD — all wrong. Those are
now declined.

## The regional-OOD risk did not materialise — and the reason matters

The largest risk on record for this work was that `distant_ood`, drawn at random
from global Plantae, is dominated by mosses, ferns and tropical flora a Europe/NA
app would never be shown, and that a deployment-realistic OOD set would be much
harder. A `regional_ood` bucket was fetched to test exactly that: same rule
(neither species nor genus in the catalogue) but restricted to iNat's Europe
(`97391`) and North America (`97394`) places. The distributions are genuinely
different — sampling 100 of each overlaps on 2 — with global giving *Pandanus*,
*Banksia* and Australian orchids while regional gives *Quercus robur*,
*Viola odorata*, *Castanea sativa*, *Kalmia latifolia*.

**Precision and coverage are effectively identical:**

| assumed OOD rate | global OOD | | regional OOD | |
|---|---|---|---|---|
| | precision | coverage | precision | coverage |
| 60% | 0.960 | 0.358 | 0.966 | 0.355 |
| 40% | 0.980 | 0.484 | 0.984 | 0.482 |
| **20%** | **0.992** | **0.611** | **0.993** | **0.610** |
| 10% | 0.996 | 0.674 | 0.997 | 0.674 |

But the prediction was not simply wrong — it was right about the *old* score and
wrong about the new one:

| bucket | AUROC on `1 − P(__OTHER__)` | AUROC on **genus confidence** |
|---|---|---|
| near-OOD | 0.838 | 0.815 |
| global-OOD | **0.929** | 0.980 |
| regional-OOD | **0.919** | **0.984** |

On the score the old rule used, regional OOD *is* harder (0.919 vs 0.929), which
is why the baseline accepts 21.5% of it against 15.3% of global. **Switching the
decline decision to genus confidence is precisely what neutralises that** —
on genus confidence regional is marginally *easier* (0.984 vs 0.980).

Median genus confidence tells the same story: 0.144 (regional) and 0.147
(global) against 0.856 for in-catalogue — a clean gap — whereas
`1 − P(__OTHER__)` puts them at 0.963 and 0.958 against 0.996, badly overlapped.

The likely reason the regional set is not harder: the `__OTHER__` pool is 800
PlantNet species, which are themselves temperate Europe/NA plants. The reject
class has already been trained on this distribution. That also predicts the
result would *not* hold for a region the catalogue and background pool do not
cover — a tropical deployment would need its own background pool.

> **The "800 species" is wrong** — it is the size of the file, not of the pool
> the head trains on, which is 149 leaf / 197 flower species after exclusions.
> The conclusion survives; the stated reason does not. See
> [The `__OTHER__` pool has been starved](#the-__other__-pool-has-been-starved-and-it-is-not-800-species).

## Precision depends mostly on how often users photograph unknown plants

Quoting precision without an assumed out-of-catalogue rate is meaningless: the
eval set is ~57% out-of-catalogue, far above what deployment is likely to be.
Per the table above, at a plausible 20% rate the app answers **61% of captures
with 99% precision**, and that figure is stable across the two very different
OOD distributions tested. `OPENSET_FINDINGS.md` already recorded coverage moving
76% → 30% from a base-rate change alone, so this is always reported as a curve,
never a scalar.

## What is the real out-of-catalogue rate?

Every precision and coverage figure here depends on it, and it had been a guess
(20%). It is measurable, and the measurement produces two answers seven times
apart — because it is a question about *users*, not about plants.

**iNaturalist, Europe + North America, frequency-weighted:** of 59 million
research-grade plant observations across the 6,000 most-observed species, only
**6.5% are of catalogue species — a 93.5% out-of-catalogue rate.** (Europe alone
9.4% in-catalogue; North America 4.7%. Sampling the most-observed species first
means the unsampled tail can only push this higher, so 93.5% is a floor.)

**PlantNet-300K, which is an actual plant-identification app:** **87.9% of its
images are of catalogue species — a 12.1% out-of-catalogue rate.**

### The optimistic number is partly circular, and the check says only partly

The catalogue was *selected* from PlantNet by taking species with ≥20 leaf and
flower images, so of course it covers most PlantNet images. It sits at 87.9%
against 91.6% for an optimal 261-species pick on that corpus — we chose well,
which is not the same as users behaving well.

A split-half test separates those: apply the catalogue rule to a random half of
the corpus, then measure coverage on the held-out half. Result: **88.0%
coverage, 12.0% out-of-catalogue** — the selection rule generalises within the
platform. It is still the same users on the same platform, so this is a ceiling,
not a forecast.

### Why the two numbers differ, and which to use

iNaturalist users are naturalists who deliberately seek out unusual plants; a
plant-ID app user photographs the tree outside their house. **The gap is entirely
the user population**, and the right reference class for this product is the
plant-ID app, not the biodiversity platform.

### How much the assumption matters

Fitting *and* reporting at the same assumed rate, on the held-out split:

| assumed OOD rate | precision | coverage | names species | declines |
|---|---|---|---|---|
| 10% | 0.974 | 0.870 | 0.678 | 0.057 |
| **12% — PlantNet-like** | **0.976** | **0.855** | 0.619 | 0.057 |
| 20% — current setting | 0.965 | 0.747 | 0.619 | 0.107 |
| 40% | 0.923 | 0.535 | 0.605 | 0.193 |
| 60% | 0.959 | 0.384 | 0.126 | 0.224 |
| **94% — iNaturalist-like** | 0.916 | **0.040** | 0.001 | 0.685 |

**At the naturalist rate the product does not exist**: it answers 4% of captures
and essentially never names a species. That is not a tuning problem — a
261-species catalogue is simply the wrong product for users who photograph the
unusual, and no threshold repairs it.

At the app-like rate it is a good product: 85% of captures answered at 97.6%
precision, naming a species on 62% of catalogue plants.

**Recommendation: plan at 12–20%.** The current 20% is a defensible hedge
slightly toward caution. But this is now the assumption the whole system rests
on, and it is a question about who the users are — worth settling with real
usage data before launch rather than more modelling.

## Choosing the utility: λ is the sensitive parameter

λ (the value of a genus answer) controls species-vs-genus, with a **sharp
transition just below 0.5**:

| λ | μ | in-cat species | genus | decline | precision @OOD20% | coverage |
|---|---|---|---|---|---|---|
| 0.25 | 1 | 0.587 | 0.097 | 0.316 | 0.948 | 0.565 |
| 0.25 | 2 | 0.534 | 0.070 | 0.397 | 0.950 | 0.498 |
| 0.25 | 4 | 0.370 | 0.078 | 0.552 | 0.960 | 0.366 |
| **0.50** | **2** | **0.094** | **0.643** | 0.263 | **0.992** | **0.611** |
| 0.75 | 4 | 0.000 | 0.702 | 0.298 | 0.998 | 0.421 |

(the λ=0.25 and λ=0.75 rows were measured before the regional bucket was added
and so sit on a smaller calibration set; the chosen row is current)

**λ=0.5 was initially read as a failure** — answering species on ~10% of
in-catalogue observations looked like the degenerate "always answer genus"
outcome. Measured at a realistic OOD rate it is the opposite: it is the point
that *delivers* the precision-first target, at the best coverage of any option
clearing 99%.

The reason is structural: **species accuracy (~0.88) caps what species-level
answering can ever achieve.** Genus accuracy is 0.95–0.98, so precision above
~88% requires answering at genus or declining. λ=0.25 buys 53% species answers
at 95% precision; there is no setting that gives both.

**Chosen: λ=0.5, μ=2.** Consistent with both stated product decisions — genus as
a real answer, precision first.

*Caveat*: this surface was computed on the test split, so choosing from it
spends that split. The operating point should be re-validated on fresh
observations before being quoted as a product claim.

## The data leak was real but small

`load_background` excluded catalogue species from the `__OTHER__` pool but not
*evaluation* species: **47 of 183 near-OOD species, covering 113 of 351
observations (32%), were in the reject class's training data** — the head was
fitted on species rejection was then measured on. Fixed via `exclude_names`
(binomial match; iNat rows carry no `species_id`), on by default in
`build_heads`.

| reject pool | distant-OOD AUROC | near-OOD AUROC |
|---|---|---|
| leaked | 0.928 | 0.845 |
| **fixed** | 0.931 | **0.840** |

Near-OOD falls 0.005, distant-OOD rises 0.003 — **negligible**. Worth fixing on
principle; it inflated nothing. Recorded because the expectation was that it
would.

## Temperature scaling: fixes the number, not the decision

The heads are **under-confident** (T < 1 everywhere), largely because
`class_weight="balanced"` and a large `__OTHER__` class pull mass away from the
predicted species. Fitted on the catalogue's *val* split, so the iNat
calibration split is not spent:

| organ | fitted T | mean confidence before → after | ECE before → after |
|---|---|---|---|
| leaf | 0.705 | 0.657 → 0.791 | 0.117 → **0.028** |
| bark | 0.486 | 0.456 → 0.821 | 0.370 → **0.052** |
| flower | 0.734 | 0.730 → 0.839 | 0.096 → **0.026** |

Bark was badly miscalibrated (ECE 0.37) — expected, given it spans only 77 of
261 species.

Effect on the decision: thresholds move (0.758 → 0.935, 0.884 → 0.979) but the
outcome barely does — **paired utility gain +0.015, 95% CI [−0.012, +0.041],
not significant**.

Note this *could* have mattered: T is applied per organ head *before* the router
mixes them, and a mixture of tempered distributions is not a monotone transform
of the tempered mixture, so unlike temperature on a single classifier it can
change rankings. It also changes which photo `combiners.trimmed` drops. It was
measured rather than assumed — and turned out not to move the decision.

**Keep it for the displayed confidence**, which is what users see and act on;
do not expect it to improve accuracy.

## Conformal prediction, scoped deliberately

Implemented (`eval/calibration.py`) but **restricted to "conditional on the
plant being in the catalogue"**. The coverage guarantee is over the true label,
so it says nothing about an out-of-catalogue plant whose label is not in the
label space, and using an empty prediction set as a decline signal reduces to
thresholding max-softmax — the weakest of the three nested scores.

What survives is the useful part: **if the prediction set falls entirely inside
one genus, answer at genus level** (`genus_containment`). That is an adaptive
alternative to a fixed `t_species`, and it is the form worth testing next.

Exchangeability is only approximate here (~6 correlated observations per
species), so `conformal_threshold` samples one observation per cluster by
default.

## Current numbers: 497 species, 5,534 observations

The evaluation set had a hole in it. 247 of the catalogue's 497 species had **no
real observation at all** and were scored only on the PlantNet test split — the
same corpus their head was fitted on. Two fetches closed most of it: a targeted
per-species query (161 species), then a `taxon_id` query for species whose
catalogue name is a superseded synonym (54 more, see
[`INAT_FINDINGS.md`](INAT_FINDINGS.md)). Coverage went **250 → 411 → 465 of
497**. The same run grew near-OOD from 64 to 120 catalogue genera.

| bucket | observations | photos | distinct |
|---|---|---|---|
| in-catalogue | 3,435 | 12,888 | 465 species |
| near-OOD | 618 | 2,058 | 120 genera |
| global-OOD | 737 | 2,793 | — |
| regional-OOD | 744 | 2,755 | — |
| **total** | **5,534** | **20,494** | |

Fitted thresholds: `t_genus = 0.486`, `t_species = 0.555`, at `mu=4` and an
assumed 20% out-of-catalogue rate.

| | mean utility (test) | 95% CI |
|---|---|---|
| baseline, single threshold | −0.586 | [−0.755, −0.428] |
| **three-way rule** | **+0.595** | [+0.521, +0.657] |
| **paired gain** | **+1.181** | **[+1.047, +1.330]** ✓ |

| bucket | n | species | genus | decline | answered wrong |
|---|---|---|---|---|---|
| in-catalogue | 1,734 | 0.521 | 0.349 | 0.130 | 0.018 |
| near-OOD | 299 | 0.197 | 0.187 | 0.615 | **0.221** |
| global-OOD | 364 | 0.019 | 0.011 | 0.970 | 0.030 |
| regional-OOD | 373 | 0.013 | 0.005 | 0.981 | 0.019 |

| assumed OOD rate | global precision | coverage | regional precision | coverage |
|---|---|---|---|---|
| 60% | 0.858 | 0.434 | 0.867 | 0.430 |
| 40% | 0.919 | 0.580 | 0.924 | 0.576 |
| **20%** | **0.955** | **0.725** | **0.957** | **0.723** |
| 10% | 0.968 | 0.798 | 0.969 | 0.797 |

**Every headline number here is lower than the last two rounds, and that is the
point.** Precision at 20% has moved 0.963 → 0.962 → 0.957 and coverage 0.762 →
0.744 → 0.723 across three successive expansions of the evaluation set. Nothing
about the model changed. What changed is that the set stopped being drawn from
the easy end of the catalogue.

### Two different mechanisms make species harder

Three cohorts, split by how each species entered the evaluation set:

| cohort | obs | species | species accuracy | genus accuracy |
|---|---|---|---|---|
| **broad** — surfaced under untargeted queries | 2,371 | 250 | 0.873 [0.841, 0.901] | 0.978 |
| **targeted** — appeared only when asked for | 913 | 161 | 0.786 [0.733, 0.836] | 0.966 |
| **recovered** — needed a `taxon_id`, name was stale | 151 | 54 | 0.748 [0.651, 0.840] | 0.967 |
| combined | 3,435 | 465 | 0.844 [0.818, 0.868] | 0.974 |

Two-sample cluster bootstrap on the gaps:

| | species | genus |
|---|---|---|
| broad − targeted | **+0.086 [+0.030, +0.145]** ✓ | +0.012 [−0.008, +0.033] |
| broad − recovered | **+0.123 [+0.027, +0.223]** ✓ | +0.011 [−0.018, +0.043] |
| recovered − targeted | −0.040 [−0.147, +0.065] | +0.001 [−0.035, +0.033] |

`broad` and `targeted` differ by how often the plant is photographed — that is
the sampling-bias correction, and it is the explanation for that pair. It is
**not** the explanation for `recovered`, which needs a separate one.

**The prediction that `recovered` would be easy was wrong, and the reason is
worth more than the prediction was.** These are not rare plants — *Anemone
nemorosa* has 84,623 research-grade observations and was invisible only because
the catalogue spells it with a superseded name. Yet it scores like the rarity
cohort. The explanation is that renamed genera are *large* ones (note the two
mechanisms are near-independent here: `broad` and `targeted` have almost the
same congener counts, so congeners cannot explain their gap either):

| cohort | mean catalogue congeners | median |
|---|---|---|
| broad | 8.2 | 3 |
| targeted | 7.5 | 3 |
| **recovered** | **13.4** | **10** |

The recovered cohort is 46 *Anemone*, 19 *Sedum*, 15 *Papaver* — exactly the
genera that have been split, because being large is what gets a genus split.
Pooling all cohorts and binning by how many congeners a species has in the
catalogue shows the mechanism directly, and it inverts between levels:

| catalogue congeners | obs | species accuracy | genus accuracy |
|---|---|---|---|
| 0 | 737 | **0.961** | 0.940 |
| 1–2 | 642 | 0.807 | 0.961 |
| 3–5 | 549 | 0.800 | 0.985 |
| 6+ | 1,507 | 0.820 | **0.992** |

A species alone in its genus is easy to name and *harder* to place at genus
level (0.940). That is structural, not noise: `gmat` gives such a genus a
one-column block, so its genus score **is** its species score — verified, they
are equal on 698 of the 737 zero-congener observations (94.7%), and `genus_ok`
matches `species_ok` on 98%. **For a species alone in its genus the genus
fallback provides no lift whatsoever**; there is no congener to absorb a near
miss. A species in a crowded genus is hard to name (0.82) and almost impossible
to misplace (0.992).
**Species and genus difficulty are anti-correlated**, which is precisely the
structure the genus fallback monetises, and it is why genus accuracy has stayed
at 0.97 through every expansion while species accuracy fell 0.874 → 0.844.

The rule responds to this without being retuned. On the recovered cohort it
answers **genus 50.7% of the time against species 36.2%**, inverting its usual
split:

| cohort | n (test) | species | genus | decline |
|---|---|---|---|---|
| broad | 1,164 | 0.543 | 0.343 | 0.114 |
| targeted | 501 | 0.493 | 0.341 | 0.166 |
| recovered | 69 | **0.362** | **0.507** | 0.130 |

### Near-OOD doubled: it helped, and it is still the weak link

The fetch grew near-OOD from 290 observations over 64 genera to 618 over 120.
Because `SPLIT_CLUSTER` clusters this bucket by genus and the bootstrap
resamples genera, *genera* — not rows — are what set its precision, so this is
the growth that should matter. Measured on both rounds with a genus-clustered
bootstrap:

| | test n | test genera | utility | 95% CI | width |
|---|---|---|---|---|---|
| before | 144 | 32 | −0.448 | [−1.029, +0.111] | 1.140 |
| **after** | 299 | 60 | **−0.186** | [−0.685, +0.252] | **0.937** |

**Doubling the genera worked in both directions** — utility improved by +0.26
and the interval narrowed 18%. It is still ~0.94 wide and still spans zero, so
the bucket remains the loosest thing in this document, but the earlier reading
that growth "did not help" was wrong.

The uncomfortable part is the ceiling. Near-OOD genera are drawn from the
catalogue's own genera, of which there are only **172, and 120 are already
covered**. Fetching harder buys at most another 1.4x, well short of what would
tighten this interval to the width of the other buckets'. Near-OOD answers wrong
22.1% of the time — the worst rate in the table by an order of magnitude — and
carries 32% of the OOD mass in every precision figure above via
`OOD_MIX_REGIONAL`. **This bucket, not the in-catalogue one, is now the limiting
factor on what can be claimed, and more data will not fix it.**

### The `__OTHER__` pool has been starved, and it is not 800 species

This document has said the reject pool is "800 PlantNet species". That is the
size of the *file*. What the head actually trains on, after `load_background`
drops catalogue species by `species_id` and evaluation species by binomial:

| organ | in pool | after catalogue exclusion | after eval exclusion | rows kept |
|---|---|---|---|---|
| leaf | 533 | 175 | **149** | 1,340 of 8,519 (16%) |
| flower | 589 | 236 | **197** | 1,884 of 9,422 (20%) |
| bark | 23 | 13 | 13 | 120 of 225 |

**Almost all of the loss is the catalogue exclusion, not the eval one.** The
background pool was built against the 261-species catalogue; the catalogue then
grew to 530, so two-thirds of the pool became species we now want to
*recognise*. `load_background`'s docstring anticipates this — "the pool was
built against an earlier, smaller catalogue" — but the size of the effect was
never measured, and the 800-species figure was left standing in the argument for
why regional OOD is not harder than global OOD.

That argument may still hold: 149–197 temperate Europe/NA species is a thinner
reject class but not a differently-distributed one, and regional OOD still
declines at 98.1% against global's 97.0%. But **the stated reason is wrong by a
factor of four**, and the pool should be rebuilt against the 530-species
catalogue before the claim is quoted again. Growing near-OOD in this round cost
a further 10 leaf / 13 flower species, so this shrinks every time the evaluation
set grows.

## Reproduce

```bash
PYTHONPATH=. python -m plantid.eval.rejection --emb data/processed/inat_bioclip2.npz
pytest -q                      # 63 tests
```

## Still open

- ~~Re-validate the chosen operating point out of sample~~ — **done**: precision
  0.960 on 161 species held out of the threshold fit entirely, against 0.963
  in-sample. The λ/μ *surface* is still test-split-derived; the chosen point
  is now validated.
- ~~Grow near-OOD~~ — **doubled** (64 → 120 genera); utility −0.448 → −0.186 and
  the CI narrowed 1.140 → 0.937. It helped, and it is still the weakest link:
  the interval spans zero and only 172 catalogue genera exist, so fetching
  cannot close it. Fixing near-OOD needs a modelling change, not more data.
- **Rebuild the background pool against the 530-species catalogue.** The reject
  class trains on 149 leaf / 197 flower species, down from 533 / 589, because
  the pool predates the catalogue expansion. This is the largest unaddressed
  defect in the rejection path and it worsens every time the eval set grows.
- **Modernise the catalogue's names.** 47 of 497 (9.5%) are superseded. The
  recovery fetch works around this per-observation via `name_map`; the
  catalogue itself is still wrong, and 0.34% of OOD rows are catalogue species
  in disguise.
- **32 species remain unevaluated** — 9 hybrid placeholders that are not
  species, 7 cultivated-only, 6 contested *Ophrys* microspecies, the rest
  unresolved.
- **Test the conformal genus-containment rule** against the fixed `t_species`.
- ~~`distant_ood` is easier than deployment OOD~~ — **tested and resolved**: a
  region-restricted bucket declines at 98.7%, versus 98.1% global. The concern
  was valid for the old score and is neutralised by the genus-confidence rule.
  It would return for a region the catalogue and `__OTHER__` pool do not cover.
- **Bark asymmetry**: the bark head spans 77 of 261 species, so for the other
  184 its component contributes no mass to the true class and structurally
  deflates confidence. Accept rates should be stratified by bark coverage.
- **iNat is inside BioCLIP-2's training data** (`DATA_STRATEGY.md`), so absolute
  levels throughout are optimistic.
