# Three-way answer decision: species, genus, or decline

Replaces the single accept/reject threshold on `P(__OTHER__)`, which accepted
60% of same-genus and 39% of unrelated out-of-catalogue plants.

> **The numbers in the next three sections are superseded.** They were measured
> at `mu=2`, on a 261-species catalogue and 2,601 observations, before
> thresholds were anchored to a stated deployment prevalence. The rule, the
> nesting argument and the reasoning are unchanged; for current figures jump to
> [Current numbers](#current-numbers-497-species-5055-observations).

Evaluated on **2,601 real iNaturalist observations** (750 in-catalogue / 351
near-OOD / 750 global-OOD / 750 regional-OOD, 9,785 photos), split calibration
vs test **by cluster** — in-catalogue and both distant OOD buckets by species,
near-OOD by genus, because ~6 observations share a species and the near-OOD
decision is genus-level.
Thresholds are fitted on calibration only; every CI is a cluster bootstrap.

## The rule

```
decline            if genus confidence   < t_genus      # 0.744
report genus only  if species confidence < t_species    # 0.897
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

## Current numbers: 497 species, 5,055 observations

The evaluation set had a hole in it. 247 of the catalogue's 497 species had **no
real observation at all** and were scored only on the PlantNet test split —
the same corpus their head was fitted on. A targeted per-species fetch
(`inat_eval.py --species-file`) closed 161 of those 247. The set is now:

| bucket | observations | photos | distinct |
|---|---|---|---|
| in-catalogue | 3,284 | 12,451 | 411 species |
| near-OOD | 290 | 1,092 | 64 genera |
| global-OOD | 737 | 2,793 | — |
| regional-OOD | 744 | 2,755 | — |
| **total** | **5,055** | **19,091** | |

Fitted thresholds: `t_genus = 0.478`, `t_species = 0.545`, at `mu=4` and an
assumed 20% out-of-catalogue rate.

| | mean utility (test) | 95% CI |
|---|---|---|
| baseline, single threshold | −0.525 | [−0.703, −0.356] |
| **three-way rule** | **+0.673** | [+0.614, +0.726] |
| **paired gain** | **+1.198** | **[+1.051, +1.362]** ✓ |

| bucket | n | species | genus | decline | answered wrong |
|---|---|---|---|---|---|
| in-catalogue | 1,644 | 0.566 | 0.324 | 0.110 | 0.010 |
| near-OOD | **144** | 0.243 | 0.215 | 0.542 | **0.271** |
| global-OOD | 364 | 0.016 | 0.014 | 0.970 | 0.030 |
| regional-OOD | 381 | 0.008 | 0.010 | 0.982 | 0.018 |

| assumed OOD rate | global precision | coverage | regional precision | coverage |
|---|---|---|---|---|
| 60% | 0.850 | 0.456 | 0.859 | 0.451 |
| 40% | 0.918 | 0.601 | 0.923 | 0.598 |
| **20%** | **0.960** | **0.745** | **0.962** | **0.744** |
| 10% | 0.976 | 0.818 | 0.976 | 0.817 |

**Near-OOD is on 144 observations** and is the one bucket the top-up did not
grow, while carrying 32% of the OOD mass in every precision figure above. Its
0.271 answered-wrong is the worst number in the table and it is also the least
solid; do not read it as comparable to the 1,644-row in-catalogue line.

### Adding 919 harder observations barely moved the operating point

Replaying the *previous* manifest through *today's* code separates what this
data changed from what `mu=4` and prevalence anchoring changed two commits ago:

| | species | t_genus / t_species | in-cat species acc | genus acc | precision @20% | coverage |
|---|---|---|---|---|---|---|
| before top-up | 250 | 0.474 / 0.495 | 0.874 | 0.980 | 0.963 | 0.762 |
| **after top-up** | 411 | 0.478 / 0.545 | **0.850** | 0.975 | 0.962 | 0.744 |

The aggregate operating point is essentially unchanged — which is
`deployment_weights` working as designed, since it anchors to a stated
prevalence rather than to the evaluation set's composition. **The movement is
not in the aggregate; it is in the cohort split.**

### The new species are harder, and that is the finding

The 250 species that turned up under broad queries versus the 161 that only
appeared when asked for by name — that ordering ranks species by how heavily
photographed they are, which is exactly the sampling bias this document has been
listing as its dominant caveat. Correcting part of it should cost accuracy:

| cohort | obs | species | species accuracy | genus accuracy |
|---|---|---|---|---|
| broad-query ("established") | 2,371 | 250 | 0.872 [0.841, 0.900] | 0.978 [0.966, 0.987] |
| targeted ("new") | 913 | 161 | **0.791** [0.738, 0.841] | 0.967 [0.948, 0.984] |
| combined | 3,284 | 411 | 0.850 [0.823, 0.875] | 0.975 [0.965, 0.984] |

Two-sample cluster bootstrap on the gap:

- species: **+0.081, 95% CI [+0.025, +0.141]** ✓ excludes zero
- genus: +0.011, 95% CI [−0.008, +0.032] — includes zero

**Species accuracy is significantly worse on the under-photographed cohort;
genus accuracy is not.** That is the strongest evidence yet for the genus-first
design: the level the product leans on is the level that survives the harder
sample. The rule responds correctly without being retuned — on the new cohort it
answers species 49.0% of the time against 59.5%, and declines 14.5% against 9.7%.

### The operating point transfers to species it was never fitted on

The top open item in this document was that the λ/μ surface had been read off
the test split. The 161 newly-covered species are fresh: refitting on a
calibration set with **every new-cohort observation removed** gives
`t_genus = 0.479`, `t_species = 0.547` (against 0.478 / 0.545 with them), and
applying those thresholds to the new cohort's test rows:

| cohort | n | utility | precision @20% OOD | coverage |
|---|---|---|---|---|
| established (in-sample species) | 1,195 | +0.698 [+0.644, +0.751] | 0.963 | 0.754 |
| **new (out-of-sample species)** | 449 | +0.616 [+0.528, +0.694] | **0.960** | 0.716 |

**Precision holds at 0.960 on species absent from the fit, with coverage falling
0.754 → 0.716** — the rule declines more on harder plants, which is what it is
supposed to do. Scoped claim: this validates the operating point against
*unseen species*, not the threshold-fitting procedure in general. `t_genus` is
largely set by the OOD buckets, which were identical in both fits.

### Catalogue synonyms mislabel a small slice of the OOD buckets

The catalogue carries PlantNet's pre-split names (`Anemone nemorosa`,
`Sedum kamtschaticum`), so an observation filed under iNat's current name
(`Anemonoides nemorosa`, `Phedimus kamtschaticus`) escapes the binomial match
that assigns buckets — and lands in an OOD bucket while being a plant the model
is supposed to name. Checked rather than assumed:

- **6 of 1,771 OOD rows (0.34%)** are catalogue species under a modern name.
- **39 of 1,481 distant/regional rows (2.63%)** sit in a genus the catalogue
  covers, so they are really near-OOD.

Both push the reported numbers **pessimistic**, and re-running with corrected
buckets confirms the direction: precision 0.962 → 0.967, utility +0.673 →
+0.686, global-OOD decline 0.970 → 0.992.

**Take the direction, not the magnitudes.** Bucket membership feeds
`make_splits`, so relabelling 39 rows also moved the calibration/test split
(calib 2,484 vs 2,522) — `distant_ood`'s test n went *up*, 364 → 368, while 15
rows left the bucket. The 0.970 → 0.992 jump is far too large for 15 rows and is
mostly a different test split. The sign is trustworthy; the size is not.

Small either way, but it grows with the catalogue — see
[`INAT_FINDINGS.md`](INAT_FINDINGS.md) for the name-resolution pass and what it
says about the 86 species still unevaluated.

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
- **Grow near-OOD.** 144 test observations carrying 32% of the OOD mass, with
  the worst answered-wrong rate in the table, is the weakest link in every
  precision figure quoted here.
- **Modernise the catalogue's names.** 0.34% of OOD rows are catalogue species
  in disguise; the fix is a one-time resolution pass, not a threshold change.
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
