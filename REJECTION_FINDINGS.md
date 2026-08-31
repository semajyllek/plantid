# Three-way answer decision: species, genus, or decline

Replaces the single accept/reject threshold on `P(__OTHER__)`, which accepted
60% of same-genus and 39% of unrelated out-of-catalogue plants.

Evaluated on **1,851 real iNaturalist observations** (750 in-catalogue / 351
near-OOD / 750 distant-OOD, 7,010 photos), split calibration vs test **by
cluster** — in-catalogue and distant-OOD by species, near-OOD by genus, because
~6 observations share a species and the near-OOD decision is genus-level.
Thresholds are fitted on calibration only; every CI is a cluster bootstrap.

## The rule

```
decline            if genus confidence   < t_genus      # 0.758
report genus only  if species confidence < t_species    # 0.884
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

| | mean utility (test) | 95% CI |
|---|---|---|
| baseline, single threshold | +0.418 | [+0.292, +0.540] |
| **three-way rule** | **+0.690** | [+0.640, +0.741] |
| **paired gain** | **+0.272** | **[+0.147, +0.405]** ✓ |

| bucket | n | species | genus | decline | answered wrong |
|---|---|---|---|---|---|
| in-catalogue | 373 | 0.123 | 0.579 | 0.298 | **0.000** |
| near-OOD | 175 | 0.046 | 0.223 | 0.731 | 0.046 |
| distant-OOD | 366 | 0.003 | 0.016 | **0.981** | 0.019 |

The baseline, having no genus level, answered confident species on 45.7% of
near-OOD and 14.8% of distant-OOD — all wrong. Those are now declined.

## Precision depends mostly on how often users photograph unknown plants

This is the number that matters, and quoting it without an assumed
out-of-catalogue rate is meaningless. Our eval set is **59.5% out-of-catalogue**,
far above what deployment is likely to be:

| assumed OOD rate | precision | coverage |
|---|---|---|
| 60% (this eval set) | 0.951 | 0.340 |
| 40% | 0.976 | 0.461 |
| **20%** | **0.990** | **0.582** |
| 10% | 0.996 | 0.642 |

At a plausible 20% OOD rate the app answers **58% of captures with 99%
precision**. `OPENSET_FINDINGS.md` already recorded coverage moving 76% → 30%
from a base-rate change alone, so this is reported as a curve, never a scalar.

## Choosing the utility: λ is the sensitive parameter

λ (the value of a genus answer) controls species-vs-genus, with a **sharp
transition just below 0.5**:

| λ | μ | in-cat species | genus | decline | precision @OOD20% | coverage |
|---|---|---|---|---|---|---|
| 0.25 | 1 | 0.587 | 0.097 | 0.316 | 0.948 | 0.565 |
| 0.25 | 2 | 0.534 | 0.070 | 0.397 | 0.950 | 0.498 |
| 0.25 | 4 | 0.370 | 0.078 | 0.552 | 0.960 | 0.366 |
| **0.50** | **2** | **0.123** | **0.579** | 0.298 | **0.990** | **0.582** |
| 0.75 | 4 | 0.000 | 0.702 | 0.298 | 0.998 | 0.421 |

**λ=0.5 was initially read as a failure** — answering species on only 12% of
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

## Reproduce

```bash
PYTHONPATH=. python -m plantid.eval.rejection --emb data/processed/inat_bioclip2.npz
pytest -q                      # 63 tests
```

## Still open

- **Re-validate the chosen operating point out of sample** — the λ/μ surface was
  read off the test split.
- **Test the conformal genus-containment rule** against the fixed `t_species`.
- **`distant_ood` is drawn from global Plantae** — mosses, ferns, tropical flora
  — and is easier than deployment OOD for a Europe/NA app, which would be
  temperate non-catalogue plants sitting between our two OOD buckets. The 98.1%
  decline rate is on the easy case. A `place_id`-restricted re-fetch is the fix,
  and is the single most likely way these numbers disappoint in reality.
- **Bark asymmetry**: the bark head spans 77 of 261 species, so for the other
  184 its component contributes no mass to the true class and structurally
  deflates confidence. Accept rates should be stratified by bark coverage.
- **iNat is inside BioCLIP-2's training data** (`DATA_STRATEGY.md`), so absolute
  levels throughout are optimistic.
