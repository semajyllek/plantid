# Three-way answer decision: species, genus, or decline

Replaces the single accept/reject threshold on `P(__OTHER__)`, which accepted
60% of same-genus and 39% of unrelated out-of-catalogue plants.

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

## Reproduce

```bash
PYTHONPATH=. python -m plantid.eval.rejection --emb data/processed/inat_bioclip2.npz
pytest -q                      # 63 tests
```

## Still open

- **Re-validate the chosen operating point out of sample** — the λ/μ surface was
  read off the test split.
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
