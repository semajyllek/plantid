# Headroom governs the coarse-answer trap

**The hypothesis survives a test designed to kill it.** Across 1,409 arms
spanning four domains, *headroom* — coarse-rank accuracy minus fine-rank accuracy
— predicts retreat to the group rank with cross-validated **R² = 0.883**, against
0.362 for fine accuracy alone and 0.113 for coarse accuracy alone. The
pre-registered null was that headroom adds nothing over fine accuracy. It is
decisively rejected.

Predictions and the null declared in [`HEADROOM_PREREG.md`](HEADROOM_PREREG.md)
before any arm was scored. Reproduce with
`PYTHONPATH=. .venv/bin/python -m analysis.headroom_arms`.

## Why the previous evidence could not settle it

`narrowcast-kws` proposed headroom on five arms in which **coarse accuracy never
leaves [0.93, 1.00]**. Headroom was therefore nearly `1 − fine` rescaled, and
three rival predictors were collinear: headroom, fine difficulty alone, and
coarse accuracy against the 0.889 break-even the declared `UTILITY` implies. More
arms of that shape would have confirmed the hypothesis without testing it.

The lever that breaks the collinearity is the **grouping**, which the label head
never sees. Hold the label set, the encoder and the fitted head fixed, change
only the group assignment, and fine accuracy is pinned exactly while coarse
accuracy sweeps.

## The controlled comparison

One fitted head on 60 congeneric plant species (`mobileclip2_s0`). Fine accuracy
is **0.659 in every row** — the same head, the same predictions. Only the group
column differs:

| grouping | groups | fine | coarse | headroom | **group answers** | label answers | declines |
|---|---|---|---|---|---|---|---|
| random10 | 10 | 0.659 | 0.694 | +0.035 | **0.022** | 0.253 | 0.725 |
| kmeans50 | 50 | 0.659 | 0.703 | +0.044 | **0.035** | 0.204 | 0.761 |
| random5 | 5 | 0.659 | 0.731 | +0.072 | **0.019** | 0.221 | 0.760 |
| kmeans40 | 40 | 0.659 | 0.769 | +0.110 | **0.131** | 0.213 | 0.657 |
| kmeans30 | 30 | 0.659 | 0.789 | +0.130 | **0.157** | 0.192 | 0.651 |
| kmeans20 | 20 | 0.659 | 0.862 | +0.203 | **0.266** | 0.186 | 0.548 |
| kmeans10 | 10 | 0.659 | 0.900 | +0.241 | **0.310** | 0.206 | 0.484 |
| kmeans5 | 5 | 0.659 | 0.940 | +0.281 | **0.353** | 0.198 | 0.449 |
| genus | 8 | 0.659 | 0.951 | +0.292 | **0.393** | 0.211 | 0.396 |
| kmeans2 | 2 | 0.659 | 0.977 | +0.318 | **0.417** | 0.189 | 0.394 |

Group-answer share moves **0.022 → 0.417** and coverage **0.200 → 0.500** with
the model's ability to identify a species held completely fixed. Nothing about
the task got easier; only the coarseness of the answer the cascade was allowed to
retreat to changed.

## The pre-registered tests

**Admissibility.** The gate was ≥20% of arms below the 0.889 break-even. The
first sweep (929 arms) **failed at 18.4% and its analysis was not run** — see
Amendment 1 in the prereg, written before any outcome was inspected. The amended
sweep passes at **20.6%**, with `corr(fine, headroom) = −0.755` rather than the
near-perfect collinearity of the published five.

**P1 — zero headroom, no retreat. Holds.** Of 158 arms with |headroom| < 0.01,
mean group-answer share is **0.014** (max 0.113). With one label per group a
group answer *is* a label answer, and the cascade correctly never retreats.

**P2 — the hypothesis. Holds, with one qualification.** Grouped 5-fold CV R² on
group-answer share, folds keyed on the species-set identity:

| model | CV R² |
|---|---|
| `M_coarse` — coarse accuracy alone | +0.113 |
| `M_fine` — fine accuracy alone | +0.362 |
| **`M_head` — headroom alone** | **+0.883** |
| `M_full` — fine *and* coarse, two free parameters | +0.911 |

Headroom, a *single* number, recovers 97% of what the two-predictor model
achieves. The declared null — headroom adds nothing over fine accuracy — fails by
a factor of 2.4 in explained variance.

The qualification: standardised, `b(fine) = −0.226` and `c(coarse) = +0.189`, so
`b + c = −0.038` with a cluster-bootstrap CI of **[−0.046, −0.029]** that
excludes zero. A pure difference would give `b = −c` exactly. Fine accuracy
carries about 19% more weight than coarse, so **headroom is very nearly, but not
exactly, the governing quantity.**

This is not attenuation: `corr(headroom_calib, headroom_test) = +0.978`, so the
predictor measured on calibration is not a noisy proxy for the arm being scored.

**P3 — the break-even threshold. Partially supported.** Group retreat is
*suppressed* below the break-even, not eliminated. Restricting to coherent
groupings (`kmeans` + `genus`) so the comparison is not confounded with group
coherence, and matching on headroom in the 0.20–0.35 band:

| | n | mean headroom | mean group-answer share |
|---|---|---|---|
| coarse < 0.889 | 30 | 0.240 | **0.290** |
| coarse ≥ 0.889 | 316 | 0.275 | **0.538** |

At matched headroom, crossing the break-even roughly halves group retreat. So
coarse accuracy does carry a small independent effect, concentrated at the
threshold — which is the same asymmetry P2's `b + c ≠ 0` detected, seen from the
other side. It does not fall to zero below the line because the threshold governs
the *marginal* group answer: the cascade can still profit on the high-confidence
subset when overall coarse accuracy is below 0.889.

**P4 — retreat does not imply collapse. Holds.** Within a single fit, varying
only the grouping, group answers are drawn from **both** pools, and mostly from
declines:

| | median corr(group, decline) | median corr(group, label) |
|---|---|---|
| crowded arms (80 fits) | **−0.997** | −0.887 |
| varied arms (80 fits) | **−0.911** | −0.702 |

This is why `kws acoustic` looked like a counterexample and is not one. Its
group answers came out of declines (0.430 → 0.199), so coverage inflated while
label share held. Text's came substantially out of label answers (0.719 → 0.302),
so quality collapsed. **Same mechanism, two different shadows** — which is
exactly why coverage alone cannot be read as harm, and why the crowded-set
warning is a warning about a risk rather than a prediction of one.

## The rule generalises off the domain it was fitted on

Fitting `group_share ~ headroom` on the **plant arms only** and predicting the
nine published text, audio and bird arms:

| arm | headroom | actual | predicted |
|---|---|---|---|
| text-varied | 0.000 | 0.000 | 0.001 |
| text-crowded | 0.257 | 0.603 | 0.467 |
| kws-sem-varied | 0.000 | 0.000 | 0.001 |
| kws-sem-crowded | 0.047 | 0.031 | 0.085 |
| kws-ac-varied | 0.000 | 0.000 | 0.001 |
| kws-ac-crowded | 0.084 | 0.150 | 0.154 |
| esc50-varied | 0.000 | 0.000 | 0.001 |
| esc50-crowded | 0.092 | 0.200 | 0.167 |
| birds-crowded | 0.095 | 0.235 | 0.173 |

**MAE 0.033** on domains the rule never saw. The plant slope is +1.81 and the
audio slope, fitted independently, is +1.94.

As a usable approximation: **group-answer share ≈ 1.8 × headroom**, measurable on
calibration data before you know what the deployment will answer.

## The published contrast, for continuity

Plants, genus grouping, averaged over 160 arms — the comparison the earlier
findings docs report:

| | fine | coarse | headroom | coverage | precision | label share |
|---|---|---|---|---|---|---|
| varied | 0.930 | 0.966 | +0.036 | 0.655 | 0.951 | 0.698 |
| crowded | 0.702 | 0.982 | **+0.280** | **0.711** | 0.954 | **0.291** |

The trap, at n=160 instead of n=1: coverage up, precision flat, label share down
41pp. It strengthens with catalogue size — crowded headroom runs 0.234 at K=14 to
0.301 at K=100.

## What this changes

**The hypothesis becomes a finding, and the tool gains a predictive rule.**
narrowcast's crowded-set warning currently fires on label-set *structure*. It
could instead fire on measured headroom, which is computable from the same
calibration split that fits the thresholds, and which predicts the mechanism
directly rather than a proxy for it. **That change is not made here** — this repo
is the research record, and the tool should adopt the rule as a separate,
deliberate act.

What headroom does **not** predict is *harm*. It predicts retreat. Whether
retreat costs anything depends on which pool the group answers come from, and P4
says both happen. A report must still carry the label-level share.

## Limits

- **1,400 of 1,409 arms are plants through one image pipeline.** The nine
  out-of-domain arms are the already-published ones re-scored, not new corpora —
  they are a held-out check on a rule fitted elsewhere, not independent
  replication. Two of the four domains contribute one label set each.
- **k-means groupings are favourable to coherence by construction**, being built
  from the encoder's own geometry. `genus` (natural) and `random` (incoherent)
  bracket them, and P3's coherence-controlled cut excludes `random` entirely.
- **`near_ood` reached calibration in only 50% of plant arm splits.** Crowded
  arms take whole congener blocks, so few congeners remain outside the set and
  the bucket often has a single cluster. The prevalence mix is restricted per
  side so both calibration and test sit at p_ood = 0.20 regardless, and the
  per-arm flag is recorded in the CSV.
- Random groupings are a **mechanism control, not a recommendation**. A correct
  random-group answer is semantically vacuous; what they test is whether the
  cascade retreats, which is mechanical.
- One utility function. The 0.889 break-even is `UTILITY`'s, and P3's threshold
  would move with it. Nothing here tests a different payoff structure.

## Three defects found on the way, all in code aimed at something else

1. **Background rows collapsed the calibration split.** They carried `__OTHER__`
   as their clustering identity, so `make_splits` saw one cluster for
   `distant_ood` and put every negative in test — calibration had no distant
   negatives, and calibration is where thresholds are fitted. This affected the
   committed deployable-coverage table in `EMBEDDED_FINDINGS.md`, now corrected
   in place. `eval/rejection.py` is unaffected; its OOD rows are real
   observations carrying real binomials.
2. **The out-of-domain path trained and evaluated on the same rows**, making
   every predictor in-sample for the nine published arms. Fixed by delegating to
   narrowcast's `load_rows` / `score_frame` instead of reimplementing them.
3. **`deployment_weights` renormalises when a bucket is absent.** The text and
   audio arms have no `near_ood`, so at the plant mix they were scored at an
   effective p_ood of 0.145 rather than 0.20 — and crowded plant arms, whose
   `near_ood` often lands entirely in test, were calibrated at 0.145 and scored
   at 0.20 in exactly the arms carrying the result.

None was found by looking for it. That is now the third time a measurement aimed
at something else has caught an error here.
