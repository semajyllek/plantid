# Pre-registration — does headroom govern the coarse-answer trap?

Written **before** any arm in the sweep was scored. The feasibility checks that
preceded it are declared in full below, and they touched predictors only.

## The claim under test

`narrowcast-kws` proposed that **headroom** — coarse-rank accuracy minus
fine-rank accuracy — governs whether a crowded label set inflates coverage while
quality collapses. It is recorded there and in `CLAUDE.md` as a hypothesis with
n=4 arms, one of which (`kws acoustic`) already breaks it.

## Why the existing evidence cannot settle it

In every arm published so far, coarse accuracy sits between 0.93 and 1.00 while
fine accuracy ranges 0.77–0.91. Headroom is therefore very nearly `1 − fine`
rescaled, and three rival explanations are collinear across those five arms:

1. headroom, `coarse − fine`
2. fine-task difficulty alone
3. coarse accuracy against the 0.889 break-even implied by `UTILITY`
   (`0.5·p − 4·(1−p) > 0 ⟹ p > 0.889`)

Adding more arms of the same shape would reproduce the collinearity at larger n
and confirm the hypothesis without testing it. **The design's job is to move
coarse accuracy independently of fine accuracy.**

## The lever, and a feasibility check that is declared rather than hidden

Three things were measured before writing this, all of them **predictors only** —
no coverage, no label share, no group-answer share was computed:

- Plant arms grouped by genus cannot break the collinearity. Across
  `bioclip2 / bioclip1 / mobileclip2_s0` × K ∈ {14, 30} × {varied, crowded},
  coarse accuracy stayed in [0.938, 0.998] while fine ranged 0.644–0.981.
- Re-grouping the **same** label set and the **same** fitted head moves coarse
  accuracy freely. At K=60 crowded on `mobileclip2_s0`, fine is 0.600 in every
  cell by construction, while coarse runs 0.997 (k-means, 2 groups) → 0.963
  (genus) → 0.890 (k-means, 30 groups) → 0.628 (random, 10 groups).
- `UTILITY` is identical in `plantid/eval/rejection.py` and
  `narrowcast/cascade.py` (`label_correct 1.0, group_correct 0.5, wrong −4.0,
  decline_ood 1.0, decline_in_catalog 0.0`), so the 0.889 break-even is the same
  number in both and arms from the two implementations are comparable.

The grouping is the lever because the label head never sees it. Holding the label
set, the encoder and the fitted head fixed and changing only the `group` column
holds fine accuracy **exactly** constant while sweeping coarse accuracy across
and below the break-even. That is a controlled comparison rather than a
cross-arm correlation.

## Design

**Arms.** Each arm is (label set × encoder × grouping), scored through the
declared cascade at `p_ood = 0.20`.

- *label sets*: drawn from the 490-class plant catalogue, disjoint in species
  where K permits, at K ∈ {14, 30, 60}, in `varied` (random species) and
  `crowded` (whole congener blocks) forms.
- *encoders*: the cached variants `bioclip2, bioclip1, bioclip1_cml4,
  mobileclip2_s0, bioclip_inat` — a fine-accuracy range with no new compute.
- *groupings*: `genus` (natural), `kmeans-k` for k ∈ {2, 5, 10, 20, 30}
  (coherent, tunable), `random-G` for G ∈ {5, 10} (the incoherent control, the
  generalised `kws semantic` case).
- *out-of-domain arms*: the cached text, audio and bird embeddings, scored
  through the same code so the published arms sit in the same table.

Random groups are a **mechanism control, not a product recommendation** — a
correct random-group answer is semantically vacuous. What they test is whether
the cascade retreats, which is mechanical.

**Split hygiene.** Predictors (fine, coarse, headroom) are measured on the
**calibration** rows; outcomes on the **held-out test** rows. This is what makes
the result a usable ex-ante rule — you can measure headroom on the data you fit
thresholds with, before you know what the deployment will answer.

**Primary outcome: group-answer share** — the share of in-catalogue test rows
answered at group rank. The mechanism *is* retreat to the group; coverage
inflation and label-level collapse are two downstream shadows of it. It also
needs no matched twin, so every arm is a data point.

**Secondary outcomes**: coverage, label-level share, precision, and the paired
Δs against a matched varied arm, for continuity with the published tables.

## Predictions, declared now

**P1.** Group-answer share is ~0 whenever headroom is ~0, in every domain.

**P2 (the hypothesis).** In `group_share ~ a + b·fine + c·coarse`, headroom
governing means only the *difference* matters, i.e. **b ≈ −c**. Equivalently
`M_head: group_share ~ headroom` loses little to the two-predictor `M_full`, and
beats both `M_fine` and `M_coarse`.

**P3.** The rule is a threshold as well as a slope: group-answer share stays near
zero while coarse accuracy is below 0.889 regardless of headroom, because the
declared utility makes a group answer negative-value there.

**P4 (from today's re-analysis, declared as a prediction for new arms).**
Group retreat does **not** imply label-level collapse. Decomposing the published
arms shows group answers can be drawn from two different pools:

| arm | label% | group% | decline% |
|---|---|---|---|
| text varied | 0.659 | 0.000 | 0.341 |
| text crowded | 0.212 | **0.599** | 0.189 |
| kws-acoustic varied | 0.570 | 0.000 | 0.430 |
| kws-acoustic crowded | 0.593 | **0.174** | 0.234 |

Text's group answers came out of *label* answers (0.659 → 0.212) and quality
collapsed. kws-acoustic's came out of *declines* (0.430 → 0.234) and label share
was untouched. Prediction: across the new arms, Δlabel-share is predicted by
group share **interacted with** the matched varied arm's decline rate, not by
headroom alone. This was suggested by re-analysis rather than declared in
advance of it, and is labelled accordingly.

## The null, declared so the run can fail

**Headroom adds nothing over fine accuracy alone** in predicting group-answer
share, once arms span a real range of coarse accuracy — `M_head` fails to beat
`M_fine` on grouped cross-validation. Or `M_coarse ≈ M_full`, meaning the
governing quantity is coarse accuracy against the break-even and the *difference*
is the wrong parameterisation.

Either outcome kills the headroom framing and is the reason the run is worth
doing. The result will be recorded whichever way it lands.

## Analysis plan, fixed in advance

- Compare `M_fine`, `M_coarse`, `M_head`, `M_full` by **grouped** cross-validated
  R², grouping folds by label-set draw, since arms sharing a label set are not
  independent.
- Report `b`, `c` and a bootstrap interval on `b + c` (zero ⟹ headroom
  parameterisation survives). Predictors standardised so the sum is meaningful.
- **Cluster bootstrap over arms**, keyed on the label-set draw, never over rows.
- Each arm's outcome is averaged over 5 calib/test splits before entering the
  regression, so per-arm noise does not masquerade as spread.

## Admissibility condition, declared in advance

The test is only run if **at least 20% of arms have coarse accuracy below
0.889**. Below that the sweep has not escaped the published collinearity and no
conclusion about rival predictors may be drawn from it. If the condition fails,
that fact is reported and the analysis is not run.

## Known limitations, stated in advance

Most arms are plants through one image pipeline; the out-of-domain arms are the
already-published ones re-scored, not new corpora. This tests whether headroom
governs the *cascade's behaviour*, which is a claim about the decision rule, not
a claim about what makes a domain hard. Groupings produced by k-means on the
encoder's own embeddings are favourable to group coherence by construction, and
the genus and random arms bracket them.
