# Three-way answer decision: species, genus, or decline

Replaces the single accept/reject threshold on `P(__OTHER__)`, which accepted
60% of same-genus and 39% of unrelated out-of-catalogue plants.

Evaluated on 1,851 real iNaturalist observations (750 in-catalogue / 351
near-OOD / 750 distant-OOD), split calibration vs test **by cluster** —
in-catalogue and distant-OOD by species, near-OOD by genus, because ~6
observations share a species and the decision being fitted for near-OOD is
genus-level. Thresholds are fitted on calibration only.

## The leak was real but small

`load_background` excluded catalogue species from the `__OTHER__` pool but not
*evaluation* species: **47 of 183 near-OOD species, covering 113 of 351
observations (32%), were in the reject class's training data.** Fixed via
`exclude_names` (binomial match — iNat rows carry no `species_id`).

Measured effect, everything else held fixed:

| reject pool | distant-OOD AUROC | near-OOD AUROC |
|---|---|---|
| leaked (eval species included) | 0.928 | 0.845 |
| **fixed (eval species excluded)** | 0.931 | **0.840** |

Near-OOD falls 0.005 and distant-OOD rises 0.003 — **negligible**. The leak was
worth fixing on principle and the fix stays on by default, but it did not
inflate any published conclusion. Reported because the expectation going in was
that it would.

## The three-way rule beats the single threshold decisively

Pre-registered utilities (declared before fitting): correct species `+1.0`,
correct genus `+0.5`, any wrong answer `−4.0`, declining an out-of-catalogue
plant `+1.0`, declining an in-catalogue one `0.0`.

Fitted on calibration: `t_genus = 0.758`, `t_species = 0.884`.

| | mean utility on test | 95% CI (cluster bootstrap) |
|---|---|---|
| baseline, single threshold | +0.074 | [−0.140, +0.276] |
| **three-way rule** | **+0.657** | [+0.591, +0.724] |
| **paired gain** | **+0.583** | **[+0.384, +0.792]** ✓ |

Per-bucket outcome rates on the held-out split:

| bucket | n | species | genus | decline | answered wrong |
|---|---|---|---|---|---|
| in-catalogue | 373 | 0.123 | 0.579 | 0.298 | **0.000** |
| near-OOD | 175 | 0.046 | 0.223 | 0.731 | 0.046 |
| distant-OOD | 366 | 0.003 | 0.016 | **0.981** | 0.019 |

Against the baseline, which has no genus level and accepted 45.7% of near-OOD
and 14.8% of distant-OOD as confident species answers — all of them wrong.

## But the pre-registered utility is the wrong operating point

At λ=0.5 the rule answers at species level on only **12.3%** of in-catalogue
observations. It maximises the declared utility honestly; the declared utility
was simply a poor description of the product. Sweeping it:

| λ (genus value) | μ (wrong cost) | in-catalogue species | genus | decline | wrong | utility |
|---|---|---|---|---|---|---|
| 0.25 | 1 | **0.587** | 0.097 | 0.316 | 0.050 | +0.732 |
| 0.25 | 2 | **0.534** | 0.070 | 0.397 | 0.043 | +0.681 |
| 0.25 | 4 | **0.370** | 0.078 | 0.552 | 0.023 | +0.633 |
| 0.50 | 1 | 0.552 | 0.150 | 0.298 | 0.046 | +0.747 |
| 0.50 | 2 | 0.123 | 0.579 | 0.298 | 0.016 | +0.690 |
| 0.50 | 4 | 0.123 | 0.579 | 0.298 | 0.016 | +0.657 |
| 0.75 | 4 | **0.000** | 0.702 | 0.298 | 0.008 | +0.756 |

**λ controls species-vs-genus, and there is a sharp transition between 0.25 and
0.5** where the optimiser flips to answering genus almost everywhere. Above it
the product stops naming species; at λ=0.75, μ=4 it never does. μ moves coverage
but not this behaviour.

λ=0.25 keeps species answers as the norm: at μ=2, **53.4% species answers with
4.3% wrong**; at μ=4, 37.0% species with 2.3% wrong.

**Methodological caveat**: this sweep was computed on the test split, so
choosing λ from it spends that split. The chosen operating point should be
re-validated on data not used here before any number is quoted as a product
claim.

## Reproduce

```bash
PYTHONPATH=. python -m plantid.eval.rejection --emb data/processed/inat_bioclip2.npz
```

## Still open

- **Choose λ as a product decision** and re-validate the resulting operating
  point out of sample.
- **Calibration** (`plantid/eval/calibration.py`) is not yet built: temperature
  scaling as a fusion-mixture parameter, and conformal restricted to
  "conditional on being in the catalogue".
- **`distant_ood` is drawn from global Plantae** — mosses, ferns, tropical
  flora — and is easier than deployment OOD for a Europe/NA app, which would be
  temperate non-catalogue plants sitting between the two buckets. The 0.981
  decline rate is on the easy case. A region-restricted re-fetch is the fix.
- **Bark asymmetry**: the bark head spans 77 of 261 species, so for the rest its
  component contributes no mass to the true class and structurally deflates
  confidence. Accept rates should be stratified by whether the species has bark.
