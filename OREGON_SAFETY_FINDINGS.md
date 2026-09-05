# Can it tell hemlock from wild carrot?

**Verdict: BioCLIP-2 yes, but only if it is allowed to abstain. MobileCLIP2-S2
no, at any operating point.**

The pairs were declared in [`SAFETY_PREREG.md`](SAFETY_PREREG.md) before any
image was fetched. Read that first — the point of this test is that a
general-purpose identifier's mean accuracy says nothing about the specific
confusions that hurt people, so the confusions were named in advance.

## Data

27 Oregon species, research-grade iNaturalist observations, 6,023 images over
4,057 observations (`analysis/safety_fetch.py`). Split by **observation**, never
by image. Frozen encoders, logistic head, `C=10`.

27-way top-1: **BioCLIP-2 0.9743**, **MobileCLIP2-S2 0.8184**.

> 340 duplicate rows were removed before fitting. `order_by=votes` paginates
> unstably when vote counts tie, so 170 observations were fetched twice. No
> leakage — the split is by `obs_id` — but they concentrated in *Sambucus
> racemosa*, which is a reported pair, and would have been double-weighted.

## The pre-registered pairs

Fatal-direction rate P(predict B | truth A) as coverage on A is reduced. BioCLIP-2:

| truth | mistaken for | 100% | 75% | 50% | 25% | sep |
|---|---|---|---|---|---|---|
| Conium maculatum | Daucus carota | 0.008 | 0.000 | 0.000 | 0.000 | 0.946 |
| Conium maculatum | Lomatium nudicaule | **0.000** | 0.000 | 0.000 | 0.000 | 0.948 |
| Conium maculatum | Lomatium utriculatum | **0.000** | 0.000 | 0.000 | 0.000 | 0.941 |
| Conium maculatum | Anthriscus caucalis | 0.025 | 0.008 | 0.000 | 0.000 | 0.954 |
| Conium maculatum | Osmorhiza berteroi | **0.000** | 0.000 | 0.000 | 0.000 | 0.958 |
| Cicuta douglasii | Heracleum maximum | **0.000** | 0.000 | 0.000 | 0.000 | 0.984 |
| Cicuta douglasii | Lomatium dissectum | **0.000** | 0.000 | 0.000 | 0.000 | 0.966 |
| Sambucus racemosa | Sambucus cerulea | 0.037 | 0.024 | 0.012 | 0.000 | 0.958 |
| Toxicodendron diversilobum | Rubus ursinus | 0.000 | 0.000 | 0.000 | 0.000 | 0.978 |
| Rubus armeniacus | Rubus ursinus | 0.000 | 0.000 | 0.000 | 0.000 | 0.983 |
| Veratrum viride | Veratrum californicum | 0.032 | 0.000 | 0.000 | 0.000 | 0.962 |

The Pacific-Northwest-specific hazard — a forager digging what they believe is
*Lomatium* root — is **0.000 in every direction tested**, at full coverage.

## The pair framing was not enough, and this is the main finding

Every cell above looks acceptable. They are also all individually small *because
the errors are spread across many different edible species*. The number that
matters is not any one pair but their union: **how often is a lethal plant given
the name of something a person would eat?**

Hemlock named as any not-dangerous species, BioCLIP-2:

| | 100% coverage | 75% | 50% |
|---|---|---|---|
| *Conium maculatum* | **0.067** [0.017, 0.128] | 0.008 [0.000, 0.026] | 0.000 |
| *Cicuta douglasii* | 0.010 [0.000, 0.037] | 0.000 | 0.000 |

At full coverage the true fatal rate for poison hemlock is **6.7%** — more than
six times the 1% bar declared in advance — while no single pre-registered pair
exceeds 2.5%. Reporting per-pair rates alone would have passed a model that
fails.

`SAFETY_PREREG.md` says "never average across pairs", and that stands: averaging
hides the worst case. But the union is not an average, and it was missing.
**Both are required: pairs individually, and the union over everything edible.**

Abstention fixes it. At 75% coverage the rate falls to 0.8% [0.000, 0.026] and at
50% it is zero. Refusing a quarter of hemlock photographs is an entirely
acceptable product behaviour — "not sure, do not eat this" is the right answer.

## MobileCLIP2-S2 fails, and mean accuracy hides how badly

| hemlock named as something edible | 100% | 75% | 50% |
|---|---|---|---|
| BioCLIP-2 | 0.067 | 0.008 | 0.000 |
| MobileCLIP2-S2 | **0.283** | **0.167** | **0.050** |

The 27-way top-1 gap is 15.6pp (0.974 vs 0.818). The fatal-rate gap is **4× at
full coverage and 20× at 75%**. S2 is still at 5% fatal after declining half its
inputs.

Under the declared cascade S2 copes by declining **72.5%** of hemlock against
BioCLIP-2's 11.7% — safe, but a model that refuses three quarters of the photos
you most need it for is not usable for this.

**This is the tool's thesis, on the case where it costs most: the headline metric
understates safety-critical degradation.** `EMBEDDED_FINDINGS.md` put S2 within
1.6–4.8pp of BioCLIP-2 on catalogue top-1 and that remains true; it is simply not
the relevant number here.

> **Confound, stated plainly.** BioCLIP-2 is trained on iNaturalist via GBIF and
> these are iNaturalist images. MobileCLIP2 is a general web-image model with far
> less biological exposure. Some unknown part of this gap is that asymmetry
> rather than capacity. It does not change the verdict for S2 — 28.3% fails on
> its own terms — but it inflates BioCLIP-2's apparent margin.

## What the pre-registration missed

***Foeniculum vulgare*** (fennel) appears as a hemlock confusion under **both**
encoders — 0.008 for BioCLIP-2, 0.042 for S2. It is foraged, it is a feathery
umbel, and confusing it with poison hemlock is a documented real-world fatality
mode. It was not in the declared pair list.

Adding it now is **exploratory, not confirmatory**. It is recorded here so the
next pre-registration includes it, not to be quoted as a tested result.

## Contamination ceiling

iNaturalist is inside BioCLIP-2's training data (`DATA_STRATEGY.md`), so every
number here is an **upper bound**. Failure here is decisive; success here is not.
Only photographs taken outside that corpus — Tier 1 in `DATA_STRATEGY.md`, i.e.
your own camera in Oregon — can settle it.

## What this means

1. **A consequence-framed product is viable, on BioCLIP-2, with abstention.** The
   152 MB build clears the declared bar at 75% coverage on both lethal species.
2. **It is not viable on a small encoder.** The size question and the safety
   question have opposite answers, and safety wins where they conflict.
3. **Any card for a catalogue containing dangerous species must report the union
   rate**, not just per-pair confusions. This is a change to `plantid/tool/card.py`,
   not a documentation note.
4. Reproduce with:
   `PYTHONPATH=. .venv/bin/python analysis/safety_pairs.py bioclip2 mobileclip2_s2`
