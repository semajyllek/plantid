# Open-set rejection: can the app be trusted to say "I don't know"?

Scope changed from ~1000 species to "a couple hundred, but trustworthy". That
makes **open-set rejection the core technical problem, not species count** — a
200-species catalog means essentially every plant on Earth is out-of-domain, so
the model must reliably decline rather than confidently guess.

All experiments below run on cached BioCLIP-2 embeddings (`BAKEOFF_FINDINGS.md`)
— seconds of CPU, no GPU, no new data.

> **Contamination caveat (added later).** BioCLIP-2's training corpus
> (TreeOfLife-200M) draws on GBIF and a PlantNet-derived Meta-Album subset, so
> the absolute numbers in this document may be optimistic — treat them as upper
> bounds pending an out-of-distribution test set. The ranking versus the
> fine-tuned CNN is unlikely to be affected. See [`DATA_STRATEGY.md`](DATA_STRATEGY.md).

## Protocol

Split the 87 species into disjoint roles, so "unknown" species are never seen in
any form during training:

- **60 known** — the catalog. Head is trained on their train split.
- **14 background** — pooled into a single `__OTHER__` class during training.
- **13 truly unseen** — never seen in any role. The actual test of rejection.

Metric is the product metric, not accuracy: rank all queries by confidence,
accept the top fraction, and measure **precision among accepted, where accepting
an out-of-catalog plant is always an error** (as is misclassifying a known one).

## Result

Fused across leaf + bark + flower, 1,460 test groups (1,200 known-species,
260 truly-unseen):

| scoring | AUROC | coverage @99% precision | @95% | @90% |
|---|---|---|---|---|
| FUSED softmax (knowns-only head) | 0.943 | 51% | 76% | 87% |
| FUSED kNN-cosine OOD score | 0.922 | 32% | 55% | 79% |
| FUSED softmax × kNN | 0.949 | 56% | 76% | 87% |
| **FUSED + explicit `__OTHER__` class** | 0.947 | **58%** | **76%** | 85% |

On same-corpus held-out species this looks like a viable product: at 95%
precision the app answers 76% of captures and declines the rest. **But see the
cross-source test below — this does not hold when the data comes from a
different source, and these figures should not be quoted as product claims.**

## Multi-organ capture is what makes rejection work

Single-organ rejection is materially worse than fused — this is the strongest
quantitative argument for the guided-capture UI:

| input | AUROC (known vs unseen) |
|---|---|
| leaf alone | 0.832 |
| bark alone | 0.783 |
| flower alone | 0.874 |
| **all three fused** | **0.943** |

Three views of the same plant give three chances to notice that nothing in the
catalog matches. Asking the user for specific organs isn't just a data-quality
measure — it is the mechanism that makes "I don't know" reliable.

## The `__OTHER__` class works, and is under-tested here

Training an explicit reject class on background species generalises to species
never seen in any role: +7pp coverage at 99% precision over a knowns-only head
(58% vs 51%). This was tested with only **14** background species. PlantNet-300K
has **619 more** species available for this role — the lever is barely exercised,
and is the most promising cheap improvement available.

## Caveat: precision depends strongly on the out-of-catalog base rate

An earlier run of this experiment used 27 unseen species (31% of queries
out-of-catalog) rather than 13 (18%), and coverage at 95% precision fell from
76% to ~30%. Most of that difference is the base rate, not the method.

**So the headline numbers are conditional on how often users photograph
out-of-catalog plants.** This argues directly for choosing the catalog by *what
people actually photograph* rather than by what is easiest to source, and for
measuring the real-world unknown rate early. A catalog that covers the common
cases well is worth more than a larger one that doesn't.

## ⚠️ The rejection result does not survive a change of data source

Everything above uses held-out species from **the same corpus** — same
contributors, cameras, framing conventions, and regions. Testing against a
genuinely foreign source changes the answer completely.

**Test**: 300 images from BarkVN-50 (Vietnamese tree bark, research-collected,
not citizen-science). Every image is out-of-catalog by construction, so no
labels are needed. Scored against our 87-species bark head on BioCLIP-2
embeddings, versus our own 183 in-catalog bark test images.

| OOD score | AUROC | false-accept @95% TPR |
|---|---|---|
| max softmax | 0.702 | 99.3% |
| max logit | 0.619 | 99.0% |
| energy (logsumexp) | 0.482 | 99.0% |
| kNN cosine (top-1) | 0.699 | 97.0% |
| kNN cosine (mean@5) | 0.666 | 98.3% |
| **Mahalanobis** | **0.805** | **74.3%** |

At a threshold that accepts 95% of genuine in-catalog bark, the best detector
still wrongly accepts **74%** of Vietnamese tree bark; most scorers accept
essentially all of it. Compare AUROC 0.783 for bark alone on same-corpus
held-out species — and 0.943 fused.

**Confound ruled out:** only 1 of BarkVN-50's 50 genera (*Acacia*) appears in
our catalog, and at different species, so at most ~2% of these images could be
legitimately accepted.

**Caveats:** this is bark alone — the weakest organ for rejection — and fusion
could not be tested because BarkVN-50 contains only bark. n is modest (300 vs
183). But note the direction: tropical bark from a different continent and a
different collection method *should be easy* to reject. It isn't.

### What this means

1. **Our open-set numbers are same-corpus artifacts.** The 0.943 AUROC / 76%
   coverage figures measure rejection of *held-out species photographed by the
   same community*, which is a much easier problem than rejecting a plant
   photographed by a different person with a different camera in a different
   country. The product faces the latter.
2. **Softmax confidence is not an OOD detector.** Energy scored *below chance*
   (0.482). Only Mahalanobis — which actually models distance from the training
   distribution — showed real signal. Keep it; drop the rest.
3. **This is the generalisation gap `DATA_STRATEGY.md` predicted**, now measured
   rather than assumed, and it cost one zero-labelling download.

## Recommended target: ~250 species, from data already in hand

PlantNet-300K supports this without any new corpus:

| threshold | catalog species | images (cap 60) |
|---|---|---|
| leaf+flower ≥20 | **261** | ~27k |
| leaf+flower ≥30 | 217 | ~24k |
| leaf+flower ≥50 | 162 | ~19k |

Plus **619 remaining species (~11k images at cap 20) as the `__OTHER__` pool**.
The same dataset supplies both the catalog and the negatives — no iNaturalist
download required to build and validate the whole product.

**iNat is therefore deferred, not cancelled.** Its unique value is real
multi-photo observations of the same individual plant, which would retire the
synthetic-groups caveat (`CNN_FINDINGS.md`). That matters for final validation,
not for building.

## Next actions

1. **Build the 261-species catalog + 619-species `__OTHER__` pool** and download
   (~38k images total; existing threaded downloader).
2. **Re-run this experiment at full scale.** The `__OTHER__` class with 619
   background species instead of 14 is the single most promising lever.
3. **Fix the harness top-k truncation bug** (`BAKEOFF_FINDINGS.md`) before
   quoting any top-5 numbers.
4. **Calibrate**: temperature scaling, then conformal prediction for a coverage
   guarantee, so the confidence shown in the UI means something.
5. Only then: on-device distillation (BioCLIP-2 → MobileCLIP2-S0, ~19pp gap).
