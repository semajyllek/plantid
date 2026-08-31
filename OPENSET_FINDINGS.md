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

**This is a viable product.** At 95% precision the app answers 76% of captures
and declines the rest; at 99% precision it still answers ~58%.

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
