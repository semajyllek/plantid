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

> **Superseded by [`INAT_FINDINGS.md`](INAT_FINDINGS.md).** This table is built
> on *synthetic* groups that assume the organ images are conditionally
> independent given species. Tested on real iNaturalist observations of the same
> individual plant, fusion is worth only **+0.001 to +0.013 AUROC**, not the
> +0.04–0.11 implied here. The conditional-independence assumption was doing
> most of the work. Guided capture remains defensible on photo-quality grounds,
> but not on this one.

## The `__OTHER__` class works, and is under-tested here

Training an explicit reject class on background species generalises to species
never seen in any role: +7pp coverage at 99% precision over a knowns-only head
(58% vs 51%). This was tested with only **14** background species. PlantNet-300K
has **619 more** species available for this role — the lever is barely exercised,
and is the most promising cheap improvement available.

## Is bark a liability? (organ ablation)

Bark is the weakest classifier (0.738 vs flower 0.910), fails cross-source
rejection, and has almost no background data available (225 images across all
non-catalog PlantNet species). So: does including it help or hurt?

Same-corpus open-set, all organ subsets, 1,460 groups:

| organs fused | AUROC | cov@99% | cov@95% |
|---|---|---|---|
| leaf | 0.861 | 9% | 44% |
| bark | 0.832 | 16% | 17% |
| flower | 0.903 | 1% | 65% |
| leaf+bark | 0.907 | 28% | 47% |
| leaf+flower | 0.919 | 28% | 70% |
| bark+flower | 0.940 | 48% | 67% |
| **leaf+bark+flower** | **0.941** | **51%** | **76%** |

**In-distribution, bark earns its place** — adding it to leaf+flower is worth
+2.2pp AUROC and +6pp coverage at 95% precision. It is not dead weight.

But this is the same-corpus setting, and bark is precisely the organ whose
cross-source rejection collapsed. So the +6pp cannot be assumed to transfer;
it is an in-distribution benefit of unvalidated durability. Keep bark as an
opportunistic input, don't build the product's rejection guarantees on it.

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

### Extending to leaf and flower — the collapse is bark-specific

Repeating the protocol per organ against style-matched foreign sources
(all catalog-overlapping classes excluded first — *Punica* from the Indian set,
*Pelargonium* / *Anemone* / *Anthurium* from Oxford):

| organ | foreign source | style-matched? | AUROC msp | FA@95 msp | AUROC Mahal. | FA@95 Mahal. |
|---|---|---|---|---|---|---|
| leaf | Indian plant leaves (542) | **NO** — studio bg | 0.979 | 8.3% | 0.981 | 0.4% |
| bark | BarkVN-50 (300) | yes | 0.702 | 99.3% | 0.805 | 74.3% |
| flower | Oxford Flowers 102 (300) | yes (close) | **0.978** | **7.3%** | 0.925 | 46.0% |

**Validity check — this matters more than the numbers.** Mean border-pixel
colour standard deviation, a proxy for "plain studio background vs cluttered
field photo":

| set | ours | foreign |
|---|---|---|
| leaf | 52.2 | **12.2** ← studio |
| bark | 47.7 | 54.0 ← matched |
| flower | 50.7 | 43.5 ← close |

The Indian leaf set is shot on plain backgrounds, so its 0.98 AUROC is largely
the model detecting *photo style*, not species novelty. **That result should be
discarded.** BarkVN-50 and Oxford Flowers are style-matched to our corpus and
are the fair tests.

So the honest per-organ picture is: **flower rejects well cross-source
(AUROC 0.978, 7.3% false-accept), bark fails badly (0.702–0.805, 74–99%),
and leaf is untested** — we do not yet have a style-matched foreign leaf source.

Note also that the best scorer flips by organ: max-softmax wins on flower,
Mahalanobis on bark. No single OOD score dominates.

### Can fused rejection be tested cross-source?

Not yet, and the tempting shortcut is invalid. Building chimeric groups
(foreign leaf + foreign bark + foreign flower from three *different* plants)
would **overstate** fusion: three organs that agree on nothing produce a flat
posterior and are trivially rejected. Real fusion faces three organs of one
real plant that may consistently resemble one catalog species.

A valid test needs a foreign source with **multiple organs of the same
individual**. The realistic option is iNaturalist observations, which bundle
several photos per observation — accepting that iNat is inside BioCLIP-2's
training data (`DATA_STRATEGY.md`), so it tests source-shift for the *head*
while being contaminated for the *encoder*, and needs organ routing since iNat
photos are not organ-tagged.

Interim inference, clearly labelled as such: fusion is dominated by its
strongest member, flower rejects well, and bark is both the weakest and the
scarcest organ. So fused cross-source rejection is plausibly much better than
the bark-only result suggests — but that is an argument, not a measurement.

### What this means

1. **Our open-set numbers are same-corpus artifacts.** The 0.943 AUROC / 76%
   coverage figures measure rejection of *held-out species photographed by the
   same community*, which is a much easier problem than rejecting a plant
   photographed by a different person with a different camera in a different
   country. The product faces the latter. Bark is the clearest demonstration;
   flower, tested the same way, holds up well.
2. **Softmax confidence is not an OOD detector.** Energy scored *below chance*
   (0.482). Only Mahalanobis — which actually models distance from the training
   distribution — showed real signal. Keep it; drop the rest.
3. **This is the generalisation gap `DATA_STRATEGY.md` predicted**, now measured
   rather than assumed, and it cost one zero-labelling download.

## ✅ Full-scale `__OTHER__` class fixes cross-source rejection

The 800-species background pool (18,166 images: leaf 8,519 / flower 9,422 /
**bark 225**) was downloaded to exercise the reject class properly. Result,
against the same foreign sources as above:

| organ | variant | in-catalog acc | → OTHER | cross-source AUROC | FA@95 |
|---|---|---|---|---|---|
| bark | no OTHER | 0.738 | — | 0.702 | 99.3% |
| bark | OTHER, unweighted | 0.699 | 11.5% | 0.953 | 31.3% |
| **bark** | **OTHER, class_weight=balanced** | **0.825** | 0.5% | **0.968** | **18.3%** |
| flower | no OTHER | 0.910 | — | 0.978 | 7.3% |
| flower | OTHER, unweighted | 0.685 | 27.0% | 0.979 | 9.3% |
| **flower** | **OTHER, balanced** | 0.898 | 0.6% | **0.981** | **5.3%** |
| leaf | no OTHER | 0.862 | — | — | — |
| leaf | OTHER, unweighted | 0.604 | 34.3% | — | — |
| **leaf** | **OTHER, balanced** | 0.856 | 0.7% | — | — |

**Bark cross-source rejection goes from unusable to good: AUROC 0.702 → 0.968,
false-accept 99.3% → 18.3% — and in-catalog accuracy *improves* 0.738 → 0.825.**

Two things this corrects:

1. **The prediction that scarce bark negatives wouldn't help was wrong.** 225
   background bark images from 24 species were enough. Quantity of negatives
   mattered far less than *having* them.
2. **Class weighting is not optional.** Unweighted, the reject class swallows
   27–34% of legitimate in-catalog leaf and flower images and costs 22–26pp of
   accuracy — a cost the AUROC column hides entirely. With
   `class_weight='balanced'` the accuracy cost falls to ≤1.2pp and rejection
   gets *better*. Any implementation must weight the classes.

### But it helps distant OOD and hurts near OOD

Same-corpus open-set (60 known / 13 held-out *catalog* species — botanically
close to the catalog, unlike foreign flora):

| organ | no OTHER | OTHER unweighted | OTHER balanced |
|---|---|---|---|
| leaf | 0.872 | 0.892 | **0.897** |
| bark | **0.865** | 0.705 | 0.715 |
| flower | 0.907 | **0.939** | 0.926 |

Bark inverts: the reject class that fixed foreign bark (0.702 → 0.968) *hurts*
on near-neighbour species (0.865 → 0.715). The 24 background bark species teach
"bark unlike my catalog", which separates Vietnamese trees easily but distorts
the boundary against congeners of catalog species.

**This suggests two mechanisms for two failure modes**, which is also how the
product should be built:

- **distant out-of-catalog** (a plant unlike anything in the catalog) →
  `__OTHER__` reject class, class-weighted
- **near out-of-catalog** (an unseen species in a catalog genus) →
  genus-level output (`HIERARCHY_FINDINGS.md`): say *"a Sedum, unsure which"*,
  correct 78–88% of the time

Genus marginalisation also improves foreign-flower rejection (AUROC 0.985,
FA 4.0%) though it hurts bark — consistent with the same split.

### Still open

- **Leaf cross-source is untested** — no style-matched foreign leaf source yet.
- **Fused cross-source is untested** — still needs a foreign multi-organ source
  (iNaturalist observations).
- The same-corpus bark regression needs a proper fix, not just noting.

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
