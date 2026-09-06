# Pre-registration — does a change of acquisition source cost accuracy?

Written **before** any head in the design was fitted. The feasibility checks that
preceded it are declared in full below; they touched dataset sizes and species
counts only, never an accuracy.

## The claim under test

Domain shift is the last untested axis in this project. `CONTAMINATION_FINDINGS.md`
closed the memorisation question at under 1pp and said so explicitly:

> **Does not:** anything about *domain* shift. Same platform, same photographic
> culture, same kind of camera.

`DATA_STRATEGY.md` puts a clean-acquisition test at Tier 1 and has done since the
start. The dermatology attempt at the general version of this
(`narrowcast-derm` §11) produced a declared null on a population-shift axis, and
the obvious follow-up — splitting Fitzpatrick17k across its two source atlases —
is unrunnable because DermaAmin, which supplies 12,631 of 16,577 rows, is
suspended.

**There is a second acquisition source for plants already on disk, and it has
never been used as a control.**

## The reframing that makes this runnable

`build_heads()` (`plantid/eval/inat_fusion.py:40`) fits on `catalog_index`, which
is **Pl@ntNet-300K**, `split == 'train'`. Every headline number in this repo is
then scored on **iNaturalist** observations. So the production head has never
seen an iNaturalist photograph: *the published numbers are already a
cross-source measurement.*

What is missing is not the cross-source arm. It is the **within-source control** —
what the same head scores on held-out photographs from the source it trained on.
Without it, 0.8370 species is an absolute number with no baseline, and the cost
of the source change is unknown in either direction.

This does not contradict the standing statement that every plant number comes
from iNaturalist photographs. That is a claim about the *test* side and it stays
true.

## Why one direction is not enough

A gap between held-out Pl@ntNet and iNaturalist under a Pl@ntNet-trained head is
equally consistent with two readings:

1. **source shift** — the two corpora are different domains, and transfer costs
   accuracy in either direction;
2. **difficulty asymmetry** — iNaturalist photographs are simply harder (habitat
   context, whole-plant framing, clutter), and a head trained on *either* source
   would score lower on them.

One direction cannot separate these. Measured one-way, this is a difficulty
comparison wearing a domain-shift label. So the design runs **both directions**,
and the interesting result is which of the two shapes appears.

## Design

### Primary: a symmetric 2 × 2, closed-set

One flat multinomial head per source, fitted on that source, scored on held-out
photographs from both.

| | test: Pl@ntNet | test: iNaturalist |
|---|---|---|
| **train: Pl@ntNet** | within-source (the missing control) | cross-source (the published condition) |
| **train: iNaturalist** | cross-source | within-source |

Two diagonals, two off-diagonals. Two drops → mutual source shift. One drop →
the finding is the asymmetry, and it names which corpus is the harder one.

**Closed-set, deliberately.** No `__OTHER__` class, no thresholds, no cascade.
The only iNaturalist negatives that exist here are the `near_ood` /
`regional_ood` / `distant_ood` evaluation buckets, and training a reject class on
them leaks into the rejection numbers this repo already publishes. Photo-level
top-1 over the shared label set is sufficient to establish a source-shift effect
and sidesteps the leak entirely.

**No organ router.** The primary arm fits one head over all photographs of a
species regardless of organ. Pl@ntNet rows carry a true `organ` label and
iNaturalist photographs do not, so any design that uses the label on one side and
a router on the other hands the Pl@ntNet arm an oracle and books router error as
domain shift. Dropping organ entirely is symmetric; the routed architecture
returns in the secondary arm.

### Controls, each of which can silently manufacture the effect

1. **Shared species only.** Intersect the corpora under `curated_name`. Measured
   before writing this: 465 species are in both, 25 are Pl@ntNet-only, 0 are
   iNaturalist-only.
2. **Matched training budget, `B = 10` photographs per species per source.**
   Pl@ntNet supplies a mean of 62.6 training photographs per species and
   iNaturalist about 16 after the split below. Fitting one head on 30k rows and
   the other on 7k would confound head capacity with source. Both heads get
   exactly `B` per species, subsampled at a fixed seed.
3. **Photo-level on both sides.** The published 0.8370 is observation-fused
   through `trimmed`. Pl@ntNet has no observation grouping, so the iNaturalist
   side is scored single-photo here and no number in this document is comparable
   to a published fused one.
4. **iNaturalist splits by observation, not by photo.** Several photographs share
   one individual plant; a photo-level split would put the same plant on both
   sides.
5. **Macro-average over species.** Pl@ntNet-300K is long-tailed; a micro-average
   would let species mix carry part of the gap.
6. **Paired species-cluster bootstrap, 2,000 resamples.** Resample the shared
   species, recompute both cells, take the paired difference. Row-level
   intervals have twice produced effects in this repo that failed to replicate.

**Inclusion rule, declared now.** A species enters the analysis if, after the
observation-level iNaturalist split, it has ≥ `B` training photographs and ≥ 3
test photographs on **both** sides. At `B = 10` the feasibility count is
approximately 372 of the 465 shared species (at `B = 8`, 381; at `B = 12`, 363).
The exact retained count is reported with the result.

### Secondary: the product-realistic arm, one direction

The production organ-routed head from `build_heads` + `build_router`, scored
photo-level on Pl@ntNet test and on iNaturalist, reported two ways:

- **identification** — top-1 with `__OTHER__` excluded from the argmax; the clean
  shift quantity;
- **the three-way** — the full declared cascade at the standard `p_ood = 0.20`
  deployment weighting; the product quantity.

Reported separately on purpose. If they disagree, the story is that source shift
moves `P(__OTHER__)` rather than the ranking — worth knowing, and invisible if
only one is reported.

### Encoders

Cached and free: `bioclip2` (ViT-L, 152 MB), `bioclip1` (ViT-B),
`bioclip1_cml4` (int4, 46 MB). Being added with one embedding pass because the
prediction below is specifically about them: `mobileclip2_s2` (17.9 MB) and
`mobileclip2_s0`. `plantclef24` is **excluded** — it has no catalogue cache in
this format and would need two passes, and it is fine-tuned on 7,806 Pl@ntNet
species, which makes it the one encoder for which the two sides of this
comparison are not exchangeable.

## The prediction, stated before fitting

`EMBEDDED_FINDINGS.md` already contains a confounded version of this comparison —
Pl@ntNet catalogue K=20 versus 20 Oregon plants on iNaturalist, with the *label
set varying between arms*:

| encoder | Pl@ntNet catalogue | Oregon iNat | change |
|---|---|---|---|
| BioCLIP-2 | 0.9790 | 0.9769 | −0.2pp |
| MobileCLIP2-S2 | 0.9493 | 0.8557 | **−9.4pp** |

**Predicted:** the same ordering survives with catalogue and head held fixed —
BioCLIP-2 loses little to the source change, the small encoders lose a lot, and
the interaction (encoder × source) is the effect, not the main effect of source.
This makes the experiment a replication with the confound removed rather than a
new description.

**Predicted for the reverse direction:** no commitment. Both shapes are live and
the design exists because the answer is not known.

## Ways this could come out uninformative, declared in advance

- **Both diagonals could be low.** `B = 10` photographs per species over 372
  classes is a thin head. If within-source top-1 is poor on both sides the
  comparison is between two noise floors and nothing is established. Mitigation:
  the within-source Pl@ntNet cell is reported next to the production head's
  Pl@ntNet number, so a badly under-fitted head is visible rather than silent.
- **The gap could be dominated by organ mix.** Pl@ntNet photographs are
  organ-framed close-ups by construction; iNaturalist photographs are not. If
  the whole effect is framing convention, that is a real component of
  acquisition shift but a narrower claim than "a different camera in different
  hands", and it will be reported as the narrower claim.
- **A null is a result and will be published as one.** If both off-diagonals sit
  inside the paired interval, this repo records that its head transfers across
  acquisition sources at this encoder scale, and that the remaining Tier 1 gap is
  narrower than it looked.

## What this cannot establish

**Both corpora are inside BioCLIP-2's pretraining** — iNaturalist via GBIF,
Pl@ntNet via Meta-Album's `PLT_NET` subset (`DATA_STRATEGY.md`). This measures
**head** transfer across two sources with an encoder that is familiar with both.
It bounds head brittleness. It does not test encoder generalisation to a
genuinely novel acquisition process, and it **does not close the Tier 1 gap.**
Herbarium 2022 and self-collected field photographs remain the candidates for
that, and this document should not be cited as having retired them.
