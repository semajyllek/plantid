# A change of acquisition source is free at ViT-L and costs 18pp at 17.9 MB

Pre-registered in `DOMAIN_SHIFT_PREREG.md`, written before any head in the
design was fitted.

Domain shift was the last untested axis in this project. It is now tested along
one axis — **acquisition source**, Pl@ntNet-300K against iNaturalist — and the
answer is an interaction, not a main effect. The source change costs BioCLIP-2
nothing and costs every smaller encoder between 10 and 18 percentage points.

## The reframing that made it runnable

`build_heads()` (`plantid/eval/inat_fusion.py:40`) fits on `catalog_index`, which
is **Pl@ntNet-300K**, `split == 'train'`. Every headline number in this repo is
then scored on **iNaturalist** observations. The production head has never seen
an iNaturalist photograph.

**So the published figures were already the cross-source arm.** What was missing
was the *within-source control* — what the same head scores on held-out
photographs from the corpus it trained on. Without it, 0.8370 species was an
absolute number with no baseline and the cost of the source change was unknown
in either direction. It is a second acquisition source that has been on disk
since the start, and it was never used as one.

This does not contradict the standing statement that every plant number comes
from iNaturalist photographs. That is a claim about the test side and it stays
true.

## The primary result: a symmetric 2 × 2

One flat multinomial head per source, closed-set, organ-free, fitted at a matched
budget of **10 photographs per species per source** so head capacity cannot
confound the comparison; 359 shared species surviving the declared inclusion
rule; 4,934 Pl@ntNet and 4,750 iNaturalist test photographs; photo-level top-1,
macro-averaged over species; paired species-cluster bootstrap, 2,000 resamples.

| encoder | pn→pn | **pn→inat** | inat→inat | inat→pn | **deployment shift** | reverse shift |
|---|---|---|---|---|---|---|
| `bioclip2` (ViT-L) | 0.7701 | 0.7691 | 0.8615 | 0.7145 | **−0.001** [−0.029, +0.027] | −0.147 [−0.174, −0.119] |
| `bioclip2_cml4` (int4) | 0.7622 | 0.7599 | 0.8571 | 0.7119 | **−0.002** [−0.029, +0.026] | −0.145 [−0.172, −0.118] |
| `plantclef24` (43 MB) | 0.7678 | 0.6815 | 0.8414 | 0.6815 | **−0.086** [−0.119, −0.055] † | −0.160 [−0.187, −0.132] |
| `bioclip1` (ViT-B) | 0.7009 | 0.6012 | 0.7006 | 0.6013 | **−0.100** [−0.129, −0.072] | −0.099 [−0.127, −0.071] |
| `bioclip1_cml4` (46 MB) | 0.6836 | 0.5839 | 0.6837 | 0.5813 | **−0.100** [−0.129, −0.070] | −0.102 [−0.133, −0.072] |
| `bioclip_inat` | 0.6841 | 0.5787 | 0.6552 | 0.5685 | **−0.106** [−0.134, −0.079] | −0.086 [−0.116, −0.058] |
| `mobileclip2_s0` | 0.5130 | 0.3499 | 0.4281 | 0.3394 | **−0.163** [−0.194, −0.133] | −0.089 [−0.117, −0.060] |
| `mobileclip2_s2` (17.9 MB) | 0.5827 | 0.4045 | 0.5069 | 0.4123 | **−0.179** [−0.208, −0.148] | −0.095 [−0.125, −0.064] |

† `plantclef24` is fine-tuned on 7,806 Pl@ntNet species, so **whichever cell is
scored on Pl@ntNet is plausibly inflated** at the image level in a way no other
encoder's is. That cuts both ways and must be read per column: it inflates
`pn→pn`, so **−0.086 is an upper bound** on the deployment penalty; it also
inflates `inat→pn`, so **−0.160 is a lower bound** on the reverse penalty. See
below.

*Two coincidences in this table are coincidences, checked.* `pn→inat` and
`inat→pn` agree to three decimals for `bioclip1`, `bioclip1_cml4` and
`plantclef24`, which looks like an aliasing bug. It is not: the two cells are
computed from different heads on different test sets (4,934 Pl@ntNet against
4,750 iNaturalist photographs), and their per-species vectors differ on 338 of
359 species with a mean absolute difference of 0.22 and a correlation of 0.48.
Equal macro means, different measurements.

**The deployment direction — train on Pl@ntNet, test on iNaturalist — is the
column that matters, because it is the one this repo ships.** BioCLIP-2 pays
nothing for the source change, at full precision and at int4 alike. Every
smaller encoder pays, and the size of the payment tracks encoder capability.

### The prediction was declared and it held

`EMBEDDED_FINDINGS.md` contained a confounded version of this comparison — the
label set varied between arms — reporting BioCLIP-2 −0.2pp and MobileCLIP2-S2
−9.4pp. The pre-registration predicted the same ordering would survive with
catalogue and head held fixed. It did, and the small-encoder penalty is **larger**
once the confound is removed: −17.9pp rather than −9.4pp.

## The asymmetry, which was not predicted

The reverse direction was left open in the pre-registration because both shapes
were live. Three different shapes appeared, and they sort by encoder.

- **BioCLIP-2 is asymmetric.** iNaturalist is the *easier* corpus for it
  (0.8615 within-source against 0.7701), a Pl@ntNet head transfers to it for
  free, and an iNaturalist head does not transfer back (−0.147). Fitting on the
  easier corpus buys accuracy that does not survive contact with the other one.
- **BioCLIP v1 is symmetric.** 0.7009 and 0.7006 within-source, −0.100 and
  −0.099 across. Two corpora of equal difficulty, mutually shifted: source shift
  in its plain form.
- **MobileCLIP2-S2 inverts the difficulty ordering.** Pl@ntNet is the easier
  corpus for it (0.5827 against 0.5069).

So **"iNaturalist photographs are harder" is not a property of the photographs.**
It is a property of the encoder reading them. `EMBEDDED_FINDINGS.md` explained
its gap by the nature of the imagery — "iNaturalist observations are what people
actually take — variable framing, habitat context, whole-plant shots" — and that
explanation is right for the weak encoders and backwards for the strong one. The
finding it supports is unchanged; the mechanism it offered generalises less far
than it reads.

The practical form: **train the head on the harder corpus.** Pl@ntNet's
organ-framed close-ups look like the less realistic data and produce the more
transferable head. "Harder" here does not mean "contains bark" — the
organ-matched check below holds it after reweighting, and the iNaturalist head
loses on every organ rather than on one.

## The shipped head says the same thing

Secondary arm: the production organ-routed head from `build_heads` +
`build_router`, at its full training size (~62 photographs per species, not 10),
scored photo-level over all 465 shared species, `__OTHER__` masked out of the
argmax.

| encoder | Pl@ntNet species | iNat species | shift | Pl@ntNet genus | iNat genus | genus shift |
|---|---|---|---|---|---|---|
| `bioclip2` | 0.7633 | 0.7601 | **−0.003** [−0.029, +0.020] | 0.9487 | 0.9221 | −0.027 [−0.042, −0.014] |
| `plantclef24` | **0.7757** | 0.6748 | **−0.101** [−0.128, −0.075] | 0.9601 | 0.8792 | −0.081 [−0.099, −0.065] |
| `bioclip1_cml4` | 0.7104 | 0.6133 | **−0.097** [−0.119, −0.074] | 0.9042 | 0.8023 | −0.102 [−0.120, −0.084] |
| `mobileclip2_s2` | 0.6453 | 0.4844 | **−0.161** [−0.187, −0.137] | 0.8551 | 0.6698 | −0.185 [−0.209, −0.163] |

Within a point of the primary arm on every encoder, which also answers the
pre-registered worry that a 10-photograph head might be too thin to compare:
`pn→pn` at B=10 is 0.7701 against the full head's 0.7633, so the budget was not
the story.

BioCLIP-2's one measurable cost is at **genus**, −2.7pp, small and real.

**The source change barely touches the rejection channel.** Mean `P(__OTHER__)`
roughly doubles on iNaturalist and stays negligible in absolute terms — 0.0036 →
0.0070 for BioCLIP-2, 0.0032 → 0.0059 for S2. Source shift moves the ranking, not
the accept/reject decision, so the three-way cascade inherits the identification
shift and adds nothing of its own.

> **Departure from the pre-registration, recorded rather than dropped.** The
> prereg asked for the full three-way cascade at `p_ood = 0.20` on both sides.
> It cannot be run symmetrically: the out-of-catalogue buckets are
> iNaturalist-only, and the one Pl@ntNet-side negative pool on disk is what
> `__OTHER__` was fitted on, so using it as test negatives would be in-sample.
> Masked-argmax identification plus the `__OTHER__` mass is what can be measured
> leak-free, and it is reported above.

## The 43 MB middle ground pays, and it is the clearest case in the table

`plantclef24` was excluded from the pre-registered encoder set and then measured
anyway, because it is the middle of the open size decision and reading a number
off the trend between 46 MB and 152 MB would have been guesswork. It needed
three fresh embedding passes at 518px; they are now on disk.

It pays **−0.086 [−0.119, −0.055]** on the deployment direction, −0.069 after
organ matching, and −0.101 with the shipped head. Every measurement excludes
zero. It sits with the ViT-Bs, not with BioCLIP-2.

**And it is the one encoder where the mechanism is visible directly.** With the
shipped head it is the *best* encoder measured on Pl@ntNet — 0.7757 species,
ahead of BioCLIP-2's 0.7633 — and 8.5pp behind BioCLIP-2 on iNaturalist
(0.6748 against 0.7601). Its entire deficit is a cross-source deficit. That is
what fine-tuning on 7,806 Pl@ntNet species buys and what it costs: within-source
strength that does not travel.

Which is also why these are **bounds rather than point estimates**, and why the
direction has to be read per column. The fine-tuning inflates whichever cell is
scored on Pl@ntNet:

- it inflates `pn→pn`, so the deployment shift is more negative than the truth —
  the true penalty lies in `[0, 0.086]`;
- it inflates `inat→pn`, so the reverse shift is *less* negative than the truth —
  the true reverse penalty is **at least** 0.160.

That matters for the "train on the harder corpus" reading above, where
`plantclef24` is the strongest-looking case: the bias runs in the direction that
strengthens it, not the one that manufactures it.

Its `__OTHER__` mass also moves the most of any encoder — +0.0145, against
BioCLIP-2's +0.0035 and S2's +0.0027 — though all three are small in absolute
terms.

> **This does not contradict `EMBEDDED_FINDINGS.md`**, where `plantclef24` was
> 1.5pp behind BioCLIP-2 on iNaturalist photographs. That comparison was 20
> Oregon species; this one is 465. Both test on iNaturalist, so the difference
> between −1.5pp and −8.5pp is not about source — it is that `plantclef24`'s
> deficit to BioCLIP-2 **grows with label-space size**. Worth knowing before the
> catalogue grows, and a separate question from the one this document tests.

## Robustness: it is not the organ mix, and the mix is not what it looks like

The pre-registration named organ mix as a confound that could *manufacture* an
effect. It can also mask one, in the direction that flatters the headline — if
the within-source control contains hard photo types the cross-source arm does
not, the measured shift is deflated and "free" would be two offsetting biases
rather than a null. So the mix was measured rather than assumed, with the
production router on the iNaturalist test photographs (`analysis/domain_shift_organ.py`):

| | leaf | bark | flower |
|---|---|---|---|
| Pl@ntNet test, true labels | 0.458 | **0.036** | 0.506 |
| iNaturalist test, routed | 0.362 | **0.119** | 0.519 |

The expected imbalance is not there, and the residual one runs the other way:
this is a herbaceous regional catalogue, so bark is scarce in *Pl@ntNet* — 3.6%
of its test photographs — while iNaturalist observations route to bark three
times as often. Reweighting the within-source control to the iNaturalist mix
therefore makes it **harder**, not easier.

| encoder | deployment shift, pooled | deployment shift, organ-matched |
|---|---|---|
| `bioclip2` | −0.001 [−0.029, +0.027] | **+0.017 [−0.010, +0.045]** |
| `plantclef24` | −0.086 [−0.119, −0.055] | −0.069 [−0.101, −0.036] |
| `bioclip1` | −0.100 [−0.129, −0.072] | −0.065 [−0.094, −0.036] |
| `mobileclip2_s2` | −0.179 [−0.208, −0.148] | −0.149 [−0.179, −0.119] |

The null survives matching and the ordering is unchanged. Organ mix accounts for
roughly a third of the small encoders' penalty and none of BioCLIP-2's.

**The asymmetry is not a missing bark channel either.** BioCLIP-2's reverse shift
is −0.145 against the organ-matched control, against −0.147 pooled, and the
iNaturalist-trained head loses on *every* organ of the Pl@ntNet test set rather
than collapsing on one:

| | leaf | bark | flower |
|---|---|---|---|
| `pn→pn` | 0.692 | 0.714 | 0.790 |
| `inat→pn` | 0.652 | 0.681 | 0.767 |

## Robustness: the asymmetry is not near-duplicate leakage

iNaturalist observations are distinct plants by construction, but two
observations of one population on one day by one photographer are not an
independent draw, and that would inflate `inat→inat` and so inflate the reverse
shift. Re-splitting the iNaturalist side by **1-degree geographic cell**, so
train and test observations of a species come from different places:

| encoder | split by observation (205 spp) | split by geographic cell (205 spp) |
|---|---|---|
| `bioclip2` reverse shift | −0.148 | −0.135 |
| `bioclip1` reverse shift | −0.108 | −0.100 |
| `mobileclip2_s2` reverse shift | −0.081 | −0.075 |

Both columns are on the same 205 species — the subset carrying coordinates —
because the geographic split changes which species qualify, and comparing it
against the full 359-species arm would confound the split rule with species
composition. It does not: the two columns agree throughout. **Population
leakage does not explain the asymmetry.**

On that easier 205-species subset BioCLIP-2's deployment shift is +0.038
[+0.007, +0.068] rather than −0.001 — a small *gain* from the source change. The
honest reading across both subsets is that at ViT-L scale the deployment
direction costs nothing; whether it pays a little depends on which species you
ask about.

## `bioclip_inat` does not buy source robustness

`bioclip_inat` is a ViT-B pretrained on iNaturalist alone, eliminated earlier for
being worse on both corpora. It is the natural test of whether matching the
encoder's pretraining source to the deployment source protects against a source
change. It does not: its deployment shift is −0.106, no better than plain
BioCLIP v1's −0.100, and it is the one encoder whose within-source Pl@ntNet
score (0.6841) exceeds its within-source iNaturalist score (0.6552) despite
having been trained on iNaturalist. **Breadth of pretraining, not source match,
is what transfers** — the same conclusion the encoder bakeoff reached from a
different direction.

## What this does not establish

**Both corpora are inside BioCLIP-2's pretraining** — iNaturalist via GBIF,
Pl@ntNet via Meta-Album's `PLT_NET` subset (`DATA_STRATEGY.md`). This measures
**head** transfer across two acquisition sources with an encoder familiar with
both. It bounds head brittleness and it says the shipped configuration is not
resting on a source match it never had. It does **not** test encoder
generalisation to a genuinely novel acquisition process, and it **does not close
the Tier 1 gap in `DATA_STRATEGY.md`.** Herbarium 2022 and self-collected field
photographs remain the candidates for that, and this document should not be cited
as having retired them.

Two narrower limits worth naming. Pl@ntNet photographs are organ-framed
close-ups by construction, so part of what is measured here is framing
convention rather than camera and hand — a real component of acquisition shift,
but narrower than the thing the Tier 1 entry asks for. And both corpora are
citizen-science apps, so the photographic culture is shared even where the
platform is not.

## For the size decision

The three-way choice in `CLAUDE.md` now has a source-shift column, and it does
not split the way byte order does:

| build | size | deployment shift |
|---|---|---|
| BioCLIP-2 int4 | 152–160 MB | **−0.001** [−0.029, +0.027] |
| `plantclef24` | 43 MB | −0.086 [−0.119, −0.055], upper bound |
| MobileCLIP2-S2 | 17.9 MB | −0.179 [−0.208, −0.148] |

This does not decide the question — it is one axis among accuracy, latency,
hazard safety and app size, and it is still a product judgement. What it removes
is the option of treating the 43 MB build as a small accuracy concession. On
this catalogue, in the direction that ships, it is not.

## For the tool

Two things narrowcast should carry:

1. **A card built on one source overstates what a user gets, and by how much
   depends on the encoder.** At ViT-L the overstatement is zero; at 17.9 MB it is
   16–18 percentage points. Reporting encoder scale next to accuracy is not
   pedantry.
2. **The small-encoder fragility result now has three independent
   measurements** — 9.4pp to a source change with the label set varying
   (`EMBEDDED_FINDINGS.md`), 2.3pp to an upload-date change
   (`CONTAMINATION_FINDINGS.md`), and 16–18pp here with catalogue and head held
   fixed. A small encoder's headline accuracy is not merely lower, it is *less
   durable*, and nothing in a single-source evaluation reveals that.
3. **An encoder fine-tuned on the evaluation corpus scores like the best model
   in the table and transfers like a mid-sized one.** `plantclef24` is ahead of
   BioCLIP-2 on Pl@ntNet and 8.5pp behind it on iNaturalist. A card that names
   the encoder's training corpus alongside the evaluation corpus would make that
   legible; one that reports accuracy alone cannot.

## Reproduce

```
PYTHONPATH=. .venv/bin/python -m analysis.domain_shift \
    --variants bioclip2 bioclip1 bioclip1_cml4 bioclip2_cml4 \
               mobileclip2_s2 mobileclip2_s0 bioclip_inat plantclef24
PYTHONPATH=. .venv/bin/python -m analysis.domain_shift --geo-only  --variants bioclip2 bioclip1 mobileclip2_s2
PYTHONPATH=. .venv/bin/python -m analysis.domain_shift --inat-split cell --variants bioclip2 bioclip1 mobileclip2_s2
PYTHONPATH=. .venv/bin/python -m analysis.domain_shift_product \
    --variants bioclip2 bioclip1_cml4 mobileclip2_s2
PYTHONPATH=. .venv/bin/python -m analysis.domain_shift_organ \
    --variants bioclip2 bioclip1 mobileclip2_s2
```

Two encoders needed fresh caches; everything else was already on disk. From
`.venv-mps`:

```python
from plantid.features import embed_catalog, embed_inat, embed_background
for v in ("mobileclip2_s2", "mobileclip2_s0"):
    embed_inat.main(v)                       # catalogue and background existed
for f in (embed_catalog.main, embed_inat.main, embed_background.main):
    f("plantclef24")                         # 82k images at 518px, ~65 min on MPS
```
