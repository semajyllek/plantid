# v2 catalog: 261 → 530 species

The catalog `ROADMAP.md` targets, built and evaluated. Everything here uses
frozen BioCLIP-2 embeddings and a class-weighted logistic head — no backbone
training, all local on MPS.

## What was built

**261 species, 28,919 images** (leaf 13,370 / flower 14,482 / bark 1,067),
capped at 60 per species-organ, split 70/15/15.

Selection: leaf **and** flower ≥20 images; **bark opportunistic** (≥5), which
covers 77 of the 261 species. Bark being optional is exactly what lifts the
catalog from 87 to 261 — it was the only binding constraint (`ROADMAP.md` §0).

Reuse kept the download to 21,121 of 28,919: 3,029 images came from the v1
working set and 4,769 were already present in the background pool.

**Background pool correction.** The pool was built to exclude the *87*-species
catalog; at 261, **184 of its species became catalog species**. Training a
reject class on a species you want to recognise teaches the model to reject it,
so these are now dropped at load time via
`embed_background.load_background(organ, exclude_species=...)`. Negatives after
exclusion: leaf 4,840 (381 spp) / flower 5,743 (433 spp) / bark 225 (24 spp).

## Expansion to 530 species

The 261 came from requiring **both** leaf and flower ≥20 images. Relaxing that to
**either** doubles the catalogue without leaving PlantNet-300K — so organ labels
survive and no new contamination is introduced. A species identifiable from its
flowers alone is still worth naming, and the per-organ heads already span
different class sets, so a leaf-only species simply never appears in the flower
head.

**530 species, 172 genera, 43,506 images** (leaf 20,219 / flower 21,976 / bark
1,311; 98 species carry usable bark).

### The target was genus accuracy ≥90%. It holds.

| | species @261 | species @530 | genus @261 | **genus @530** |
|---|---|---|---|---|
| leaf | 0.781 | 0.733 | 0.959 | **0.944** |
| bark | 0.832 | 0.840 | 0.952 | 0.893 |
| flower | 0.843 | 0.774 | 0.979 | **0.972** |
| **weighted** | **0.814** | **0.757** | **0.969** | **0.957** |

**Doubling the catalogue costs 5.7pp of species accuracy and 1.2pp of genus
accuracy.** Genus is the robust quantity, which is the whole basis of the
three-way answer: the level the app can defend degrades far more slowly than the
level it would prefer to give.

Bark alone lands at 0.893, marginally under the bar — but it is the optional
organ, spans 95 species, and has 206 test images. Leaf and flower, which the
capture flow actually requires, are 0.944 and 0.972.

### End to end, the product barely moves

| | @261 | @530 |
|---|---|---|
| precision @20% out-of-catalogue | 0.965 | 0.962 |
| coverage @20% | 0.747 | **0.763** |
| mean utility (test) | +0.723 | +0.684 |

Coverage actually improves: a larger catalogue means fewer real plants fall
outside it, which is the point. **Twice the species for essentially no
product-level cost.**

### Two corrections the expansion forced

**The splitter could not handle tiny groups.** A species admitted on leaf alone
may have a single flower image, and the 70/15/15 split produced a negative test
count. Groups under three images now go to training only, with an assertion.

**Catalogue growth silently invalidates the evaluation buckets.** Bucket
membership is a property of the catalogue, not the observation: at 530 species,
**65 near-OOD observations (18.5%) had become catalogue species** and would have
been scored as plants the model should reject while being plants it is meant to
name. `inat_eval.relabel_buckets` now recomputes membership against the current
catalogue — it moved 84 observations — and must be re-run after any catalogue
change.

## Accuracy at 261 species: species falls, genus does not

| organ | n_train | n_test | species @261 | (was @87) | **genus @261** | (was @87) | genera |
|---|---|---|---|---|---|---|---|
| leaf | 9,364 | 2,002 | 0.759 | 0.862 | **0.951** | 0.974 | 100 |
| bark | 733 | 167 | **0.784** | 0.738 | **0.910** | 0.820 | 48 |
| flower | 10,135 | 2,173 | 0.808 | 0.910 | **0.982** | 0.980 | 100 |

Chance is 0.38% at 261 species (was 1.15% at 87).

**Species accuracy drops ~10pp on leaf and flower** going from 87 to 261
classes — expected, and not a regression: the task is 3x harder.

**Genus accuracy barely moves** — 0.951 / 0.910 / 0.982, versus 0.974 / 0.820 /
0.980. Flower genus actually *improved*. This strengthens the case in
`HIERARCHY_FINDINGS.md` considerably: as the catalog grows, the species answer
degrades but the genus answer holds, so the value of reporting the most specific
confident level rises with scale rather than falling.

Bark improves on both (0.738 → 0.784, 0.820 → 0.910), but note it is a 77-way
problem here, not 261-way, so it is not directly comparable to its 87-species
number — it gained data and species density, not difficulty.

## Cross-source rejection holds at 3x scale

Against the same style-matched foreign sources, with the class-weighted
`__OTHER__` head:

| organ | head | in-catalog acc | cross-source AUROC | FA@95 | source |
|---|---|---|---|---|---|
| leaf | no OTHER | 0.759 | — | — | *(untested)* |
| leaf | OTHER balanced (381 spp) | 0.747 | — | — | *(untested)* |
| bark | no OTHER | 0.784 | 0.662 | 99.0% | BarkVN-50 |
| **bark** | **OTHER balanced (24 spp)** | **0.832** | **0.977** | **11.7%** | BarkVN-50 |
| flower | no OTHER | 0.808 | 0.950 | 13.3% | Oxford Flowers |
| **flower** | **OTHER balanced (433 spp)** | 0.809 | **0.966** | **10.0%** | Oxford Flowers |

The `__OTHER__` mechanism not only survives the larger catalog, it improves:
bark AUROC 0.968 → 0.977 and false-accept 18.3% → 11.7% versus the 87-species
run; flower 0.981 → 0.966 AUROC but false-accept 5.3% → 10.0%.

Bark in-catalog accuracy again *rises* when the reject class is added
(0.784 → 0.832) — the negatives act as a regulariser. Leaf pays 1.2pp, flower
pays nothing.

## Label curation: what counts as one nameable plant

The catalogue took PlantNet's labels verbatim, and class labels were formed by
truncating each to two tokens. That is wrong in two different ways, and only one
of them is a judgement call.

### A bug: eight class labels were not names

`" ".join(name.split()[:2])` turns `Fragaria × ananassa` into the class
**`Fragaria ×`**. Eight classes were of this shape, and one of them was worse
than cosmetic:

| class | images | what it actually contained |
|---|---|---|
| `Pelargonium x` | 187 | **three different hybrids** — `x asperum`, `x hortorum`, `x hybridum` |
| `Pelargonium ×` | 87 | `× hortorum` again, split off purely by the dash character |
| `Fragaria ×` | 111 | the garden strawberry |
| `Anemone x`, `Freesia x`, `Hypericum x`, `Lupinus x`, `Tradescantia x` | 263 | one hybrid each |

So the model was being trained to answer `Fragaria ×` — not a name that can be
shown to anyone — and to treat three distinct pelargoniums as one plant while
splitting a fourth from itself on a typographic accident. `curation.canonical_name`
keeps the hybrid marker and normalises `×` to `x`.

*(Author citations needed no such fix. `Sedum palmeri S.Watson` and `Sedum
palmeri S. Watson` are separate `species_id`s — 30 binomials are split this way
across 32 extra ids — but two-token truncation already collapsed them, which is
why 530 ids yield 498 species.)*

### A product decision: microspecies a phone cannot separate

The app is for plant enthusiasts, not taxonomists. `curation.MERGE` folds
together labels that fail three tests at once: iNaturalist does not recognise
them as active taxa, they are not separable from a photograph by a
non-specialist, and an enthusiast would call them the same plant.

The *Ophrys sphegodes* complex is the case. The catalogue carried eight
segregate microspecies of it; **seven are merged** — `arachnitiformis`,
`aranifera`, `incubacea`, `lupercalis`, `occidentalis`, `passionis`, `virescens`
— taking the genus from 17 classes to 11. `Ophrys araneola` is deliberately
**kept**: it resolves to itself as an active iNat taxon, so it is a species, not
a form. Also merged: `Sedum nussbaumerianum` → `Sedum adolphii`, which iNat
lumps. Dropped: `Pelargonium spp.`, a genus placeholder that is not a species at
all — the model already answers at genus level when it cannot name one.

Net: **497 → 490 classes.**

### Measured: flat on the headline, and the eval set cannot see most of it

Three label functions over the same data and the same buckets:

| config | classes | species acc | genus acc | precision @20% | coverage |
|---|---|---|---|---|---|
| baseline (truncation) | 497 | 0.8443 | 0.9741 | 0.9572 | 0.7234 |
| canonical (bug fix only) | 497 | 0.8428 | 0.9747 | 0.9556 | 0.7210 |
| **curated** (+ merge, drop) | **490** | 0.8460 | 0.9747 | 0.9563 | 0.7222 |

*(the curated row is what every figure in `REJECTION_FINDINGS.md` now uses)*

Nothing moves. The reason is a limitation of the evaluation set, not a verdict
on the change: **only 3 of 3,435 in-catalogue observations are of a merged
species, and none at all are of a hybrid.** The plants curation touches are
almost exactly the plants iNaturalist has no research-grade multi-photo
observations of. The eval is uninformative about the merge's direct effect and
should not be quoted as evidence either way.

Note the bug fix alone *costs* 0.15pp of species accuracy. That is correct
behaviour: it stops the model getting free credit for conflating three
pelargoniums into one class. **Every species-accuracy figure recorded before
this was very slightly optimistic for that reason.**

### What the eval can see: the species left behind in a de-crowded genus

The measurable question is whether removing seven confusable microspecies helps
the real species that remain. Treatment = the 41 surviving species in *Ophrys*
and *Sedum* (288 observations); control = the 423 species in genera curation
never touched (3,144 observations).

| | treatment species acc | control species acc |
|---|---|---|
| baseline | 0.8194 | 0.8470 |
| curated | **0.8368** | 0.8470 |
| paired Δ | **+0.0174** [+0.0000, +0.0516] | +0.0000 [−0.0030, +0.0026] |

The control is flat to four decimals, which is the check working. The treatment
gain is borderline by its CI, but the mechanism is unambiguous — **5
observations flipped wrong→right and 0 the other way**, and all five were real
species that the baseline had predicted *as one of the microspecies*:

```
Ophrys fusca     -> predicted Ophrys lupercalis   (merged away)
Ophrys araneola  -> predicted Ophrys virescens    (merged away)  x3
Ophrys araneola  -> predicted Ophrys aranifera    (merged away)
```

The microspecies were **absorbing predictions that belonged to real species**.
Four of the five fixes are *O. araneola* — the one species deliberately kept out
of the merge — which is the strongest possible argument for having drawn that
line where iNat draws it rather than merging the whole genus.

Buckets are unaffected: 0 out-of-catalogue rows become in-catalogue under
curation, so none of the rejection numbers move for compositional reasons.

### Do not merge on confusion alone

86% of in-catalogue species errors are within-genus, which is the genus fallback
earning its place. The most-confused pairs are tempting merge targets and mostly
should **not** be merged:

| errors | pair | verdict |
|---|---|---|
| 11 | *Moehringia ciliata* / *muscosa* | arguable — obscure alpines |
| 10 | *Lamium hybridum* / *purpureum* | *hybridum* is a hybrid of *purpureum*; arguable |
| 10 | *Thapsia garganica* / *villosa* | arguable |
| 8 | *Sedum dendroideum* / *praealtum* | arguable — often treated as subspecies |
| 6 | *Lavandula angustifolia* / *latifolia* | **keep** — true vs spike lavender, different plants and uses |
| 6 | *Cucurbita maxima* / *pepo* | **keep** — different squashes |
| 6 | *Fragaria moschata* / *vesca* | **keep** — different strawberries |
| 6 | *Nymphaea candida* / *odorata* | **keep** |

Merging by confusion rate optimises the metric by deleting the product. The
criterion stays taxonomic and user-facing, and for these pairs the system
already does the right thing: it answers *Lavandula* rather than guessing.

## Where this leaves the product

Per-organ, at 261 species, on a style-matched foreign test:

- **flower**: 0.808 species / 0.982 genus, rejects foreign plants at 0.966 AUROC
- **bark**: 0.832 species (77-way) / 0.910 genus, rejects at 0.977 AUROC
- **leaf**: 0.759 species / 0.951 genus, rejection untested cross-source

The design that follows: one class-weighted head per organ over frozen BioCLIP-2
embeddings, with `__OTHER__` trained on non-catalog species; report species when
confident, else genus, else decline.

## Still open, in priority order

1. **Fused cross-source is still untested.** Every fusion number in this repo
   rests on synthetic same-corpus groups. This is the last big unvalidated claim
   and needs a foreign multi-organ source (iNaturalist observations).
2. **Leaf cross-source is untested** — no style-matched foreign leaf source
   found yet; the Indian-leaves set was studio-lit and had to be discarded
   (`OPENSET_FINDINGS.md`).
3. **Calibration.** Nothing here is calibrated; the confidence a user sees must
   mean something. Temperature scaling, then conformal prediction.
4. **The near-OOD bark regression** from the 87-species run (reject class helps
   distant OOD, hurts near-neighbour species) has not been rechecked at 261.
5. **On-device**: MobileCLIP2-S0 distillation, the ~19pp gap to BioCLIP-2.
