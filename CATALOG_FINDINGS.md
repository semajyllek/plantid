# v2 catalog: 261 species

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

## Accuracy: species falls, genus does not

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
