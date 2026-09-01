# On-device: there is no giant-model problem to solve

**BioCLIP v1 fits the deployment budget and clears the genus target. Distillation
is not needed.**

Phase 5 assumed the shipped encoder would have to be compressed or distilled,
because the chosen one is large. Measuring first made that project unnecessary.

## The premise was worse than stated, then dissolved

BioCLIP-2's image tower is **304M parameters — a ViT-L**, not the ViT-B assumed
earlier. At 152 MB even under 4-bit palettization and ~98 ms per image, it does
not ship.

But the bake-off already contained the clue that compression was the wrong fix:

| encoder | params | weighted top-1 @87 species |
|---|---|---|
| MobileCLIP2-S0 | 11M | 0.674 |
| DINOv3-S | 21M | 0.669 |
| DINOv3-B | 86M | 0.724 |
| BioCLIP-2 | 304M | **0.863** |

**DINOv3-B is 86M and reaches only 0.724.** A 4x larger general-purpose model
does not approach a biology-trained one, so BioCLIP-2's advantage is its
training data, not its size. That reframes the fix: find a *small model trained
on the right data*, rather than compress a large one.

## The deployment frontier

| encoder | params | int4 size | latency (MPS) | fits <50 MB / <100 ms |
|---|---|---|---|---|
| MobileCLIP2-S0 | 11.4M | 5.7 MB | 9.1 ms | ✅ |
| MobileCLIP2-S2 | 35.8M | 17.9 MB | 18.1 ms | ✅ |
| **BioCLIP v1 (ViT-B)** | **86.2M** | **43.1 MB** | **21.8 ms** | ✅ |
| BioCLIP-2 (ViT-L) | 304M | 152 MB | 97.8 ms | ✗ |

## Accuracy at 530 species, both fitted identically

Catalogue train split plus a class-weighted `__OTHER__` from the same background
species — see the note below on why that matters.

| encoder | organ | species | genus |
|---|---|---|---|
| BioCLIP v1 | leaf | 0.682 | 0.896 |
| BioCLIP v1 | bark | 0.675 | 0.757 |
| BioCLIP v1 | flower | 0.761 | **0.961** |
| **BioCLIP v1** | **weighted** | **0.722** | **0.925** ✅ |
| BioCLIP-2 | leaf | 0.733 | 0.944 |
| BioCLIP-2 | bark | 0.840 | 0.893 |
| BioCLIP-2 | flower | 0.774 | 0.972 |
| BioCLIP-2 | weighted | 0.757 | 0.957 |

**BioCLIP v1 costs 3.5pp of species accuracy and 3.2pp of genus accuracy, for a
3.5x smaller and 4.5x faster model that clears the ≥90% genus target at 0.925.**

Per organ the picture is uneven and worth stating: flower is strong at 0.961,
leaf is marginal at 0.896 — just under the bar on its own — and bark is 0.757,
well under, though it is the optional organ. The weighted figure the product
actually delivers is 0.925.

## A comparison error caught before reporting

The first run measured BioCLIP v1 **without** the `__OTHER__` reject class,
because `load_background` looks for a per-encoder cache and silently had none
for v1, while BioCLIP-2 had one. That flatters v1: with no reject class competing
for probability mass, in-catalogue accuracy rises.

The effect turned out to be small — v1 weighted genus 0.925 either way — but the
comparison was not apples-to-apples and would have been reported as though it
were. The background pool is now embedded per encoder.

## What this means for Phase 5

- **Ship BioCLIP v1.** 43 MB at int4, ~22 ms, genus 0.925.
- **Distillation is cancelled.** It was the plan of record; it would have been
  weeks of work to recover part of a gap that an off-the-shelf model already
  closes.
- **MobileCLIP2-S2 (17.9 MB) is the fallback** if 43 MB proves too large in
  practice, and is untested — worth measuring only if the size budget tightens.

## Still to do for a shippable model

1. **Core ML export and real-device benchmarking.** The 21.8 ms figure is MPS on
   an M4 Max, not the Neural Engine on a phone. Same order, different number.
2. **Quantization accuracy check.** 43 MB assumes 4-bit palettization; the
   accuracy cost of that has not been measured, and ViTs can be sensitive.
3. **Verify on the newly added species.** 269 of the 530 have no
   real-observation evaluation — their figures come from the PlantNet test split
   alone.
