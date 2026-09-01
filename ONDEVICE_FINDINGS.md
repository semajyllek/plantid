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

- **Ship BioCLIP v1 at 4-bit.** 43 MB, ~22 ms, genus 0.914 after quantization.
- **Distillation is cancelled.** It was the plan of record; it would have been
  weeks of work to recover part of a gap that an off-the-shelf model already
  closes.
- **MobileCLIP2-S2 (17.9 MB) is the fallback** if 43 MB proves too large in
  practice, and is untested — worth measuring only if the size budget tightens.

## What 4-bit palettization costs: 1.1pp of genus accuracy

The 43 MB figure assumes 4-bit weights, and ViTs can be sensitive to aggressive
quantization, so this needed measuring rather than assuming.

Simulated in PyTorch as Core ML does it — `2**n_bits` k-means centroids per
weight tensor, per output channel, with norm scales and biases left in float
(`pretrained.palettize_`). Catalogue and background were then re-embedded through
the quantized model and the head refitted, which is the deployment case: if the
device runs quantized weights, the head is trained on quantized embeddings too.

| | species | genus |
|---|---|---|
| BioCLIP v1, fp32 | 0.722 | 0.925 |
| **BioCLIP v1, 4-bit** | **0.709** | **0.914** ✅ |
| cost | −1.3pp | −1.1pp |

**43 MB, ~22 ms, genus 0.914 — the target still holds.** Quantization is not the
obstacle.

### Embedding drift badly overstates the damage

The intermediate diagnostics looked alarming and would have been the wrong thing
to act on:

| bits | weight error | cosine to fp32 | nearest-neighbour preserved |
|---|---|---|---|
| 8 | 0.016 | 0.998 | 91.8% |
| 6 | 0.044 | 0.986 | 86.2% |
| **4** | 0.113 | **0.898** | **62.0%** |

At 4 bits only 62% of nearest-neighbour relations survive and embeddings move to
cosine 0.898 — yet accuracy falls barely a point. **A retrained head absorbs a
systematic shift almost entirely.** Drift metrics measure whether the geometry
moved, not whether it stopped being separable, and only the second matters here.

### The implementation nearly produced the opposite conclusion

The first version used one shared palette per whole tensor, with centroids
initialised on a linear range from min to max. Weight distributions are
heavy-tailed, so that spends most of the 16 centroids in near-empty tails:

| 4-bit variant | cosine to fp32 | NN preserved |
|---|---|---|
| linear init, per-tensor (first attempt) | **0.214** | 0.8% |
| quantile init, per-channel (realistic) | 0.898 | 62.0% |

That would have been reported as "4-bit destroys the model" — a conclusion about
a crude implementation, not about quantization. Both fixes matter: per-channel
granularity and mass-aware centroid placement.

## Still to do for a shippable model

1. **Core ML export and real-device benchmarking.** The 21.8 ms figure is MPS on
   an M4 Max, not the Neural Engine on a phone. Same order, different number.
   The palettization here is a faithful simulation, but Core ML's own converter
   should be checked against it rather than trusted to match.
2. **Verify on the newly added species.** 269 of the 530 have no
   real-observation evaluation — their figures come from the PlantNet test split
   alone.
