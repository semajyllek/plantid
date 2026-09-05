# Pruning a big encoder loses to picking a small one

**Depth-pruned BioCLIP-2 at 127 MB scores 0.9323. Off-the-shelf PlantCLEF2024 at
43.3 MB scores 0.9621. The pruned model is 3x larger and 3pp worse.**

Distillation does recover a real fraction of what pruning costs. It does not
recover enough to matter, and the size ceiling is structural.

## The size ceiling is arithmetic, not tuning

99% of ViT-L/14's visual tower is its 24 residual blocks at 12.60M each; the
non-block floor — patch embedding, positional embedding, norms, projection — is
1.66M. So depth pruning is linear in size:

| blocks kept | params | int4 | zero-shot top-1 |
|---|---|---|---|
| 24 | 304.0M | 152 MB | 0.9754 |
| 20 | 253.6M | 127 MB | 0.9017 |
| 16 | 203.2M | 102 MB | 0.6943 |
| 12 | 152.8M | 76 MB | 0.5827 |
| 4 | 52.0M | 26 MB | — |

Reaching MobileCLIP2-S2's 17.9 MB would take **~3 of 24 blocks**. Twelve blocks
already collapses to 0.583. Depth pruning cannot reach a small budget from a
large model, whatever the recovery method.

The zero-shot column is free: probing layer *k* (`LAYER_FINDINGS.md`) *is*
truncating to *k* blocks and refitting the head.

## Distillation recovers 26–49%, and it is not enough

Last 2 surviving blocks plus a linear adapter trained against cached full-teacher
embeddings, cosine loss, 2 epochs, 4,251 Oregon images
(`analysis/prune_distill.py`). Teacher: 0.9760.

| blocks | int4 | before | after | recovered | vs teacher |
|---|---|---|---|---|---|
| 20 | 127 MB | 0.8902 | **0.9323** | 49% of 8.6pp | −4.4pp |
| 18 | 114 MB | 0.8037 | 0.8753 | 42% of 17.2pp | −10.1pp |
| 16 | 102 MB | 0.6660 | 0.7473 | 26% of 31.0pp | −22.9pp |

Recovery is real and the recoverable *fraction* falls as pruning deepens —
49% → 42% → 26%. Deeper cuts remove information later blocks cannot reconstruct.

Set against the alternatives on the same 20 Oregon plants, same protocol:

| model | int4 | top-1 |
|---|---|---|
| BioCLIP-2, unpruned | 152 MB | 0.9769 |
| **PlantCLEF2024, off the shelf** | **43.3 MB** | **0.9621** |
| BioCLIP-2 pruned to 20 blocks + distilled | 127 MB | 0.9323 |
| MobileCLIP2-S2, off the shelf | 17.9 MB | 0.8557 |

Every pruned point is dominated. There is no budget at which pruning wins.

## Bounds on this claim

Two epochs, with cosine loss still falling (0.25 → 0.13 at 20 blocks), so more
training recovers more. 4,251 transfer images, against the 48k the earlier
distillation attempt used. Depth-only pruning, where Minitron-style recipes prune
width, heads and MLP as well. One domain, one teacher.

All of those would improve the pruned numbers. To change the conclusion they
would have to close ~3pp **and** deliver a 3x smaller model, and nothing here
suggests that margin is available.

## What it settles

**For large compression ratios, selecting a natively-small encoder dominates
compressing a large one.** Measured twice now: PlantCLEF2024 beat BioCLIP-2 on
hazard safety at 3.5x smaller (`OREGON_SAFETY_FINDINGS.md`), and pruning cannot
reach it.

So the constraint-driven tool is a **search-and-verify loop**, not a compression
pipeline — spend the GPU evaluating breadth rather than forcing one architecture
into a shape it resists. That also generalises to domains with no teacher worth
distilling, which is most of them.

Third strike for the same rule: **cosine to a teacher does not predict downstream
accuracy.** Loss fell by half here while top-1 moved 4.2pp at 20 blocks and 8.1pp
at 16 — the mapping is not monotone across configurations.
