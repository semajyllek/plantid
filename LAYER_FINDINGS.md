# We are reading the right layer

**Negative result. For BioCLIP-2 on narrow-catalogue classification, the final
projected embedding is the best representation in the network, and nothing is
gained by reading earlier layers or by concatenating them.**

## Why it was worth testing

Meta's [Perception Encoder](https://www.alphaxiv.org/abs/2504.13181) — *"The best
visual embeddings are not at the output of the network"* — reports that for
contrastively-trained encoders the strongest downstream representations sit in
the middle of the stack, because the last layers specialise toward the
contrastive objective and discard detail a downstream head could use.

Every number in this repo reads BioCLIP-2's **output**: `proj(ln_post(x[:, 0]))`
after all 24 blocks. If the claim transferred it would be free accuracy — no
compression, no training, no change in model size. It cost one embedding pass to
find out.

## Method

4,251 iNaturalist images of 20 common Oregon plants, split by observation, 8
seeds, logistic head, `C=10`. CLS token captured after blocks 12/16/18/20/21/22/23/24
by forward hook, plus mean-pooled patch tokens at the same depths — pooling is
otherwise confounded with depth. `analysis/layer_probe.py`.

## Result

| representation | dim | top-1 | sd | vs final |
|---|---|---|---|---|
| **final (projected)** | 768 | **0.9765** | 0.0032 | — |
| cls24 (last block, pre-projection) | 1024 | 0.9754 | 0.0021 | −0.0011 |
| cls23 | 1024 | 0.9643 | 0.0041 | −0.0122 |
| mean24 | 1024 | 0.9615 | 0.0040 | −0.0150 |
| mean20 | 1024 | 0.9594 | 0.0035 | −0.0171 |
| cls22 | 1024 | 0.9532 | 0.0027 | −0.0233 |
| cls20 | 1024 | 0.9017 | 0.0048 | −0.0748 |
| cls18 | 1024 | 0.8107 | 0.0077 | −0.1658 |
| cls16 | 1024 | 0.6943 | 0.0137 | −0.2822 |
| cls12 | 1024 | 0.5827 | 0.0128 | −0.3938 |

Monotone. Every intermediate layer is worse and the degradation accelerates with
depth removed. `cls24` is statistically tied with the projected output (within
one sd), which only says the final projection neither adds nor destroys anything
for a linear head.

Concatenation does not rescue it either:

| representation | dim | top-1 | vs final |
|---|---|---|---|
| final alone | 768 | 0.9765 | — |
| final + cls24 | 1792 | 0.9765 | +0.0000 |
| final + cls24 + mean20 | 2816 | 0.9755 | −0.0010 |
| final + mean18 | 1792 | 0.9732 | −0.0033 |

## Why it probably does not transfer

**Offered as an explanation, not a measurement.** The Perception Encoder result
concerns general contrastive encoders, whose final layers specialise toward
image–text alignment — an objective quite different from fine-grained visual
discrimination. BioCLIP-2 is trained *hierarchically on taxonomic labels*, so its
output layer is already optimised for precisely the thing a narrow biological
catalogue asks of it. When the pretraining objective and the downstream task
coincide, there is nothing left in earlier layers to recover.

That predicts the finding would look different for a general encoder
(MobileCLIP2, SigLIP 2) probed on the same task. Untested.

## One real pattern, not actionable

Mean-pooled patches beat CLS at mid-depths and lose to it at the end —
mean18 0.944 vs cls18 0.811, but mean24 0.962 vs cls24 0.975. CLS aggregates
globally only in the last blocks, so mid-stack it has less information than the
patch mean. Consistent with how ViTs are known to behave; no use here.

## What this settles

The representation being compressed in any future pruning or distillation work
is the right one. That was worth a day to know before spending a GPU.
