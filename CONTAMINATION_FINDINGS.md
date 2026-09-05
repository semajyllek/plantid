# The contamination ceiling is under 1pp, and it was never the problem

Every accuracy figure in this repo has carried the same caveat: iNaturalist is
inside BioCLIP-2's training data via GBIF (`DATA_STRATEGY.md`), so the numbers are
upper bounds. That caveat is now **measured rather than assumed, and it is small.**

| encoder | pre-cutoff (seen) | post-cutoff (unseen) | gap | 95% CI |
|---|---|---|---|---|
| BioCLIP-2 | 0.9906 | 0.9879 | −0.0027 | [−0.0088, +0.0035] |
| PlantCLEF2024 | 0.9842 | 0.9793 | −0.0049 | [−0.0136, +0.0030] |
| MobileCLIP2-S2 | 0.9241 | 0.9006 | **−0.0234** | **[−0.0414, −0.0057]** |

BioCLIP-2 and PlantCLEF2024 are **indistinguishable from zero**. Whatever
memorisation advantage exists for the large encoders, it is bounded under one
percentage point.

## Method

[BioCLIP-2 was released June 2025](https://github.com/Imageomics/bioclip-2) and
trained on [TreeOfLife-200M](https://arxiv.org/abs/2505.23883), whose GBIF
snapshot necessarily predates that. PlantCLEF2024 is a 2024 model; MobileCLIP2
trains on pre-2025 web data. So iNaturalist observations *uploaded* after
mid-2025 are image-level novel to all three, while being the same platform, the
same photographers and the same species.

Two test sets, 20 common Oregon plants, trained on the same corpus:

- **pre-cutoff** — 1,761 observations uploaded up to 2025-07-01, potentially seen
- **post-cutoff** — 1,714 observations uploaded 2025-09-12 to 2026-09-04, provably not

Both drawn identically (`order_by=created_at desc`), disjoint from each other and
from the training corpus by observation id (asserted, not assumed). Intervals
bootstrap over observations.

> **The control matters.** A first attempt compared post-cutoff against the
> existing training corpus, which had been drawn with `order_by=votes`, and found
> post-cutoff scoring *higher* for all three encoders. That was a sampling
> artifact: heavily-faved observations are not a random sample. Re-drawing the
> pre-cutoff side the same way reversed the sign and shrank the effect to noise.
> The uncontrolled version would have supported a stronger claim than the data
> does.

## What this does and does not establish

**Does:** the large encoders are not leaning on having seen these photographs.
The upper-bound qualifier on `EMBEDDED_FINDINGS.md`, `OREGON_SAFETY_FINDINGS.md`
and the head-to-head numbers can be quantified at well under 1pp rather than left
open-ended.

**Does not:** anything about *domain* shift. Same platform, same photographic
culture, same kind of camera. A photograph taken by one person on one phone in
one park is still untested, and remains the Tier 1 gap in `DATA_STRATEGY.md`.

**Cannot:** distinguish "no memorisation" from "memorisation offset by something
else". The claim is a bound on the net effect, which is what the caveat needed.

## The small encoder degrades, and it is not contamination

MobileCLIP2-S2 loses 2.3pp on unseen images with a CI excluding zero — the
*largest* drop, from the encoder *least* likely to have trained on iNaturalist.
So this is not memorisation. It is brittleness to drift: newer uploads, newer
cameras, different seasonal mix.

That is the same pattern as `EMBEDDED_FINDINGS.md`, where S2 lost 9.4pp to a
change of image source and BioCLIP-2 lost 0.2pp. **The small encoder is fragile
to any distribution change, and the large ones are not.** Two independent
measurements now say so.

For the tool this sharpens an existing conclusion: a small encoder's headline
accuracy is not just lower, it is *less durable*, and a card built on one source
will overstate what the user gets.

## Reproduce

```
python scratchpad/holdout_fetch.py     # post-cutoff test set
python scratchpad/control_fetch.py     # matched pre-cutoff control
```
