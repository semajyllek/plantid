# The coarse-answer trap is not a plant finding

**It reproduces on birds, in the predicted direction, at comparable magnitude.**
The tool's domain-general claim is earned rather than assumed.

Prediction declared in [`BIRDS_PREREG.md`](BIRDS_PREREG.md) before any bird image
was fetched.

## Result

Same encoder (BioCLIP-2), same pipeline, same region, same declared `UTILITY`,
prevalence anchored at 20%. Two 13-species Oregon bird sets, 5,250 images over
4,018 observations, split by observation.

| arm | genera | coverage | precision | **species-level** | top-1 |
|---|---|---|---|---|---|
| VARIED — 13 genera | 13 | 0.766 | 0.991 | **0.958** | 0.970 |
| CROWDED — *Larus* + *Calidris* | 2 | **0.784** | 0.989 | **0.718** | 0.899 |

All three pre-registered predictions hold:

1. **Higher coverage** for the crowded set — 0.784 vs 0.766
2. **Comparable precision** — 0.989 vs 0.991
3. **Substantially lower species-level share** — 0.718 vs 0.958, a **24pp** drop

The crowded set answers *more* queries at the *same* precision and is the worse
model. It buys that coverage with genus answers — "it is a *Larus*" on a list
that is more than half gulls — exactly as on plants.

## Against the plant result

| | coverage | precision | species-level |
|---|---|---|---|
| plants, varied → crowded | 0.618 → 0.806 | 0.985 → 0.978 | 0.761 → **0.476** |
| birds, varied → crowded | 0.766 → 0.784 | 0.991 → 0.989 | 0.958 → **0.718** |

Species-level falls 28.5pp on plants and 24.0pp on birds. The coverage rise is
much smaller on birds (+1.8pp vs +18.8pp), so the trap is not identical in shape
— but the part that matters is: **the two metrics everyone reports move the wrong
way or not at all, while the metric that determines usefulness collapses.**

*Larus* and *Calidris* are the groups birders themselves consider hardest, so the
crowded arm is a fair analogue of the all-*Sedum* list rather than a contrived one.

## What this licenses, and what it does not

**Licenses:** stating the finding as a property of hierarchical label sets rather
than of plant taxonomy. The tool can carry the warning into any domain with a
label/group hierarchy, and `plan`'s structural warnings are justified without a
domain profile.

**Does not license:** the *numbers*. `frontier.json` remains plant-and-BioCLIP-2
specific, and projection must stay disabled absent a matching profile. This test
was run through the same encoder on the same platform, so it establishes
generality across label hierarchy and visual domain — not across encoders or
data sources.

## Reproduce

```
python analysis/bird_fetch.py
```
