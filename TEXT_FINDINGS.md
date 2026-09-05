# The coarse-answer trap is not about vision either

**It reproduces on text, in all three pre-registered directions, and more
strongly than on either image domain.** Prediction declared in
[`TEXT_PREREG.md`](TEXT_PREREG.md) before anything was embedded.

## Result

20 Newsgroups (headers/footers/quotes removed), `all-MiniLM-L6-v2` (22.7M
params), 7 labels per arm, 400 documents per label, negatives drawn from the 8
unused newsgroups. Text embedded outside the tool and passed through
`--embeddings` — the boundary narrowcast is designed around.

| arm | groups | coverage | precision | **label-level** | top-1 |
|---|---|---|---|---|---|
| VARIED — 7 top-level groups | 7 | 0.552 | 0.960 | **0.634** | 0.864 |
| CROWDED — 4 `comp.*` + 3 `rec.*` | 2 | **0.768** | 0.966 | **0.309** | 0.761 |

All three predictions hold: **higher coverage** (+21.6pp), **comparable
precision** (+0.6pp), **substantially lower label-level share** (−32.5pp).

## Across three domains and two modalities

| | coverage | precision | label-level |
|---|---|---|---|
| plants, varied → crowded | 0.618 → **0.806** | 0.985 → 0.978 | 0.761 → **0.476** |
| birds, varied → crowded | 0.766 → **0.784** | 0.991 → 0.989 | 0.958 → **0.718** |
| **text**, varied → crowded | 0.552 → **0.768** | 0.960 → 0.966 | 0.634 → **0.309** |

Text shows the largest effect on both axes. The trap is a property of
hierarchical label sets, not of images, biology, or one encoder family.

## The condition it depends on

A crowded set only *inflates coverage* if the cascade actually uses the group
rank, and it only uses it when a group answer beats declining under the declared
utility. With `group_correct=+0.5`, `wrong=−4.0`, `decline_in_catalog=0`:

    0.5·p − 4·(1−p) > 0   ⟹   p > 0.889

So the group rank must be **~89% accurate** to be worth using at all. Measured:

| arm | label acc | group acc | headroom | clears 0.889 |
|---|---|---|---|---|
| text, crowded | 0.771 | 0.965 | +0.194 | yes |
| text, varied | 0.854 | 0.854 | +0.000 | no |
| plants (490-class) | 0.837 | 0.974 | +0.137 | yes |

The varied arm's zero headroom is not a defect — with one label per group, a
group answer *is* a label answer, so no headroom is possible and the cascade
correctly never falls back. That is the mechanism, stated precisely: **the
coverage inversion needs a coarse rank that is both informative and much more
accurate than the fine one.**

## A bug this found, in code aimed at something else

The first two attempts showed **zero** headroom in *both* arms — group accuracy
exactly equal to label accuracy, to three decimals. That is not a plausible
measurement, and it was not one.

narrowcast read the caller's `group` column, carried it, and then discarded it:
`score_frame` and `cascade.group_matrix` both re-derived the group as the label's
first whitespace token, a Latin-binomial convention. On `comp.sys.mac.hardware`
every label became its own group, so the group rank carried no information and
the cascade correctly refused to use it.

That silently disabled the coarse rank for **every non-binomial domain** — the
whole domain-generality claim. Fixed, with a regression test.

**No previously committed result is affected.** Plants (`Sedum acre`), birds
(`Larus occidentalis`) and the synthetic defect set (`weld defect0`) are all
whitespace-separated, so the default rule was already correct for each.

## Detour worth recording

A first arm design used one group of five against five groups of one. Neither
exercises the group fallback — one has a group that narrows nothing, the other
has groups identical to labels — so the mechanism under test could not appear at
all. The image arms had been 2 groups × ~6 members; the text arms were rebuilt to
match before any conclusion was drawn.

## Limits

One text encoder, one corpus, seven labels per arm. Establishes the effect is not
vision-specific; does not establish a magnitude for text in general. Documents
are independent so no cluster column was supplied, and the card records that its
intervals are anticonservative accordingly.
