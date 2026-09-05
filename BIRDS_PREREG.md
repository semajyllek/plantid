# Pre-registration — does the coarse-answer trap reproduce outside plants?

Written **before** any bird image was fetched or fitted.

## Why

`EMBEDDED_FINDINGS.md` and `plantid/tool/` rest on one claim: **the shape of a
label set decides how a model fails, and the standard metrics hide it.** A
genus-crowded catalogue buys coverage with genus answers that narrow nothing, so
it scores *better* on coverage and precision while being the worse model.

That is measured on plants, one encoder, one label hierarchy. The tool is
intended to be domain-general. This tests whether the finding is.

## Design

Same encoder (BioCLIP-2), same pipeline, same region (Oregon), different kingdom.
Two 13-species bird sets from iNaturalist research-grade observations:

- **CROWDED** — 2 genera: *Larus* (7 gulls) + *Calidris* (6 sandpipers). The two
  groups birders consider hardest to identify to species.
- **VARIED** — 13 genera, one species each: mallard, junco, robin, great blue
  heron, song sparrow, Canada goose, bald eagle, scrub-jay, red-tailed hawk,
  crow, Anna's hummingbird, spotted towhee, house finch.

Split by observation, never by image. Same head, same three-way cascade, same
declared `UTILITY`, prevalence anchored at 20%.

## Prediction, declared now

If the finding is general, the crowded set will show:

1. **Higher or comparable coverage** than the varied set
2. **Comparable precision**
3. **Substantially lower species-level share**

On plants the same contrast gave coverage 0.806 vs 0.618, precision 0.978 vs
0.985, species-level **0.476 vs 0.761**. Direction is what is being tested, not
magnitude.

## What would falsify it

The crowded set showing *lower* coverage, or a species-level share close to the
varied set's. Either would mean the trap is a property of plant taxonomy or of
BioCLIP-2's plant training rather than of hierarchical label sets, and the tool's
domain-general claim should be dropped or restricted to plants.

## Known limitation

Birds and plants are both in BioCLIP-2's training data, and both from
iNaturalist. This tests generality across *label hierarchy and visual domain*,
not across encoders or data sources.
