# Data strategy: augment, supplement, and (above all) test

Register of candidate datasets beyond PlantNet-300K, and the constraint that
governs how they can be used.

## The finding that governs everything here

**BioCLIP-2 — the encoder we selected — was trained on TreeOfLife-200M, whose
sources include GBIF and a PlantNet-derived dataset.** From the TreeOfLife-200M
dataset card:

> "[GBIF] … cataloging biodiversity data from citizen science sources (e.g.,
> iNaturalist and Observation.org)…"

> "[Meta-Album]: Specifically, we used the Plankton, Insects, Insects 2,
> **PlantNet**, Fungi, PlantVillage, and Medicinal Leaf datasets."

Three consequences, in increasing order of importance:

1. **iNaturalist is inside our encoder's training data** (via GBIF, and via
   iNat21 in TreeOfLife-10M). iNat remains excellent for *training* our head —
   but it can never serve as an honest test of BioCLIP-2's generalisation.
2. **Our own benchmark is probably partially contaminated.** Meta-Album's
   PlantNet subset derives from Pl@ntNet-300K. So the headline
   `BioCLIP-2 weighted top-1 = 0.863` (`BAKEOFF_FINDINGS.md`) may be inflated by
   the encoder having seen images from this exact source. The Meta-Album subset
   is a *sample* of PlantNet rather than all of it, so contamination is likely
   partial — but the magnitude is unknown and currently unmeasurable.
3. **This is the same trap we caught PlantCLEF in, one level deeper.** We
   rejected PlantCLEF for being fine-tuned on 7,806 Pl@ntNet species, then chose
   an encoder whose own corpus also ingests PlantNet. The lesson is not "pick a
   different encoder" — nearly every strong biology encoder is trained on GBIF
   -derived data. The lesson is that **a clean test set is a scarce resource we
   do not currently have.**

### What this does and doesn't invalidate

- The *ranking* (frozen encoders ≫ our fine-tuned CNN) is probably safe: the
  gap is 29pp, and our fine-tuned CNN trained on the actual train split.
- The *absolute* numbers — 0.863 top-1, 76% coverage at 95% precision — should
  be treated as **optimistic upper bounds** until validated out-of-distribution.
- Nothing about the architecture decision changes. Frozen encoder + trained head
  is still right; we just can't yet say how well it will do in a user's hand.

## The three roles, and which one is scarce

| role | purpose | supply |
|---|---|---|
| **Augment** | more images of catalog species → robustness | abundant |
| **Supplement** | more species → catalog breadth + `__OTHER__` pool | abundant |
| **Test** | honest generalisation estimate | **scarce — this is the bottleneck** |

Augmentation and supplementation are close to solved: iNat and GBIF have orders
of magnitude more data than we need. Everything hard is on the test row.

## Register

### Tier 1 — clean test (highest priority, smallest supply)

| source | what it gives | contamination risk | notes |
|---|---|---|---|
| **Self-collected field photos** | the only guaranteed-clean test set | **none** | Also the only data that matches deployment: your camera, your framing, the guided-capture flow. 200–500 photos over 30–50 common species would be transformative for confidence in every number in this repo. |
| **Herbarium 2022 (NYBG/FGVC9)** | pressed, dried specimens | low-moderate | Extreme domain shift — a *hard* generalisation probe, not a proxy for field use. Good for measuring how brittle the embedding is. |
| **BarkVN-50** (`Voxel51/BarkVN-50` on HF) | 50 Vietnamese bark species | low | Non-European, research-collected. Doubles as a genuine out-of-catalog OOD test — exactly the "reject unknown species" case. |
| **Bark-101 / BarkNet 1.0** | bark texture, ~101 / ~23 species | low | Research-collected rather than citizen-science, so less likely inside GBIF. |
| **Lab-condition leaf sets** (Flavia, Swedish Leaf, MalayaKew) | segmented leaves, clean backgrounds | moderate (old, web-indexed) | Small species counts. Useful as a controlled probe of the shape/texture pathway, not as a product benchmark. |

### Tier 2 — augment and supplement (abundant, contaminated for testing)

| source | what it gives | contamination risk | notes |
|---|---|---|---|
| **iNaturalist Open Data / iNat21** | ~10k species, millions of images; **real multi-photo observations of one plant** | **high** — in TreeOfLife | The multi-photo observation structure is its unique value: it retires the synthetic-groups caveat (`CNN_FINDINGS.md`) that qualifies every fusion number we have. Use for training and for *realistic group structure*, never for headline generalisation claims. Filter to "research grade". |
| **GBIF** | occurrence records + media; global | **high** — is the TreeOfLife source | Its real value here is **not images** but the **geographic/seasonal prior** (ROADMAP Phase 8) — occurrence density by location and month, which needs no images at all and cannot be contaminated. |
| **PlantCLEF 2024/2025** | ~1.4M images, 7,806 species | high | Large and in-domain. Fine as head-training data; useless as a test set for either candidate encoder. |
| **Oxford 102 Flowers** | 102 flower categories, ~8k images | high (old, web-scraped) | Classic benchmark, almost certainly inside CLIP-family pretraining. Augmentation only. |
| **PlantVillage** | crop leaf disease, 14 crop species | high — named in TreeOfLife | Wrong task (disease, not species). Low value here. |

### Tier 3 — infrastructure, not images

- **GBIF occurrence density** → the Phase 8 geographic prior. Free, no images,
  no contamination, and likely the highest accuracy-per-effort item in the whole
  roadmap.
- **POWO / World Flora Online** → authoritative taxonomy for reconciling species
  names across datasets. Necessary the moment we merge two sources; PlantNet,
  iNat, and GBIF disagree on synonyms and authorities.

## Recommended actions

1. **Start a self-collected test set now.** It is the only clean measurement
   available, it compounds in value over time, and it is the cheapest item here
   — a phone and a walk. Even 30 species × 10 plants × 3 organs is enough to
   detect a large generalisation gap. Prioritise species in the planned catalog.
2. **Add BarkVN-50 and Herbarium as domain-shift probes** before trusting any
   deployment number. Both are downloadable and require no labelling work.
3. **Use iNat freely for training** the head and the `__OTHER__` class, and for
   real observation-grouped fusion evaluation — but caveat any number computed
   on it, in writing, at the point of use.
4. **Build the taxonomy reconciliation layer early.** Merging PlantNet + iNat
   without resolving synonyms will silently create duplicate and split classes,
   which looks like model error and is very hard to debug after the fact.
5. **Re-report the headline with the caveat.** `BAKEOFF_FINDINGS.md` and
   `OPENSET_FINDINGS.md` numbers are upper bounds pending a clean test.

## Open question worth resolving

Can we *measure* the PlantNet contamination rather than assume it?

The path: Meta-Album's PlantNet subset ("PLT_NET") has a published image
manifest. Intersecting it with our 10,777 working images gives a concrete
overlap count — if it's small or zero, most of concern (2) dissolves.

**Attempted and not yet done.** Meta-Album is not published on Hugging Face
(searched — no such repos); it is distributed via OpenML and
`meta-album.github.io`, so the manifest has to be pulled from there. Worth
doing before re-reporting any headline number.

One bounding consideration in the meantime: Meta-Album ships each dataset in
Micro / Mini / Extended sizes, and the smaller variants are on the order of
10³ images. If BioCLIP-2 ingested a small variant, the overlap with our 10,777
images could be modest. **This is a hypothesis, not a result** — the TreeOfLife
card does not state which variant was used, and the direction of the bias is
known even if the magnitude isn't.
