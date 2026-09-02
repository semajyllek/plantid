# plantid

Multi-organ plant species identification. Given one or more photos of a plant —
leaf, flower, and optionally bark — name the species, name the genus, or say
"I'm not sure", and be right about which of those three it can defend.

**Current state:** a **490-class** European/North-American catalogue over frozen
BioCLIP embeddings with a class-weighted reject class, trained entirely on local
Apple Silicon with no backbone fine-tuning. At an assumed 20% out-of-catalogue
rate it answers **72% of captures at 95.6% precision**, evaluated on 5,534 real
iNaturalist observations. Target is an on-device iOS app with guided multi-organ
capture. See [`ROADMAP.md`](ROADMAP.md).

The headline is deliberately *not* top-1 accuracy. Species accuracy (0.85) caps
what species-level answering can achieve, genus accuracy (0.97) is far higher,
and the product's value is in knowing which it can offer — so the metric is
**precision at coverage with an abstain option**, reported as a curve over the
assumed out-of-catalogue rate rather than as a scalar.

## How it works

```
photo → [frozen BioCLIP encoder] → embedding → [per-organ logistic head] → P(species)
            (never trained)                       (trained locally in seconds)
                                                            ↓
                                    organ router marginalises over leaf/bark/flower
                                                            ↓
                                    trimmed-mean fusion across an observation's photos
                                                            ↓
                        decline < genus confidence < report genus < species confidence
```

Embeddings are extracted **once** per image, so every subsequent experiment —
head, label space, fusion rule, calibration, thresholds — is minutes of CPU on
cached vectors. That is what makes a no-cloud-GPU project tractable.

**The three-way decision** ([`REJECTION_FINDINGS.md`](REJECTION_FINDINGS.md)) is
the core of the product. Its scores are *nested by construction*
(`max P(species) ≤ max P(genus) ≤ 1 − P(__OTHER__)`), so the cascade is
well-ordered and thresholds are fitted by expected-utility maximisation with
utilities declared before fitting — never read off a test set.

## Results

| | |
|---|---|
| catalogue | 490 classes, 172 genera |
| evaluation | 5,534 real iNat observations / 20,494 photos, 465 species covered |
| in-catalogue accuracy | species **0.846**, genus **0.975** |
| precision / coverage @20% OOD | **0.956 / 0.722** |
| out-of-catalogue declined | 96.7% global, 97.9% regional |
| deployable encoder | BioCLIP v1 ViT-B, 43 MB at int4, genus 0.925 |

## Findings

Each document records what was measured, including the things that failed. The
negative results are the more reusable half.

**Current architecture**
- **[`BAKEOFF_FINDINGS.md`](BAKEOFF_FINDINGS.md)** — frozen-encoder bake-off.
  Biological pretraining beats scale: DINOv3-B at the same size reaches 0.724
  against BioCLIP-2's 0.863.
- **[`CATALOG_FINDINGS.md`](CATALOG_FINDINGS.md)** — catalogue construction,
  261 → 530 species, and label curation (hybrid-name bug, *Ophrys* microspecies).
- **[`REJECTION_FINDINGS.md`](REJECTION_FINDINGS.md)** — the three-way rule,
  the measured out-of-catalogue rate, and the operating point.
- **[`INAT_FINDINGS.md`](INAT_FINDINGS.md)** — real multi-photo observations.
  Fusion helps accuracy, **not** rejection; 9.5% of catalogue names are a
  taxonomic generation behind, which is why half the catalogue was unevaluated.
- **[`ONDEVICE_FINDINGS.md`](ONDEVICE_FINDINGS.md)** — BioCLIP v1 clears the
  budget, distillation cancelled, 4-bit palettization costs 1.1pp.
- **[`HIERARCHY_FINDINGS.md`](HIERARCHY_FINDINGS.md)** — why genus is a
  first-class answer rather than a hedge.
- **[`COMPETITIVE_FINDINGS.md`](COMPETITIVE_FINDINGS.md)** — head-to-head with
  Pl@ntNet on identical photographs. We name fewer plants correctly than they do
  and give **26x fewer wrong answers**, which is the whole product thesis.

**Measured and rejected**
- **[`LOCATION_FINDINGS.md`](LOCATION_FINDINGS.md)** — geographic prior. Real
  signal (0.715 AUROC, collapses to 0.511 on shuffled coordinates) that does
  **not** pay as a gate: +0.007 utility, CI includes zero. Within-genus
  re-ranking looked good at n=373 and failed to replicate at n=1,150.
- **[`CROP_FINDINGS.md`](CROP_FINDINGS.md)** — organ isolation via VLM
  grounding. Only ~26–33% of photos yield a usable crop, because "a single
  flower" is ill-posed for a dense mat. Guided capture is the response.
- **[`OPENSET_FINDINGS.md`](OPENSET_FINDINGS.md)** — the reject class, and the
  synthetic-group fusion numbers that `INAT_FINDINGS.md` later corrected.

**Superseded (v1, 87 species)** — the project began as a classical-CV and
image-retrieval exercise. It was built, evaluated honestly, and lost:

| stage | leaf top-1 | bark top-1 | flower top-1 |
|---|---|---|---|
| classical descriptors + k-NN | 0.065 | 0.098 | 0.156 |
| fine-tuned `mobilenet_v3_small` | 0.53–0.55 | 0.28–0.32 | 0.66–0.68 |
| **frozen BioCLIP-2, zero training** | **0.78** | **0.83** | **0.84** |

Chance is 0.011 at 87 species. Details in
[`PHASE1_FINDINGS.md`](PHASE1_FINDINGS.md) (corpus audit),
[`PHASE2_FINDINGS.md`](PHASE2_FINDINGS.md) (classical descriptors),
[`PHASE3_FINDINGS.md`](PHASE3_FINDINGS.md) (k-NN and the RRF dilution finding),
and [`CNN_FINDINGS.md`](CNN_FINDINGS.md) — which includes an audit showing the
once-headline `top-10 = 0.980` was largely a candidate-list-length artifact.

Sibling projects: [imret](https://github.com/semajyllek/imret) and
[artfinder](https://github.com/semajyllek/artfinder). imret was **evaluated and
rejected** for this task — ORB keypoint matching is built for instance-level
near-duplicate retrieval and performs at chance for species identification.

## Data

- **PlantNet-300K** — 306,146 images, 1,081 species. Source of the catalogue and
  the `__OTHER__` background pool. Bark is 0.7% of it, which is why bark is an
  optional organ rather than a required one.
- **iNaturalist** — research-grade observations with ≥2 photos, used for
  *evaluation only*. Real multi-photo groups of one individual plant, which
  PlantNet structurally cannot provide (`obs_id` is 1:1 with images).

Both are treated with a standing caveat: iNaturalist feeds GBIF, which feeds
BioCLIP-2's training data, so absolute levels are optimistic even where the
comparisons between arms are sound. See
[`DATA_STRATEGY.md`](DATA_STRATEGY.md).

## Layout

```
plantid/
├── plantid/
│   ├── data/        # catalogue + background construction, iNat fetch, curation
│   ├── features/    # frozen-encoder embedding, descriptor cache
│   ├── matching/    # k-NN matcher + late fusion (v1)
│   └── eval/        # rejection rule, fusion, combiners, calibration, location
├── notebooks/       # Colab: CNN fine-tuning, crop experiment (both superseded)
├── data/            # gitignored
└── tests/           # 97 tests
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Reproduce the headline (needs the cached embeddings):

```bash
PYTHONPATH=. python -m plantid.eval.rejection
```
