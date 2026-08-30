# plantid

Multi-organ plant species identification. Given one or more photos of a plant —
leaf, flower, and optionally bark/fruit — identify the species.

**Current state:** a working 87-species research pipeline (fine-tuned CNN
embeddings + k-NN + late fusion, `fused top-1 = 0.738 ± 5.6pp`). The project is
now pivoting to v2: ~1000 European/North American species, running entirely
on-device on iOS, from a guided capture flow. See [`ROADMAP.md`](ROADMAP.md).

Sibling projects: [imret](https://github.com/semajyllek/imret) (image retrieval
engine) and [artfinder](https://github.com/semajyllek/artfinder) (painting
matcher). Note that imret was **evaluated and rejected** for this task — ORB
keypoint matching is built for instance-level near-duplicate retrieval and
performs at chance for species-level identification (`PHASE3_FINDINGS.md`).

## What was learned (v1)

The project began as a classical-CV and image-retrieval exercise. That approach
was built, evaluated honestly, and **lost** — the findings docs record it:

| stage | leaf top-1 | bark top-1 | flower top-1 |
|---|---|---|---|
| Classical descriptors + k-NN (Phase 2/3) | 0.065 | 0.098 | 0.156 |
| Fine-tuned `mobilenet_v3_small` k-NN | 0.53–0.55 | 0.28–0.32 | 0.66–0.68 |

Chance is 0.011 (87 species). Key results, each documented in full:

- **[`PHASE1_FINDINGS.md`](PHASE1_FINDINGS.md)** — PlantNet-300K audit. Real
  field photos with heavy background clutter; `obs_id` is 1:1 with images, so
  there are no native multi-organ observations to group on.
- **[`PHASE2_FINDINGS.md`](PHASE2_FINDINGS.md)** — per-organ classical
  descriptors (Hu moments / LBP / GLCM / HSV) and a label-noise outlier pass
  that flags ~4% of images. GrabCut was tried and abandoned.
- **[`PHASE3_FINDINGS.md`](PHASE3_FINDINGS.md)** — k-NN matching, the imret
  rejection, and the finding that RRF fusion *diluted* the strongest organ when
  the others carried near-noise.
- **[`CNN_FINDINGS.md`](CNN_FINDINGS.md)** — fine-tuned CNN embeddings, a 6–9x
  jump that reversed the fusion conclusion. Includes an **audit** showing the
  once-headline `top-10 = 0.980` is largely a candidate-list-length artifact;
  quote `fused top-1 = 0.738 ± 5.6pp` instead.
- **[`CROP_FINDINGS.md`](CROP_FINDINGS.md)** — organ isolation via
  LocateAnything-3B grounding. **Negative result**, with a mechanism: only
  ~26–33% of photos yield a usable crop, because "a single flower" is ill-posed
  for a dense mat or a compound inflorescence.

The recurring theme: several plausible ideas failed, and the reasons why are
more reusable than the successes.

## Data

PlantNet-300K (306,146 images, 1,081 species), images fetched via the Pl@ntNet
image API. The v1 working set is 87 species / 10,777 images — a number set
entirely by **bark scarcity** (bark is 0.7% of the corpus; only 13 species have
≥20 images of leaf, flower *and* bark). v2 drops bark as a required organ and
adds iNaturalist as the primary corpus.

## Layout

```
plantid/
├── plantid/
│   ├── data/        # dataset audit, index building, group sampling
│   ├── features/    # classical descriptors, outlier pass, descriptor cache
│   ├── matching/    # k-NN matcher + late fusion
│   └── eval/        # evaluation harness
├── notebooks/       # Colab: CNN fine-tuning, LocateAnything crop experiment
├── data/            # gitignored
└── tests/           # 48 tests
```

The evaluation harness is descriptor-agnostic: any cached
`descriptors_{organ}_{variant}.npz` drops into
`store.load_descriptors(organ, variant=...)` and is directly comparable through
`match_eval.evaluate_organ` / `fusion.evaluate_fusion`. This is how the classical,
CE, SupCon, and crop-trained variants were all compared, and how v2's frozen
pretrained encoders will be.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```
