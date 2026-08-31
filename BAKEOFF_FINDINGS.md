# v2 Phase 3: frozen pretrained encoder bake-off

**Result: the gate passed by a wide margin, and `BioCLIP-2` is the winner.
A frozen, untrained encoder beats the fine-tuned CNN by +29pp weighted top-1.
No backbone training happens for the rest of this project.**

Setup: each encoder runs inference-only over all 10,777 images; embeddings are
cached as `descriptors_{organ}_{variant}_emb.npz` and evaluated through the
existing `match_eval.evaluate_organ` harness, so every number is directly
comparable to the classical and fine-tuned-CNN results. All local, on MPS
(M4 Max) — the whole bake-off is ~20 minutes of compute.

> **Contamination caveat (added later).** BioCLIP-2's training corpus
> (TreeOfLife-200M) draws on GBIF and a PlantNet-derived Meta-Album subset, so
> the absolute numbers in this document may be optimistic — treat them as upper
> bounds pending an out-of-distribution test set. The ranking versus the
> fine-tuned CNN is unlikely to be affected. See [`DATA_STRATEGY.md`](DATA_STRATEGY.md).

## Headline

Weighted top-1 across the 1,624 test images (leaf 755 / bark 183 / flower 686):

| approach | trained here? | weighted top-1 |
|---|---|---|
| Classical descriptors (Phase 2/3) | — | 0.107 |
| Fine-tuned `mobilenet_v3_small`, k-NN | ✅ GPU hours | 0.577 |
| Fine-tuned `mobilenet_v3_small`, classifier head | ✅ GPU hours | 0.624 |
| **BioCLIP-2, frozen, k-NN** | ❌ **zero training** | **0.863** |

Per organ, top-1 (k swept on val):

| encoder | dim | leaf | bark | flower | wtd top-1 |
|---|---|---|---|---|---|
| classical | 529/594 | 0.065 | 0.098 | 0.156 | 0.107 |
| fine-tuned CNN | 1024 | 0.517 | 0.317 | 0.679 | 0.577 |
| MobileCLIP2-S0 | 512 | 0.615 | 0.426 | 0.806 | 0.674 |
| DINOv3-S | 384 | 0.653 | 0.410 | 0.755 | 0.669 |
| DINOv3-B | 768 | 0.714 | 0.497 | 0.796 | 0.724 |
| **BioCLIP-2** | 768 | **0.850** | **0.738** | **0.910** | **0.863** |
| PlantCLEF2024 ViT-B | 768 | 0.854 | 0.749 | 0.910 | 0.866 |

Bark — the weakest organ throughout v1, and the one whose scarcity capped the
project at 87 species — goes from 0.317 to 0.738. That is the largest single
improvement in the project's history and it required no training.

## BioCLIP-2 over PlantCLEF2024, despite the tie

PlantCLEF2024 edges BioCLIP-2 on raw weighted top-1 (0.866 vs 0.863), but it is
**contaminated for this benchmark**: it is a DINOv2 ViT-B fine-tuned to
discriminate 7,806 Pl@ntNet species, and **71.3% (62/87) of our test species are
in its training list**. Our test images come from the same Pl@ntNet ecosystem.

To separate memorisation from feature quality, both were evaluated on two
same-size (25-way) label spaces: the 25 species *absent* from PlantCLEF's list,
and a random 25 of the species *present* in it. DINOv3-B is included as a
control — it has no relationship to PlantCLEF's species list, so its gap
measures how much harder the unseen species simply are.

| encoder | organ | seen-25 | unseen-25 | gap | excess over DINOv3 control |
|---|---|---|---|---|---|
| PlantCLEF2024 | leaf | 0.942 | 0.846 | +0.096 | +0.029 |
| PlantCLEF2024 | bark | 0.921 | 0.700 | **+0.221** | **+0.158** |
| PlantCLEF2024 | flower | 0.938 | 0.925 | +0.013 | +0.084 |
| BioCLIP-2 | leaf | 0.924 | 0.851 | +0.073 | +0.006 |
| BioCLIP-2 | bark | 0.889 | 0.840 | +0.049 | −0.014 |
| BioCLIP-2 | flower | 0.938 | 0.908 | +0.030 | +0.101 |
| DINOv3-B *(control)* | leaf | 0.831 | 0.764 | +0.067 | — |
| DINOv3-B *(control)* | bark | 0.683 | 0.620 | +0.063 | — |
| DINOv3-B *(control)* | flower | 0.820 | 0.891 | −0.071 | — |

The 25 unseen species *are* genuinely harder — DINOv3 loses ~6pp on leaf and
bark too, so the raw gaps overstate contamination. But PlantCLEF's bark gap
(+0.221) is 3.5x the control's, and **on unseen species BioCLIP-2 beats
PlantCLEF on bark by 14pp (0.840 vs 0.700)** while matching it elsewhere.

**Decision: BioCLIP-2.** It generalises better to species outside the in-domain
model's vocabulary, which is exactly the regime the 1000-species v2 target lives
in. It is also 224px versus PlantCLEF's 518px — ~3.5x cheaper per image
(BioCLIP-2 ran the full dataset in ~3 min, PlantCLEF took ~8) and a far better
starting point for on-device work.

## Implications for the roadmap

- **Phase 3 gate passed.** The frozen-backbone architecture is confirmed.
  Backbone fine-tuning is off the table for the rest of the project; all
  remaining modelling is heads over cached embeddings, which is minutes of CPU.
- **Phase 4 can start immediately** — the scaling curve (87 → 250 → 500 → 1000
  species) now runs against BioCLIP-2 embeddings.
- **Revisit the bark decision.** `ROADMAP.md` §0 drops bark as a required organ
  because of data scarcity, and that conclusion stands — 13 species with ≥20
  bark images is not a dataset. But bark at 0.738 top-1 is no longer a weak
  signal, so it is worth keeping as an opportunistic input wherever it exists.
- **On-device**: neither winner is deployable as-is (ViT-B, ~86M params).
  MobileCLIP2-S0 at 0.674 is the deployable-today floor and is already ahead of
  the fine-tuned CNN. Phase 5's distillation target is the BioCLIP-2 → MobileCLIP
  gap of ~19pp.

## Known issue in the harness (affects top-5/top-10 only)

`ClassicalMatcher.rank_species` returns at most one entry per distinct species
among the `k` nearest neighbours, so a ranking can never be longer than
`k_match`. The val sweep now selects small k for these strong encoders (k=1 for
PlantCLEF bark, k=5 for several), which truncates the candidate list and makes
top-5/top-10 collapse onto top-1/top-5 — e.g. PlantCLEF bark reports
0.749/0.749/0.749. **The top-1 column is unaffected and is what this document
compares on.** Before top-5/top-10 are quoted anywhere, the harness needs to
decouple "k neighbours used for voting" from "length of the returned candidate
list" (see the same issue analysed in `CNN_FINDINGS.md`).

## Reproduce

```bash
# .venv-mps: python3.12 + torch/timm/open_clip (Apple Silicon MPS)
PYTHONPATH=. .venv-mps/bin/python -m plantid.features.pretrained bioclip2
python -c "from plantid.eval.match_eval import evaluate_organ; print(evaluate_organ('leaf', variant='bioclip2_emb'))"
```
