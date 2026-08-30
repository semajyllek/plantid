# Fine-tuned CNN embeddings: findings

`notebooks/finetune_colab.ipynb` fine-tunes `mobilenet_v3_small` (ImageNet
init) on the 87-species working set, in two variants:

- **CE**: standard cross-entropy.
- **SupCon**: cross-entropy + supervised contrastive loss (Khosla et al. 2020)
  on a 128-dim L2-normalized projection head, two augmented views per image.

Both export a 1024-dim penultimate embedding (`ce_emb` / `supcon_emb`) and,
for SupCon, the 128-dim projection (`supcon_proj`). Exported descriptors live
in `data/processed/descriptors_{organ}_{variant}.npz` (same format as the
Phase 2 classical descriptors); models in `data/processed/cnn_models/`.

## Classifier head (test set)

| organ  | CE top1 | CE top5 | CE top10 | SupCon top1 | SupCon top5 | SupCon top10 |
|--------|---------|---------|----------|-------------|-------------|---------------|
| leaf   | 0.577   | 0.875   | 0.931    | 0.581       | 0.881       | 0.936         |
| bark   | 0.437   | 0.721   | 0.803    | 0.443       | 0.710       | 0.776         |
| flower | 0.723   | 0.930   | 0.965    | 0.741       | 0.923       | 0.961         |
| overall| 0.623   | 0.881   | 0.931    | 0.633       | 0.879       | 0.929         |

CE and SupCon classifier heads are essentially tied (SupCon marginally ahead
on leaf/bark/flower top1, marginally behind on top5/top10). Both are a **huge**
jump over the Phase 3 classical k-NN baseline (leaf 0.065/0.183/0.283, bark
0.098/0.257/0.383, flower 0.156/0.348/0.472) — 6-9x better at top-1.

## k-NN over embeddings (`plantid.eval.match_eval`, per-organ k swept on val)

| organ  | variant     | best_k | top1  | top5  | top10 |
|--------|-------------|--------|-------|-------|-------|
| leaf   | classical   | 15     | 0.065 | 0.183 | 0.283 |
| leaf   | ce_emb      | 10     | 0.532 | 0.816 | 0.856 |
| leaf   | supcon_emb  | 20     | **0.550** | **0.840** | **0.897** |
| leaf   | supcon_proj | 20     | 0.485 | 0.791 | 0.853 |
| bark   | classical   | 30     | 0.098 | 0.257 | 0.383 |
| bark   | ce_emb      | 5      | **0.317** | 0.530 | 0.530 |
| bark   | supcon_emb  | 10     | 0.284 | **0.563** | **0.634** |
| bark   | supcon_proj | 10     | 0.251 | 0.437 | 0.497 |
| flower | classical   | 15     | 0.156 | 0.348 | 0.472 |
| flower | ce_emb      | 15     | 0.659 | **0.895** | **0.920** |
| flower | supcon_emb  | 5      | **0.676** | 0.847 | 0.847 |
| flower | supcon_proj | 15     | 0.620 | 0.854 | 0.894 |

k-NN over CNN embeddings underperforms the classifier head by ~5-12pp at
top-1 (e.g. flower 0.676 vs 0.723), as expected — the classifier head is
trained directly to discriminate, while k-NN only approximates that with a
generic distance metric. But k-NN is what `matching/classical.py` /
`matching/fusion.py` are built around (and what generalizes to species/images
not seen as classifier-head classes), so it's the relevant baseline for the
fusion pipeline.

`ce_emb` and `supcon_emb` trade wins per organ (`supcon_emb` better on
leaf/flower top1, `ce_emb` better on bark and on flower top5/top10);
`supcon_proj` (128-dim, no standardization) is consistently a bit behind both
1024-dim embeddings. **No clear universal winner between CE-only and
CE+SupCon** — the SupCon objective didn't hurt, but didn't deliver the hoped-
for large improvement either, at least at `lambda_supcon=1.0`/15 epochs.

## Fusion — reverses the Phase 3 conclusion

`plantid.matching.fusion.main()`, using each organ's best k-NN variant
(`CNN_DESCRIPTOR_CONFIG`: leaf=supcon_emb@k20, bark=ce_emb@k5,
flower=supcon_emb@k5), on the same 1740 synthetic test groups as Phase 3:

| top-k | leaf  | bark  | flower | fused | fused (val-weighted) |
|-------|-------|-------|--------|-------|------------------------|
| 1     | 0.530 | 0.241 | 0.640  | **0.738** | 0.735              |
| 5     | 0.833 | 0.456 | 0.819  | **0.941** | 0.940              |
| 10    | 0.891 | 0.456 | 0.819  | **0.980** | 0.976              |

(Per-organ accuracies on these synthetic groups are a bit lower than
match_eval's full-test-split numbers above — `sample_groups` draws one image
per organ per group, a different/smaller sample than the full test split,
especially for bark.)

With the classical descriptors, fusion **diluted** the strongest organ
(flower 0.156 -> fused 0.094 at top1, Phase 3). With CNN embeddings, fusion
**beats every single organ** at top-1 — flower alone 0.640 -> fused 0.738
(+10pp). Once each organ carries real signal (not near-noise), RRF's "organs
that agree get boosted" behavior works in fusion's favor instead of against it.
Unweighted and val-weighted fusion are essentially tied; weighting no longer
matters much once all organs are informative.

### How to read these numbers (audited)

**The top-1 result is solid. The top-5/top-10 columns overstate fusion's
advantage, and the headline 0.980 should not be quoted as evidence for fusion.**

1. **The per-organ top-10 column is capped by `k_match`, not by the model.**
   `ClassicalMatcher.rank_species` returns at most one entry per distinct
   species among the `k` nearest neighbours, so a ranking can never be longer
   than `k_match`. With bark and flower at `k_match=5`, their candidate lists
   hold ≤5 species — which is why their top-5 and top-10 are *identical*
   (0.456, 0.819). Their "top-10" was never measured. Measured list lengths on
   these groups: leaf 7.8 mean, bark 3.9 (max 5), flower 2.4 (max 5), **fused
   11.8 (max 23)**. Fusion is allowed to name 3–5x more candidates.

   Give a single organ a comparable budget and most of the gap closes:

   | organ | top-10 @ tuned k | top-10 @ k=60 | fused top-10 |
   |-------|------------------|---------------|--------------|
   | flower | 0.819 (k=5) | **0.949** | 0.980 |
   | leaf   | 0.891 (k=20) | **0.930** | 0.980 |

   The flower-vs-fused top-10 gap is +3pp, not the +16pp the table implies.
   At top-1 there is no such artifact — flower's top-1 is *best* at its tuned
   k=5 (0.640) and falls to 0.571 by k=60 — so the +9.8pp top-1 gain is real.

2. **1,740 groups is not 1,740 independent trials.** The groups draw on only
   678 distinct leaf / **180 bark** / 628 flower test images; each bark image
   is reused 9.7x on average (max 20x). A species-cluster bootstrap gives
   **fused top-1 = 0.738, 95% CI [0.680, 0.791] (±5.6pp)**, versus ±2.1pp if
   the groups are naively treated as independent. Per-species fused top-1 has
   median 0.800, with 21 species at 1.000 and 1 at 0.000.

3. **top-10 out of 87 species is a weak bar** — 11.5% of the label space, and
   the fused list averages 11.8 candidates, so 0.980 is close to measuring
   "the answer is somewhere in everything the system was willing to name."

4. **The groups are synthetic, and that is the load-bearing caveat for
   deployment.** `sample_groups` draws leaf/bark/flower *independently* within
   a species, so the evaluation assumes the three organ images are conditionally
   independent given species. A real user photographs one individual plant —
   same lighting, same day, same specimen — where errors are correlated. That
   could help (consistent conditions) or hurt (correlated failure); PlantNet-300K
   has no real multi-organ observations, so this dataset cannot tell us which.

**Quote `fused top-1 = 0.738 ± 5.6pp on synthetic groups`.** Reproduce all of
the above with `plantid.matching.fusion.evaluate_fusion` +
`CNN_DESCRIPTOR_CONFIG`.

## Implications

- **Drop the Phase 3 "query with flower alone" recommendation.** With CNN
  embeddings, always fuse all available organs — it's a strict improvement.
- **CNN embeddings + RRF fusion is now the matcher of choice**, replacing the
  classical descriptors for the matching pipeline. The classical descriptors
  (`features/leaf.py` etc.) and `outlier_scores.parquet` remain useful as the
  label-noise filter for the gallery (`build_matcher` still excludes flagged
  images).
- For a single default config, `ce_emb` is the simplest choice (one model,
  one embedding, no projection head needed at inference) and is competitive
  with `supcon_emb` everywhere except leaf/flower top1 — but since both
  models were exported, either is available.
- Phase 6 (CLI tool) should load `data/processed/cnn_models/mobilenet_v3_small_{ce,supcon}.pt`
  (or the `.onnx` exports for on-device deployment) to embed new images, then
  reuse `matching/classical.build_matcher(organ, variant=..., standardize=...)`
  + `matching/fusion.fuse_rankings` for the gallery search.
- Open question for later: would more epochs / tuned `lambda_supcon` /a
  larger backbone (e.g. `mobilenet_v3_large` or `efficientnet_b0`) push the
  classifier-head numbers (already 62% overall top-1) meaningfully higher,
  and would that translate to better k-NN/fusion numbers too? Not pursued
  yet — current numbers are already a step-change over Phase 3.
