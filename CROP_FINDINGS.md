# Organ-isolation crops (LocateAnything-3B): findings

`notebooks/locate_crop_colab.ipynb` tested whether cropping each photo to a
grounded bounding box around the target organ — removing background clutter —
would (a) make classical shape descriptors meaningful and (b) improve the CNN.

Run config: `MAX_PER_GROUP=20`, `MIN_AREA_FRAC=0.15`, CE-only, 15 epochs,
prompts `a single leaf` / `the trunk of the plant` / `a single flower`.

**Verdict: the hypothesis is not supported, but the accuracy tables do not
establish that — they are triple-confounded. The real finding is mechanistic:
LocateAnything does not isolate an organ on this dataset, and for most of this
corpus the request is ill-posed.**

## The mechanism finding (this is the durable result)

`found_rate = 1.0` for all three organs — the model returns a box every time.
But the box is usually not an isolation. The `area_frac` distribution is
strongly **bimodal**: a spike at ~1.0 (box ≈ the whole frame, so nothing was
removed) and, for leaf/flower, a second spike at ~0 (a tiny fragment, dropped
by the `MIN_AREA_FRAC` filter). Very little mass in the middle, which is the
only region where cropping does useful work.

Exact, computed from `data/processed/organ_crops.parquet` (4,326 grounded rows):

| organ  | grounded | box ≥95% of frame | below `MIN_AREA_FRAC=0.15` | **mid-range 0.15–0.95** | kept |
|--------|----------|-------------------|-----------------------------|--------------------------|------|
| leaf   | 1,730    | 45.5%             | 28.3%                       | **26.2%**                | 1,241 |
| bark   | 937      | 59.8%             | 11.1%                       | **29.1%**                | 833   |
| flower | 1,659    | 23.3%             | 43.7%                       | **33.0%**                | 934   |

The mid-range column is the only one where cropping does anything: a box that
is neither the whole frame nor a discarded fragment. It is roughly a quarter to
a third of grounded images.

**The consequence for the experiment that was actually run:** among the images
that survived `MIN_AREA_FRAC` and were fed to the crop-trained CNN, the fraction
whose box covers ≥95% of the frame is **63.4% (leaf) / 67.2% (bark) / 41.3%
(flower)**. For leaf and bark, roughly two-thirds of the "crop" training set is
the original uncropped image. The treatment arm was mostly not treated.

The cause is visible in the Step 7/14 grids: much of this corpus is
**mat-forming, compound, or whole-habit subjects** — dense *Sedum* flower mats,
*Humulus* cone tangles, whole shrubs, potted-plant shots. For those, "a single
flower" has no well-posed answer. The correct box is either one floret
(`area_frac`→0, filtered out) or the entire mat (`area_frac`→1, no
decluttering). That is exactly the observed bimodality.

This is a **task-specification problem, not a prompt-tuning problem**. Revisiting
Step 4 with different prompt wording will not create a single-organ box where
the subject genuinely has no isolable single organ. Bark is the extreme case at
~60% whole-frame boxes — bark close-ups already fill the frame, so cropping is
close to a no-op there by construction.

### Consequence for the Phase 6 input-validation idea

`good_crop_rate_among_detections` was proposed as a "not enough organ visible —
get closer" gate. As measured it is unusable: flower 0.563, leaf 0.717, bark
0.889. A gate at `MIN_AREA_FRAC=0.15` would reject **44% of legitimate flower
photos**. The bimodality is the reason — a rejected photo is usually a dense
inflorescence, not a bad photo. Any such gate needs a different signal.

## Accuracy tables — confounded, reported for completeness

| organ | clf crop | clf whole (base) | k-NN crop | k-NN whole (base) |
|--------|---------|------------------|-----------|-------------------|
| leaf   | 0.305 | 0.581 | 0.252 | 0.517 |
| bark   | 0.284 | 0.459 | 0.187 | 0.317 |
| flower | 0.473 | 0.716 | 0.355 | 0.679 |

Everything is down 17–36pp, but **three things changed at once**, and they do
not all push the same direction:

1. **Training data shrank** (`MAX_PER_GROUP=20` + the area filter): leaf 0.25x,
   flower 0.20x, bark 0.73x of the baseline's train set. Deflates crop numbers.
   Both crop models overfit hard (train acc 0.89–0.97 vs val 0.35–0.42),
   which is the signature of this.
2. **The test set changed.** `MIN_AREA_FRAC` filters *all* splits, so the crop
   models were scored on n=131/134/110 versus the baseline's 755/183/686 — only
   **17% / 73% / 16%** of the original test sets. That subpopulation is images
   where the organ fills ≥15% of frame, i.e. closer-up, better-framed photos,
   which are plausibly *easier*. Inflates crop numbers.
3. **The input changed** (the variable of interest).

Net direction unknown. `BASELINE_CLF_CE` / `BASELINE_KNN_CE` are hardcoded from
a different run on a different population; they are not a valid comparison here.

Bark makes the problem concrete: 0.73x the training data but −17.5pp. That is
too large a drop for 27% less data, so either cropping genuinely hurts bark
(which contradicts the crops≈whole-image picture above) or the test population
shifted materially. This run cannot distinguish the two.

## Step 8 go/no-go (Hu moments on crops): inconclusive, not a clear negative

| organ  | hu-on-crop top1/5/10 | whole-image classical | chance |
|--------|----------------------|------------------------|--------|
| leaf   | 0.031 / 0.076 / 0.137 | 0.065 / 0.183 / 0.283 | 0.011 / 0.057 / 0.115 |
| bark   | 0.015 / 0.060 / 0.119 | 0.098 / 0.257 / 0.383 | " |
| flower | 0.009 / 0.118 / 0.227 | 0.156 / 0.348 / 0.472 | " |

Tempting to read flower top-1 = 0.009 as "at chance" — but that is **1 correct
out of 110**, 95% CI [0.000, 0.050]. The same organ's top-5 (0.118, CI
[0.065, 0.194]) and top-10 (0.227, CI [0.153, 0.317]) are both ~2x chance and
exclude it. So shape-on-crop still carries *some* signal; the top-1 number is
noise at this sample size.

The gallery is also ~7.5 examples/class (654 flower crops / 87 species) versus
~37/class for the classical baseline. A 7-dim Hu descriptor degrading sharply
at that density is expected independent of cropping. **Not a clean answer
either way.**

## The one result that is matched on data: crop + whole-image dual input

Steps 12b–12d trained a dual-input model (crop *and* the uncropped image,
`WHOLE_IMAGE_WEIGHT=0.5`) on the **same subset, same splits, same epochs** as
the crop-only model:

| organ  | crop-only | crop + whole | Δ |
|--------|-----------|--------------|---|
| leaf   | 0.305 | 0.351 | **+4.6pp** |
| bark   | 0.284 | 0.254 | −3.0pp |
| flower | 0.473 | 0.518 | **+4.5pp** |
| val acc | 0.349 | 0.419 | **+7.0pp** |

Adding the whole image back **helps on leaf and flower** — the crop is
discarding information the model wants. It does not help on bark, consistent
with bark crops being ≈ the original image already (nothing new in the second
input). The sign pattern matches the mechanism story exactly.

Caveat: the dual model has ~2x the parameters and sees 2x the pixels, so this
bounds the crop penalty rather than measuring it. "Restoring whole-image
context recovers ~4.5pp" is defensible; "cropping costs exactly 4.5pp" is not.

## Experiments considered — item 1 superseded, see the Conclusion

> **Read this box first. Items 1–3 below were written before
> `organ_crops.parquet` was inspected locally, and item 1 is now
> superseded — do not run the matched control. It cannot answer the question
> at this sample size.** Item 3 is already answered (below). Items 4–5 are
> still live and are carried into the Conclusion. The whole section is kept
> for the record of what was considered and why it was dropped.
>
> Two numbers kill it. (a) 63% / 67% / 41% of the crop arm's images are
> already the whole frame, so on most of the data the two arms receive an
> *identical* input and the delta is diluted toward zero by construction.
> (b) Restricting to the mid-range stratum where the treatment is real leaves
> **n_test = 38 (leaf) / 44 (bark) / 59 (flower)** — a 95% CI half-width of
> ±12–15pp. Any effect smaller than ~15–20pp is invisible.
>
> So the run costs ~10 min of GPU and returns a number that is uninterpretable
> whichever way it lands. The cells below are left in place in case the design
> is revisited at larger scale, but item 1 is **not** recommended as-is.
>
> Item 3 has already been answered locally, and it is a non-issue: zero-training-
> crop species are 0 (leaf) / 0 (bark) / 1 (flower), affecting 1 of 110 flower
> test images (0.9%). Coverage is not a contributor to the accuracy drop.

Steps 15b–15e have been added to `notebooks/locate_crop_colab.ipynb` for items
1–3. They sit between Step 15 and the Step 16 export, and the export cell now
carries their results into `metadata.json` (each field is omitted rather than
erroring if the cell wasn't run).

1. **The matched control (cheap, settles it) — Steps 15d/15e.** Retrain CE on
   the **same 3,008 image_ids and splits, uncropped**, via
   `run_training_generic(WholeImagePlantDataset)` — Step 10's `run_training`
   with the dataset class parameterized, so architecture/seed/epochs/LR/
   augmentation are all held fixed. One variable, matched n, matched test
   population. ~10 min on an A100, no VLM. Run this before concluding anything
   about cropping.
2. **Free ablation on the already-trained dual model — Step 15c.**
   `topk_accuracy_dual` takes `whole_image_weight` at eval time; the existing
   model is scored at w = 0 / 0.25 / 0.5 / 0.75 / 1.0. One model, one test set,
   no retraining. (Trained at w=0.5, so other weights are off-distribution for
   the head — directional only.) **If the previous Colab runtime is still
   alive, paste this cell in and it runs immediately.**
3. **Per-species crop coverage — Step 15b.** CPU-only, needs only Step 9. At
   934 flower crops over 87 species, some species could have **zero** training
   crops while remaining in the 87-way softmax and in test — mechanically
   unlearnable, and an artifactual accuracy ceiling. The cell reports that
   ceiling per organ. (A random-subsample simulation locally suggests coverage
   holds up — min 1–7 train crops/species — but the real filter is area-based,
   not random, so it may concentrate differently.)
4. **Classical descriptors on crops — worth it, but run matched and locally.**
   Step 8 only tested Hu moments; LBP+HSV carried most of the classical signal
   in Phase 2. Recompute the full `features/{leaf,bark,flower}.py` descriptors on
   crops *versus the same image_ids uncropped*, ~5 min CPU, zero GPU.
5. **Skip the imret-on-crops retest.** It was near-chance on whole images in
   Phase 3, and ~half the crops *are* the whole image — there is no plausible
   path to a different answer.

If the matched control in (1) shows cropping is neutral-to-harmful — which the
bimodality predicts — the crop line of work closes, and the open lever from
`CNN_FINDINGS.md` (larger backbone, more epochs, tuned SupCon) is the better
use of GPU time. Current best remains whole-image CE embeddings + RRF fusion at
top-1 0.738 / top-10 0.980.

## Conclusion

**The crop line is closed.** Not because cropping was measured and found
harmful — it wasn't cleanly measured at all — but because the grounding step
does not produce the intervention the experiment was designed to test. On
~26–33% of images the box is a genuine crop; on the rest it is either the whole
frame or a fragment too small to keep. No amount of downstream training fixes
that, and the sample sizes that survive stratification (n_test 38/44/59) cannot
resolve a plausible effect.

The one design that *could* answer it is grounding the full 10,777 images and
keeping only the mid-range stratum (~3,200 images, ~30% yield), then training
crop and whole arms on that. That is ~10,777 3B-VLM forward passes to test a
hypothesis whose upside is capped: even a win would apply only to the third of
photos where grounding produces a usable box, requiring a fallback path for the
rest. Given the whole-image CNN pipeline already reaches fused top-1 0.738 /
top-10 0.980, that is not the best available use of GPU time.

**Recommended instead:** the open lever from `CNN_FINDINGS.md` — a larger
backbone (`mobilenet_v3_large`, `efficientnet_b0`), more epochs, on the full
dataset. `mobilenet_v3_small` at 15 epochs is a small model trained briefly,
and top-10 is already near saturation, so top-1 is the number with room.

Two items from the superseded list remain valid:

- **Optional, ~5 min CPU, zero GPU:** recompute the full Phase-2 classical
  descriptors (Hu + LBP + HSV, not just Step 8's Hu) on crops vs. the same
  `image_id`s uncropped, using the local `organ_crops.parquet`. Worth knowing,
  but it inherits the same n_test 38/44/59 stratification ceiling, so treat it
  as a curiosity rather than a decision input.
- **Skip the imret-on-crops retest.** Near-chance on whole images in Phase 3,
  and ~63% of the crops *are* the whole image — there is no path to a different
  answer.

## Artifacts preserved

From `locate_crop_export.zip`, now in the repo (all gitignored under `data/`):

- `data/processed/organ_crops.parquet` — 4,326 bounding boxes with `area_frac`.
  Reusable independent of this conclusion; the analysis above is reproducible
  from it.
- `data/processed/descriptors_{organ}_ce_crop_emb.npz` — crop-trained CE
  embeddings, same format as the Phase 2 / CNN descriptors.
- `data/processed/cnn_models/mobilenet_v3_small_ce_crop.{pt,onnx}` and
  `metadata_ce_crop.json`.
