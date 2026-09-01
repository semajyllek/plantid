# plantid v2 roadmap: on-device, ~250 species, Europe + North America

**Target**: an iOS app that identifies ~250 common European/North American plant
species entirely on-device, from a *guided* multi-photo capture, with an explicit
"not sure" rather than a confident wrong answer.

> **Target revised from ~1000 to ~250 species.** Trust matters more than
> coverage: a catalog users can rely on beats a larger one they can't. This makes
> **open-set rejection the core problem** — see [`OPENSET_FINDINGS.md`](OPENSET_FINDINGS.md),
> which shows ~250 species is both achievable from data already in hand and
> reaches 76% coverage at 95% precision. Phase 2's iNaturalist download is
> **deferred**: PlantNet-300K alone supplies 261 catalog species *and* 619
> background species for the reject class.

**Constraint**: all training compute is local Apple Silicon (MPS). No cloud GPU.

That constraint is less limiting than it sounds, and it dictates the central
architectural decision below.

---

## 0. The finding that reframes the target

The 87-species working set was **never a modelling choice — it was bark.**

Species reachable from PlantNet-300K by organ protocol (≥20 images/organ):

| protocol | species |
|---|---|
| flower only | 409 |
| leaf only | 382 |
| leaf **or** flower | 530 |
| leaf **and** flower | 261 |
| leaf **and** flower **and** bark | **13** |

At the ≥5 threshold, leaf+flower+bark = **87** — exactly the current working set.
Bark is 0.7% of PlantNet-300K (2,076 images); only 111 species have ≥5 bark
images and 3 have ≥50. External bark corpora don't rescue this: BarkNet ~23
species, Bark-101 ~101, Trunk12 ~12. **There is no dataset on Earth that supports
1000-species bark identification.**

Two consequences, both load-bearing:

1. **Drop bark as a required organ.** It becomes an optional bonus signal for
   the ~100 woody species where data exists. The capture UI must not require it.
1b. **Organ protocol (decided): leaf + flower required, bark optional.**
   Confirmed by the table above and by `HIERARCHY_FINDINGS.md`. Whole-plant
   "habit" shots were considered as a shape signal and rejected on the same
   grounds — 4,445 images, only 54 species with ≥20. The coarse shape
   information they would carry is recoverable from leaf/flower photos via
   genus-level output instead.
2. **PlantNet-300K alone cannot reach 1000 species** — it only contains 1,081,
   and only 530 have ≥20 images of leaf or flower. A second corpus is mandatory.

---

## 1. Architecture decision: frozen backbone + trained head

Local-only compute makes backbone fine-tuning impractical at 1000 species. The
answer is to not fine-tune backbones at all:

```
image → [frozen pretrained encoder] → embedding → [small trained head] → species
             (never trained;                        (trained locally in
              runs on ANE at inference)              seconds-to-minutes)
```

This is cheap *and* close to state of the art, because the best available plant
encoders are **already trained on plants** — a PlantCLEF-pretrained ViT or
BioCLIP has seen millions of plant images across tens of thousands of taxa. On
in-domain data, a linear/metric head over such features closes most of the gap
to full fine-tuning.

It also means embeddings are extracted **once** per image (pure inference,
batched on MPS) and every subsequent experiment — head architecture, species
count, fusion rule, calibration — is minutes of CPU on cached vectors. That is
the "don't waste compute" discipline made structural.

---

## 2. Metric: stop optimising top-1

"Extremely accurate" at 1000 species is not achievable as raw top-1 from one
casual photo, and chasing it would produce a worse product. The right target:

> **Precision at coverage, with an abstain option — and a genus level between
> them.** Delivered: at an assumed 20% out-of-catalogue rate, **95.6% precision
> on 72% of captures** — on BioCLIP-2, which *cannot ship*. On the deployable
> BioCLIP v1 it is **94.6% on 53%** (`ONDEVICE_FINDINGS.md`), which reopens the
> encoder question. Full figures below are the BioCLIP-2 ones:, on a curated 490-class catalogue and 5,534 real
> observations covering 465 of its 497 source species (`REJECTION_FINDINGS.md`,
> `CATALOG_FINDINGS.md`). The earlier "99.0% on 58%" was the λ=0.5/μ=2 row on
> a 261-species catalogue, before thresholds were anchored to a stated
> prevalence. Anchoring — plus a harder, twice-as-large species set — is what
> traded precision for coverage; raising `mu` to 4 pushed the other way
> (0.944→0.965 at a consistent 20% rate) and partly offset it.

The metric is now three-way rather than binary: the app answers at **species**,
answers at **genus**, or **declines**. Genus is a first-class answer, not a
hedge — genus accuracy (0.95–0.98) far exceeds species accuracy (0.76–0.83), and
species accuracy structurally caps what species-level answering can achieve, so
precision above ~88% *requires* the genus level. Thresholds are fitted by
expected-utility maximisation with utilities declared in advance
(`eval/rejection.py:UTILITY`), never by reading a threshold off the test set.

Precision is always reported as a curve over the assumed out-of-catalogue rate,
never as a scalar: it moves from 0.951 to 0.996 across a 60%→10% assumption on
the same model.

An app that says "I'm not confident — try a photo of the leaf" beats one that is
confidently wrong. This is also the only framing in which a defensible accuracy
claim can be made at 11.5x the current label space.

Report alongside it: top-1/top-5 (comparability), **macro**-averaged accuracy
(the long tail is severe — the top 10% of species hold 77% of images), and
coverage-vs-precision curves.

Deprecate the current headline: `fused top-10 = 0.980` is a candidate-list-length
artifact (see `CNN_FINDINGS.md`). The v1 87-species baseline was
`fused top-1 = 0.738 ± 5.6pp`; v2 supersedes it at 261 species
(`CATALOG_FINDINGS.md`, `REJECTION_FINDINGS.md`).

---

## Phase 0 — Housekeeping (30 min, no compute)

- `git init` commit. There are still **zero commits**; v2 must not be built on
  unversioned work.
- Rewrite `README.md`. It currently claims the system uses "classical computer
  vision and image-retrieval techniques rather than LLMs or heavy deep learning"
  and marks Phase 1 as current — both false. It describes a project this repo
  has already disproved.

## Phase 1 — Species list + protocol (no compute)

- Build the ~1000-species target list: intersect iNat21 / GBIF taxa with a
  Europe + N. America occurrence filter, rank by observation frequency, take the
  top ~1000 with ≥100 images available.
- Freeze the **organ protocol** for the capture UI: `flower` and `leaf` primary,
  `fruit` / `habit` / `whole plant` secondary, `bark` optional-woody-only.
- Deliverable: `species_v2.parquet` + a written protocol spec.

## Phase 2 — Data acquisition (bandwidth-bound, not compute-bound)

- **Primary source: iNaturalist (iNat21 / iNat Open Data on S3).**
  ~10k species, 2.7M images, Euro/US-heavy — the right corpus for this target.
- **Go/no-go to check first:** iNat observations bundle *multiple photos of the
  same individual plant*. If the distribution preserves observation grouping,
  it fixes the single biggest flaw in the current evaluation — every fusion
  number to date rests on **synthetic** groups that assume organ images are
  conditionally independent given species. Real observations give correlated,
  realistic groups. If the release flattens to one image per row, pull from the
  iNat API / GBIF export instead. **Verify before building on it.**
- Volume: 1000 species × ~200–300 images ≈ 200–300k images (~30–60 GB). Reuse
  the existing threaded downloader from `locate_crop_colab.ipynb` Step 2.
- **Split at observation level, never image level.** Multiple photos of one
  plant across train/test is the classic leak and would silently inflate
  everything.

## Phase 3 — Frozen-feature bake-off ⟵ *the decisive cheap gate*

Extract embeddings once per candidate encoder; probe with a linear head + k-NN.

Candidates (verify current availability on HF before pulling — releases shift):

| encoder | why |
|---|---|
| **PlantCLEF-pretrained ViT** | in-domain: trained on the plant corpus this task comes from. Likely winner. |
| **BioCLIP** | trained on TreeOfLife-10M (~450k taxa); strong few-shot biology features |
| **DINOv2 / v3** | best-in-class self-supervised features for fine-grained k-NN |
| **MobileCLIP** (Apple) | already on-device-shaped; the deployment fallback |
| **Apple `VNGenerateImageFeaturePrint`** | free, ANE-native, zero-install — worth one run as the floor |

**Run on the existing 87-species harness first.** `match_eval.evaluate_organ`
and `fusion.evaluate_fusion` already accept arbitrary cached descriptors through
`store.load_descriptors(organ, variant=...)`, so each candidate is a
`descriptors_{organ}_{variant}.npz` drop-in — hours, not days, and directly
comparable to every number in `CNN_FINDINGS.md`.

**Gate:** best frozen probe ≥ 0.62 top-1 on 87 species (the current *fine-tuned*
mobilenet number). If a frozen off-the-shelf encoder matches a model you trained,
the frozen path is confirmed and no backbone training happens for the rest of
the project. Expected to pass comfortably.

*(The `local-mps-model-setup` skill covers the recurring `trust_remote_code` /
CUDA-only-code failures these HF models hit on MPS.)*

## Phase 4 — Scale to 1000 species

- Same frozen embeddings, larger label space.
- **Measure the scaling curve**: 87 → 250 → 500 → 1000 species. This tells you
  whether 1000 is viable and where accuracy actually breaks, rather than
  discovering it at the end.
- Head bake-off on cached vectors (each trains in minutes on CPU): linear,
  cosine-classifier, **ArcFace/margin head** (SOTA for fine-grained retrieval,
  and gives a metric space that supports open-set rejection), k-NN.
- Deliverable: accuracy-vs-species-count curve + chosen head.

## Phase 5 — On-device deployment — **encoder chosen, distillation cancelled**

> See [`ONDEVICE_FINDINGS.md`](ONDEVICE_FINDINGS.md). **BioCLIP v1 (ViT-B, 86M)
> fits the budget at 43 MB / ~22 ms and clears the genus target at 0.925**, so
> the distillation project below is unnecessary.

BioCLIP-2's image tower turned out to be 304M parameters — a ViT-L, 152 MB even
at 4-bit — so it genuinely cannot ship. But compression was the wrong fix: the
bake-off showed DINOv3-B at 86M reaching only 0.724 against BioCLIP-2's 0.863,
so the advantage is *biological pretraining, not scale*. The answer was a small
model trained on the right data, and BioCLIP v1 is exactly that: 3.5x smaller,
4.5x faster, and 3.2pp of genus accuracy behind.

Remaining: Core ML export, real-device Neural Engine benchmarking, and measuring
what 4-bit palettization costs in accuracy.

- ~~Swap to the best deployable encoder and distill the big encoder into it~~ —
  unnecessary; an off-the-shelf biology-trained ViT-B already clears the bar.
- Core ML export → palettization / int8 quantization → **Neural Engine**
  benchmark on a real device.
- Budget: **<50 MB, <100 ms/image on ANE.** The head is negligible
  (1000 × 512 ≈ 512k params).

## Phase 6 — Guided capture UX ⟵ *where the crop lesson pays off*

The LocateAnything experiment failed because "a single flower" is ill-posed for
a *found* photo of a dense Sedum mat — only ~26–33% of images yielded a usable
crop (`CROP_FINDINGS.md`). **Guided capture inverts the problem: you control the
frame, so a well-posed subject exists by construction.** This is the correct
response to that failure, and it is the user-facing half of the accuracy story.

- **Subject isolation**: `VNGenerateForegroundInstanceMaskRequest` (iOS 17+,
  subject lifting) — on-device, fast, no VLM. This is what LocateAnything was
  standing in for, done properly.
- **Live quality gates** before the shutter: Laplacian-variance blur, subject
  area fraction, exposure, and a "multiple subjects detected" check (directly
  addressing the conflated-plants problem).
- **Coaching**: "move closer to the leaf", "hold steady", "too dark",
  "more than one plant in frame".
- **Organ prompting**: request leaf, then flower, then optional extras — each
  validated on capture.

## Phase 7 — Fusion, calibration, abstain

- **Real multi-organ groups** from iNat observations (Phase 2) — retires the
  synthetic-groups caveat.
- **Score-level fusion, not RRF.** Phase 3 found RRF is rank-shaped: with
  `RRF_K=60`, `1/(60+rank)` varies only ~20% across candidates, so it behaves
  like "count organs that agree" and cannot defer to a decisive organ. With
  *calibrated probabilities* you can combine log-odds properly.
- **Temperature scaling** on the head → then **conformal prediction** for
  set-valued output with a coverage guarantee ("the species is in this set with
  95% probability"). This is exactly the abstain machinery the UX needs.
- Deliverable: the precision-vs-coverage curve that defines the product claim.

## Phase 8 — Geographic prior — **MEASURED AND REJECTED as a gate**

> Closed. See [`LOCATION_FINDINGS.md`](LOCATION_FINDINGS.md). This was billed
> below as "likely the single highest accuracy-per-effort item in the entire
> plan". It is not, in the form tested.

Device location narrows the label space 2.7x in European cities and up to 9.5x
in North American ones, and the signal is genuinely biological — shuffling
coordinates within a bucket collapses it from 0.715 AUROC to 0.511. It adds
+0.023 AUROC on near-OOD, the weakest case, and is partly independent of the
vision scores (r = 0.33 against genus confidence).

But as a **gate** — withholding or generalising an answer, never renaming, per
the product decision — it buys **+0.0069 expected utility, 95% CI
[−0.0025, +0.0220]**. The interval includes zero and the pre-registered rule
says do not ship. It fails because the vision thresholds are already
conservative enough that the gate only changes 0.3–2.9% of decisions: the AUROC
gain lands where the operating point never looks.

**Within-genus re-ranking was then tested and also rejected.** Letting location
choose among congeners while the photograph fixes the genus looked worth +3.5pp
on 373 observations; on 1,150 it is **+0.6pp, 95% CI [−0.011, +0.022]**, with
the fix-to-break ratio falling from 20:7 to 28:21. The first measurement was
small-sample noise.

What remains untested: unconstrained re-ranking (ruled out on product grounds),
prevalence-weighting the prior, and seasonality.

Seasonality (date) was never tested and remains unexplored.

---

## Sequencing and what to do first

Phases 0–3 are the critical path and cost almost nothing. Phase 3's gate decides
the architecture for everything after it; do not start Phase 2's bulk download
before Phase 3 has run on the existing 87 species, because the winning encoder
determines what you need to store.

**Immediate next actions:**
1. Phase 0 — commit, rewrite README (30 min).
2. Phase 3 on the existing 87-species data — no new downloads required, reuses
   the current harness end to end. This is the highest-information step
   available and it runs locally today.
3. Phase 2 go/no-go — confirm iNat observation grouping survives in the
   distribution you can actually download.

## Known risks

- **iNat label noise** is real (community IDs vary in quality); filter to
  "research grade". Expect a Phase-2-style outlier pass to be needed again.
- **Long tail**: PlantNet's top 10% of species hold 77% of images; iNat is
  similar. Macro accuracy will lag top-1 badly. Budget for class-balanced
  sampling and report macro numbers.
- **Frozen features may plateau** below the target. Fallback is parameter-
  efficient fine-tuning (LoRA / attentive probing) on a small backbone, which
  is feasible on MPS — not full fine-tuning.
- **iOS Visual Look Up already identifies plants.** The differentiators must be
  accuracy at regional depth, guided multi-organ capture, offline operation,
  and calibrated abstention. Worth benchmarking against directly.
