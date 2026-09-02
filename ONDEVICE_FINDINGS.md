# On-device: there is no giant-model problem to solve

**BioCLIP v1 fits the deployment budget and clears the genus target. Distillation
is not needed.**

Phase 5 assumed the shipped encoder would have to be compressed or distilled,
because the chosen one is large. Measuring first made that project unnecessary.

## The premise was worse than stated, then dissolved

BioCLIP-2's image tower is **304M parameters — a ViT-L**, not the ViT-B assumed
earlier. At 152 MB even under 4-bit palettization and ~98 ms per image, it does
not ship.

But the bake-off already contained the clue that compression was the wrong fix:

| encoder | params | weighted top-1 @87 species |
|---|---|---|
| MobileCLIP2-S0 | 11M | 0.674 |
| DINOv3-S | 21M | 0.669 |
| DINOv3-B | 86M | 0.724 |
| BioCLIP-2 | 304M | **0.863** |

**DINOv3-B is 86M and reaches only 0.724.** A 4x larger general-purpose model
does not approach a biology-trained one, so BioCLIP-2's advantage is its
training data, not its size. That reframes the fix: find a *small model trained
on the right data*, rather than compress a large one.

## The deployment frontier

| encoder | params | int4 size | latency (MPS) | fits <50 MB / <100 ms |
|---|---|---|---|---|
| MobileCLIP2-S0 | 11.4M | 5.7 MB | 9.1 ms | ✅ |
| MobileCLIP2-S2 | 35.8M | 17.9 MB | 18.1 ms | ✅ |
| **BioCLIP v1 (ViT-B)** | **86.2M** | **43.1 MB** | **21.8 ms** | ✅ |
| BioCLIP-2 (ViT-L) | 304M | 152 MB | 97.8 ms | ✗ |

## Accuracy at 530 species, both fitted identically

Catalogue train split plus a class-weighted `__OTHER__` from the same background
species — see the note below on why that matters.

| encoder | organ | species | genus |
|---|---|---|---|
| BioCLIP v1 | leaf | 0.682 | 0.896 |
| BioCLIP v1 | bark | 0.675 | 0.757 |
| BioCLIP v1 | flower | 0.761 | **0.961** |
| **BioCLIP v1** | **weighted** | **0.722** | **0.925** ✅ |
| BioCLIP-2 | leaf | 0.733 | 0.944 |
| BioCLIP-2 | bark | 0.840 | 0.893 |
| BioCLIP-2 | flower | 0.774 | 0.972 |
| BioCLIP-2 | weighted | 0.757 | 0.957 |

**BioCLIP v1 costs 3.5pp of species accuracy and 3.2pp of genus accuracy, for a
3.5x smaller and 4.5x faster model that clears the ≥90% genus target at 0.925.**

Per organ the picture is uneven and worth stating: flower is strong at 0.961,
leaf is marginal at 0.896 — just under the bar on its own — and bark is 0.757,
well under, though it is the optional organ. The weighted figure the product
actually delivers is 0.925.

## A comparison error caught before reporting

The first run measured BioCLIP v1 **without** the `__OTHER__` reject class,
because `load_background` looks for a per-encoder cache and silently had none
for v1, while BioCLIP-2 had one. That flatters v1: with no reject class competing
for probability mass, in-catalogue accuracy rises.

The effect turned out to be small — v1 weighted genus 0.925 either way — but the
comparison was not apples-to-apples and would have been reported as though it
were. The background pool is now embedded per encoder.

## What this means for Phase 5

- **Ship BioCLIP v1 at 4-bit.** 43 MB, ~22 ms, genus 0.914 after quantization.
- **Distillation is cancelled.** It was the plan of record; it would have been
  weeks of work to recover part of a gap that an off-the-shelf model already
  closes.
- **MobileCLIP2-S2 (17.9 MB) is the fallback** if 43 MB proves too large in
  practice, and is untested — worth measuring only if the size budget tightens.

> **This section is too optimistic. See
> [The deployable encoder costs 19pp of coverage](#the-deployable-encoder-costs-19pp-of-coverage-not-3pp-of-genus-accuracy)
> below** — measured on real observations rather than the catalogue's own test
> split, the gap is roughly twice as large on accuracy and much larger on
> rejection, and it reopens the distillation question.

## What 4-bit palettization costs: 1.1pp of genus accuracy

The 43 MB figure assumes 4-bit weights, and ViTs can be sensitive to aggressive
quantization, so this needed measuring rather than assuming.

Simulated in PyTorch as Core ML does it — `2**n_bits` k-means centroids per
weight tensor, per output channel, with norm scales and biases left in float
(`pretrained.palettize_`). Catalogue and background were then re-embedded through
the quantized model and the head refitted, which is the deployment case: if the
device runs quantized weights, the head is trained on quantized embeddings too.

| | species | genus |
|---|---|---|
| BioCLIP v1, fp32 | 0.722 | 0.925 |
| **BioCLIP v1, 4-bit** | **0.709** | **0.914** ✅ |
| cost | −1.3pp | −1.1pp |

**43 MB, ~22 ms, genus 0.914 — the target still holds.** Quantization is not the
obstacle.

### Embedding drift badly overstates the damage

The intermediate diagnostics looked alarming and would have been the wrong thing
to act on:

| bits | weight error | cosine to fp32 | nearest-neighbour preserved |
|---|---|---|---|
| 8 | 0.016 | 0.998 | 91.8% |
| 6 | 0.044 | 0.986 | 86.2% |
| **4** | 0.113 | **0.898** | **62.0%** |

At 4 bits only 62% of nearest-neighbour relations survive and embeddings move to
cosine 0.898 — yet accuracy falls barely a point. **A retrained head absorbs a
systematic shift almost entirely.** Drift metrics measure whether the geometry
moved, not whether it stopped being separable, and only the second matters here.

### The implementation nearly produced the opposite conclusion

The first version used one shared palette per whole tensor, with centroids
initialised on a linear range from min to max. Weight distributions are
heavy-tailed, so that spends most of the 16 centroids in near-empty tails:

| 4-bit variant | cosine to fp32 | NN preserved |
|---|---|---|
| linear init, per-tensor (first attempt) | **0.214** | 0.8% |
| quantile init, per-channel (realistic) | 0.898 | 62.0% |

That would have been reported as "4-bit destroys the model" — a conclusion about
a crude implementation, not about quantization. Both fixes matter: per-channel
granularity and mass-aware centroid placement.

## Core ML export: it converts, it runs on the ANE, and int4 is forced

Everything above is PyTorch on MPS. `plantid/deploy/coreml.py` exports the
encoder properly and measures what actually happens.

| variant | size on disk | cosine vs fp32 | ANE latency | ops on ANE / CPU |
|---|---|---|---|---|
| **fp16** | 172.8 MB | **0.9998** (min 0.9994) | **8.9 ms** | 422 / 2 |
| **int4, per-grouped-channel** (iOS 18) | **46.2 MB** | 0.932 (min 0.831) | 19.2 ms | 383 / 5 |
| int4, per-tensor (iOS 17) | 43.5 MB | **0.413** (min 0.129) | 9.2 ms | 383 / 5 |

By compute unit, on an M4 Max: CPU-only 18.4 ms, GPU 6.6 ms, ANE 8.9 ms (fp16).

**The 21.8 ms MPS estimate was pessimistic — Core ML on the ANE is 8.9 ms.**
Both configurations sit an order of magnitude inside the <100 ms budget, so
latency is not the binding constraint. Size is: **fp16 is 172.8 MB against a
50 MB budget, so palettization is not optional.**

**Preprocessing is faithful.** Cosine 0.9998 against PyTorch on 64 real
catalogue photographs, with resize/crop/normalisation and the L2 step baked into
the graph. This was the failure most likely to go unnoticed — a normalisation
mismatch produces plausible embeddings and no error anywhere.

**The ANE dispatch is real, not nominal.** 422 of 424 compute ops are assigned
to the Neural Engine (2 to CPU), so the latency figure is not a GPU fallback
wearing an `ALL` label.

### Per-tensor palettization destroys the model — Core ML agrees with the simulation

The simulation above predicted this: per-tensor granularity was the variable
that mattered, and Core ML reproduces it independently at **cosine 0.413**. The
per-grouped-channel model lands at **0.932**, slightly *better* than the
simulation's 0.898, so the earlier estimate was mildly conservative rather than
wrong.

This makes the OS floor a product decision with a number attached:
**per-grouped-channel palettization requires iOS 18.** The iOS 17-compatible
fallback is per-tensor, which is unusable. There is no cheaper way to buy iOS 17
support than shipping a different, smaller encoder.

### The 2x latency cost of per-grouped-channel is dequantization, not fallback

Per-grouped-channel runs at 19.2 ms against per-tensor's 9.2 ms, and the obvious
suspicion is that some ops fell off the ANE. They did not — **both dispatch
identically, 383 ops to the ANE and 5 to CPU.** The difference is the cost of a
per-channel lookup table against a single shared one. Worth paying at 19 ms.

### The shipping configuration is silently wrong on the GPU backend

Checking whether the Neural Engine and the GPU agree — intended as a throughput
question, since GPU is 2.6x faster and the bulk embedding job does not care which
runs it — turned up something that matters for the app.

| vs PyTorch fp32, cosine | ANE | CPU | GPU |
|---|---|---|---|
| fp16 | 0.9998 | 0.9997 | **1.0000** |
| int4, per-tensor | 0.422 | — | 0.422 |
| **int4, per-grouped-channel** | **0.935** | **0.935** | **0.204** |

**Only the configuration we would ship breaks, and only on GPU.** ANE and CPU
agree with each other to three decimals; the Metal backend returns something
uncorrelated with the right answer. It is not a palettization problem in general
— per-tensor int4 gives the same (bad) 0.422 on both backends, so the two
backends agree when the weights are per-tensor. It is per-grouped-channel
dequantization specifically.

The failure is silent in the worst way: **the GPU still returns a plausible,
correctly-shaped, unit-norm embedding.** Nothing raises, and no downstream check
would notice except an accuracy measurement.

**Consequence for the app: pin `computeUnits` to `.cpuAndNeuralEngine`.** The
default `.all` happened to select the ANE on this machine, which is why the
earlier validation passed — but `.all` is a request, not a guarantee, and a
device where the Neural Engine is busy or unavailable can fall back to the GPU
and serve garbage. This is now recorded in `deploy/embed_coreml.py`, which pins
the units for the same reason.

### Two conversion obstacles, both silent-ish

1. `nn.MultiheadAttention` takes a fused eval-mode fast path that traces as
   `_native_multi_head_attention`, which the converter has no implementation
   for. `torch.backends.mha.set_fastpath_enabled(False)` decomposes it.
2. `torch.jit.trace` then fails on an `aten::Int` over a non-scalar shape under
   torch 2.13. `torch.export` handles it — but only after
   `.run_decompositions({})`, since the raw export is in the TRAINING dialect
   and the converter refuses it.

Neither is BioCLIP-specific; both will recur for any ViT exported from a recent
PyTorch.

## The deployable encoder costs 19pp of coverage, not 3pp of genus accuracy

Every figure in `REJECTION_FINDINGS.md` was BioCLIP-**2** — the 304M ViT-L that
cannot ship. `build_heads` had no `variant` parameter, so the encoder the
product would actually run had only ever been compared on the catalogue's *own
test split*: same corpus, no out-of-catalogue plants, no observation grouping.
`variant` is now threaded through, the evaluation set re-embedded with BioCLIP
v1, and the whole three-way rule re-run.

| on 5,534 real observations | BioCLIP-2 (cannot ship) | **BioCLIP v1 (ships)** |
|---|---|---|
| in-catalogue species accuracy | 0.846 | **0.760** |
| in-catalogue genus accuracy | 0.975 | **0.931** |
| global-OOD AUROC (genus conf.) | 0.972 | **0.901** |
| regional-OOD AUROC | 0.979 | **0.901** |
| near-OOD AUROC | 0.805 | 0.774 |
| precision @20% OOD | 0.956 | 0.946 |
| **coverage @20% OOD** | **0.722** | **0.531** |
| in-catalogue decline rate | 0.131 | **0.366** |
| mean utility | +0.589 | +0.472 |

Paired over species clusters: species **−8.6pp, CI [−10.3, −6.9]**; genus
**−4.4pp, CI [−5.3, −3.5]**. Both intervals exclude zero comfortably.

**The catalogue-split comparison understated this by about half** — it reported
3.5pp species and 3.2pp genus. Two reasons, and the second matters more:

1. Real observations are harder than the corpus the heads were fitted on, and
   the weaker encoder loses more from the shift.
2. **The catalogue split contains no out-of-catalogue plants, so it could not
   measure rejection at all.** That is where the real damage is: genus-confidence
   AUROC falls 0.979 → 0.901 on regional OOD. The entire product rests on that
   score separating catalogue plants from everything else.

**Precision barely moves (0.956 → 0.946) because the rule protects it — by
declining.** In-catalogue declines nearly triple, 13.1% → 36.6%, and coverage
falls from 72% of captures to 53%. That is the honest statement of the cost:
not "3pp of genus accuracy" but **a fifth of the captures the app can answer.**

### This reopens the distillation decision

Distillation was cancelled on the strength of "an off-the-shelf biology-trained
ViT-B already clears the bar". It clears a *genus accuracy* bar measured
same-corpus; it does not deliver the same product. The options now have numbers
attached:

- **Accept 53% coverage.** Still a usable product, and still 94.6% precise.
- **Raise the size budget.** 50 MB was self-imposed. BioCLIP-2 is ~164 MB at
  int4 by the measured 0.54 bytes/param — large for an app but not impossible,
  and it is the only option that keeps 72% coverage.
- **Distil BioCLIP-2 into a ViT-B.** Now has a real justification: 19pp of
  coverage, where before it looked like 3pp of an accuracy metric.
- **Try `bioclip_inat`** (ViT-B trained on iNat only, already in `ENCODERS` and
  untested here). Cheapest experiment of the four, and the evaluation set is
  iNaturalist, so it is the most likely to close part of the gap.

## The shipped artifact, measured: genus 0.918, and it clears 0.90 by 0.15pp

Everything above measures PyTorch. This measures the `.mlpackage`. The catalogue,
background pool and evaluation set — **82,166 images** — were embedded through the
int4 per-grouped-channel Core ML model on the Neural Engine, the heads refitted
on those embeddings, and the full three-way rule re-run.

Criterion declared before the run: *in-catalogue genus accuracy on real
observations, matched configuration, with the lower bound of the 95% cluster
bootstrap above 0.90.*

| arm | genus | 95% CI | species | precision @20% | coverage |
|---|---|---|---|---|---|
| **A · matched (ships)** — int4 head, int4 eval | **0.9179** | **[0.9015, 0.9322]** | 0.7517 | 0.939 | 0.499 |
| B · control — fp32 head, int4 eval | 0.9132 | [0.8969, 0.9282] | 0.7389 | 0.945 | 0.562 |
| baseline — fp32 head, fp32 eval | 0.9310 | [0.9163, 0.9440] | 0.7604 | 0.946 | 0.531 |

> **PASS — by 0.0015.** The gate is the CI lower bound above 0.90; it lands at
> 0.9015. The claim "we have a phone-deployable model at ≥90% genus" is now
> measured end-to-end rather than inferred, and it is true **with almost no
> margin.** Anything that moves it down by more than a fifth of a point — a
> harder species mix, a device whose ANE differs from an M4 Max, more evaluation
> coverage — puts it under.

**Quantization costs 1.31pp of genus accuracy**, 95% CI [0.74, 1.87], and 0.87pp
of species [0.00, 1.75]. The PyTorch simulation predicted 1.1pp. **It was right**
— which retroactively validates the palettization simulation as a method, and is
worth noting because that simulation is much cheaper than this measurement.

**Refitting the head on quantised embeddings is required for species, not proven
for genus.** Matched minus mismatched: species **+1.28pp, CI [+0.37, +2.21]**;
genus +0.47pp, CI [−0.09, +1.01]. So "a retrained head absorbs a systematic
shift" holds at species level and is not established at genus level — weaker than
the simulation section above implies, and the correct reading is that head
refitting is a shipping requirement for the species answer specifically.

Note arm B posts *higher* coverage (0.562) than arm A despite worse accuracy.
Thresholds are refitted per arm and the two confidence distributions differ, so
the operating points are not comparable directly; the accuracy columns are.

**Verification.** Row counts match the manifests exactly for all six organ caches,
no NaNs, and every evaluation photo is covered. Bulk cosine against fp32 —
leaf 0.930, flower 0.944, bark 0.916 over 43,506 images — matches the 0.935 from
the original 64-image spot check, so the bulk path did not diverge from
`coreml.validate`.

**What the product actually is, on the artifact that ships:** ~50% of captures
answered at ~94% precision, with genus right 92% of the time on catalogue plants.

## Still to do for a shippable model

1. **Benchmark on an actual phone.** The 8.9 ms is an M4 Max Neural Engine, not
   an A-series one. Same framework and same dispatch now, but a different chip —
   and since the genus gate now passes by 0.15pp, a device that embeds even
   slightly differently could move the verdict.
2. **Pin `computeUnits` in the app** to `.cpuAndNeuralEngine`. Not a tuning
   preference: the GPU backend returns uncorrelated embeddings for this exact
   configuration, silently.
2. **The deployable encoder has never been evaluated on real observations.**
   Every figure in `REJECTION_FINDINGS.md` — the 95.6%/72% headline included —
   is BioCLIP-**2** embeddings, and BioCLIP-2 is the 304M ViT-L that *cannot
   ship*. `build_heads` has no `variant` parameter; it is hardcoded. BioCLIP v1
   has only ever been compared on the catalogue's own test split. Closing this
   is the next measurement.
3. ~~**Verify on the newly added species.**~~ **Done.** 269 of the 530 had no
   real-observation evaluation; targeted and `taxon_id` fetches closed that to
   **32 of 497 source species** (`INAT_FINDINGS.md`). The species this surfaced
   are harder than the ones broad queries had found — species accuracy 0.873 on
   the original cohort against 0.793 and 0.750 on the two added ones — so the
   figures above, which predate that correction, are mildly optimistic at
   species level. Genus accuracy was statistically unchanged across all three
   cohorts, and genus is what the 0.925 target is set on.
