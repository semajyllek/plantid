# Small encoders on narrow catalogues — the size/accuracy frontier

**Question.** A community tool that trains a per-user model over K chosen species
for a small device. Does the penalty for a small encoder shrink as K shrinks?

**Answer: yes, monotonically — if the chosen species are unrelated. If they are
congeners, narrowing the catalogue buys almost nothing, for any encoder.**

## Method

530-species catalogue, frozen encoders, logistic head over L2-normalised
embeddings with a `__OTHER__` reject class trained on the background pool
(`class_weight="balanced"`, mirroring `inat_fusion.build_heads`). 15 draws per
cell. Organs: leaf + flower; bark excluded (1,311 rows over 530 species is too
thin to subset), so top-1 here is **not** comparable to the bark-inclusive
numbers in `BAKEOFF_FINDINGS.md`.

Two arms, reported separately and never averaged, because random subsets are
mostly cross-genus and flatter the small encoder:

- **EASY** — K species drawn uniformly.
- **HARD** — whole genera drawn until K is reached, so congeners stay together.

Three test populations: in-set (subset test rows), **near-OOD** (test rows of the
catalogue species *not* chosen — the case that actually bites a narrow
catalogue), **far-OOD** (held-out background rows).

Two corrections made before the numbers were trusted, both of which moved them:

1. `background_*.npz` still contains catalogue species (documented at
   `embed_background.py:50`). Loading it raw labelled subset species `__OTHER__`
   *and* by name — contradictory supervision on the classes under test. Now goes
   through `load_background(exclude_species=catalog_species())`.
2. `C` is swept per encoder on val, not fixed. In the 87-species pilot the sweep
   alone was worth 5pp to MobileCLIP2-S0 — enough to be mistaken for a capacity
   gap.

## Size

Image tower only; the text tower does not ship.

| encoder | params | int8 | int4 |
|---|---|---|---|
| BioCLIP-2 | 304.0 M | 304 MB | **152 MB** |
| MobileCLIP2-S2 | 35.8 M | 35.8 MB | **17.9 MB** |
| MobileCLIP2-S0 | 11.4 M | 11.4 MB | **5.7 MB** |

BioCLIP-2 int4 at 152 MB reconciles with the shipped 160 MB build.

## Closed-set top-1

**EASY** — the gap to BioCLIP-2 shrinks monotonically as the catalogue narrows.

| K | BioCLIP-2 | S2 | Δ | S0 | Δ |
|---|---|---|---|---|---|
| 10 | 0.9888 | 0.9730 | **1.6pp** | 0.9546 | **3.4pp** |
| 20 | 0.9790 | 0.9493 | 3.0pp | 0.9290 | 5.0pp |
| 50 | 0.9386 | 0.8990 | 4.0pp | 0.8666 | 7.2pp |

An 87-species pilot on the descriptor set put the S0 gap at 14.3pp, consistent
with continued growth, but it is **not a fourth point on this curve**: different
species pool, bark included, and a k-NN rather than swept-C head.

### The trend is not a background-ratio artifact

The species:background ratio in the fit shifts ~5x between K=10 and K=50, since
the reject class draws on a fixed pool (2,329 training rows after excluding
catalogue species). That could produce a spurious K-dependence. Re-running K=10
with background capped proportionally — halving it to 1,000 rows, from
background-dominated to roughly balanced — barely moves the gaps:

| K=10 easy | full background | proportional background |
|---|---|---|
| S2 Δ | 1.6pp | 1.2pp |
| S0 Δ | 3.4pp | 3.5pp |

The gap does not widen when the confound is removed. The K-dependence is
encoder capacity, not fit composition. (The cap does not bind at K=50, where the
pool is already the smaller term.)

**HARD** — the same ordering, but note the BioCLIP-2 column barely moves.

| K | BioCLIP-2 | S2 | Δ | S0 | Δ |
|---|---|---|---|---|---|
| 10 | 0.7524 | 0.7159 | 3.7pp | 0.6858 | 6.7pp |
| 20 | 0.7478 | 0.6996 | 4.8pp | 0.6802 | 6.8pp |
| 50 | 0.7100 | 0.6584 | 5.2pp | 0.6217 | 8.8pp |

**Congener discrimination is nearly independent of K.** 0.710 → 0.752 going from
50 species to 10. Telling ten *Sedums* apart is almost as hard as telling fifty
apart, because the difficulty is pairwise rather than a function of label-space
size. Narrowing the catalogue is not a remedy for a confusable catalogue.

## Rejection (AUROC, score = 1 − P(`__OTHER__`))

| K | arm | BioCLIP-2 near | S2 near | S0 near |
|---|---|---|---|---|
| 10 | easy | 0.9682 | 0.9447 | 0.9322 |
| 20 | easy | 0.9494 | 0.9217 | 0.9030 |
| 50 | easy | 0.9261 | 0.8699 | 0.8414 |
| 10 | hard | 0.9952 | 0.9883 | 0.9812 |
| 20 | hard | 0.9870 | 0.9748 | 0.9556 |
| 50 | hard | 0.9840 | 0.9627 | 0.9404 |

**The two difficulty axes are anti-correlated.** Rejection is *better* on the
hard arm. Choosing congeners absorbs the confusable species into the catalogue,
leaving an easy boundary; choosing unrelated species leaves every congener
outside it, where it becomes a confident false positive.

So set composition selects the failure mode:

- **congener-dense set** → hard to tell apart (0.75), easy to reject (0.99)
- **unrelated set** → easy to tell apart (0.98), harder to reject (0.95)

A tool that trains on a user-chosen set should inspect that set and say which
regime the user is in.

## int4 costs nothing

`bioclip2_cml4` is statistically indistinguishable from fp32 at all three K and
both arms — measured, not assumed:

| K | arm | fp32 | int4 |
|---|---|---|---|
| 10 | easy | 0.9888 | 0.9898 |
| 20 | easy | 0.9790 | 0.9777 |
| 50 | easy | 0.9386 | 0.9408 |
| 10 | hard | 0.7524 | 0.7476 |
| 20 | hard | 0.7478 | 0.7504 |
| 50 | hard | 0.7100 | 0.7086 |

Consistent with the existing finding, and it means the 152 MB build is the one to
quote, not the 304 MB one.

## Deployable coverage (`analysis/subset_coverage.py`)

AUROC ranks encoders but does not say what a deployment answers. Fitting the
three-way cascade from `eval/rejection.py` — thresholds by expected-utility
maximisation on a calibration split, `UTILITY` as declared, prevalence anchored
by `deployment_weights` — gives the deployable numbers. K=20, 12 draws.

**coverage / precision / species-level share of in-catalogue answers**

| arm | encoder | p_ood=0.5 | **p_ood=0.2** | p_ood=0.1 |
|---|---|---|---|---|
| EASY | BioCLIP-2 | 0.528/0.937/0.649 | **0.820/0.963/0.849** | 0.908/0.971/0.928 |
| EASY | S2 | 0.497/0.901/0.484 | **0.771/0.934/0.833** | 0.862/0.951/0.891 |
| EASY | S0 | 0.444/0.893/0.474 | **0.736/0.922/0.802** | 0.830/0.946/0.840 |
| HARD | BioCLIP-2 | 0.686/0.857/0.412 | **0.872/0.945/0.412** | 0.933/0.967/0.412 |
| HARD | S2 | 0.665/0.855/0.305 | **0.856/0.941/0.305** | 0.919/0.961/0.305 |
| HARD | S0 | 0.683/0.826/0.307 | **0.856/0.929/0.307** | 0.912/0.954/0.307 |

At the project's standing 20% anchor, a 20-species catalogue answers **82% of
queries at 96.3% precision**, with 85% of in-catalogue answers at species level.
MobileCLIP2-S2 at 17.9 MB gives 0.771/0.934, about 5pp of coverage and 3pp of
precision behind a model 8.5x its size.

> Do **not** read 0.820 against the 0.722 coverage in `CLAUDE.md`. That figure is
> measured on 5,534 iNaturalist observations with bark included and
> near/distant/regional OOD buckets; this one is catalogue test rows, leaf and
> flower only, with near-OOD defined as same-genus catalogue species. Same metric
> name, different populations — the comparison `headtohead.py` exists to prevent.

### The HARD arm's good numbers are an artifact — read the third column

On the congener arm coverage looks *better* (0.872 vs 0.820) and precision is
comparable. It is not a better deployment. The species-level share collapses to
**0.41 for BioCLIP-2 and 0.31 for both small encoders**: the cascade is buying
coverage with genus answers, and on a catalogue where every species is a *Sedum*,
"it is a Sedum" carries no information. `UTILITY` scores a genus answer at 0.5
regardless of how much the genus narrows the field, so the fit takes that trade
happily.

This is the same anti-correlation seen in the AUROC table, now with a price
attached: congener-dense catalogues do not fail by answering wrongly, they fail
by answering vacuously. **Any report to a user must carry the species-level
share, or a congener-dense set will look like the best case rather than the
worst.**

Note also that on the HARD arm the encoder gap nearly vanishes (S0 0.856/0.929
vs BioCLIP-2 0.872/0.945) — but only because both are falling back to the same
vacuous genus answer. The 6.8pp closed-set gap is real and is being hidden.

### Why the HARD species rate is invariant to prevalence

The species-level share is identical to three decimals across p_ood 0.5/0.2/0.1
while coverage moves 0.686 → 0.933. That is not a grid artifact — the fitted
thresholds are genuinely constant, and `t_species = 0.9065` is an interior
optimum, not the top of `s_grid` (max species confidence is 0.9903). On the EASY
arm the same fit *does* respond to prevalence (t_genus 0.8513 → 0.3973,
t_species 0.8579 → 0.6708).

The mechanism: on a congener-dense catalogue the near-OOD bucket is *by
construction* species from genera you chose, so a genus answer is correct for
in-catalogue and near-OOD observations alike. Genus answering therefore earns
utility no matter what the out-of-catalogue rate is, `t_genus` sits near the
floor, and declining is never worth it. Coverage still moves with p_ood because
the bucket *weights* change, not the thresholds.

So the 0.41 / 0.31 species rates are the fitted operating point for this utility
function on this kind of catalogue, not a property of the encoders.

## The frontier is optimistic for small encoders on real field photographs

Every number above is measured on the PlantNet-derived catalogue. Repeating the
K=20 easy cell on **iNaturalist photographs of 20 common, family-diverse Oregon
plants** — 4,251 images, 2,964 observations, split by observation — gives a very
different answer:

| encoder | PlantNet catalogue, K=20 easy | 20 common Oregon plants, iNat | change |
|---|---|---|---|
| BioCLIP-2 | 0.9790 | **0.9769** | −0.2pp |
| MobileCLIP2-S2 | 0.9493 | **0.8557** | **−9.4pp** |
| gap | 3.0pp | **12.1pp** | 4x |

**BioCLIP-2 transfers to real field photography and MobileCLIP2-S2 does not.**
PlantNet images are curated and plant-centred; iNaturalist observations are what
people actually take — variable framing, habitat context, whole-plant shots, bark.
The weaker encoder loses most of its catalogue-measured performance to that shift
while the stronger one loses none of it.

So the size/accuracy frontier above should be read as **an upper bound for the
small encoders specifically**. It is not wrong — it is measured on the data it
names — but it does not survive a change of image source, and a user's photographs
resemble iNaturalist far more than they resemble PlantNet.

Part of BioCLIP-2's stability here is that iNaturalist is in its training data via
GBIF, so this overstates its robustness too. The practical conclusion is
unaffected: at 20 species on realistic photographs, 17.9 MB buys 0.856 and 152 MB
buys 0.977.

## The middle ground exists, and it is not a distilled model

Distilling BioCLIP-2 into a ViT-B was tried and failed (`CLAUDE.md`, closed list):
`bioclip1_distil` *lost* 0.7pp of genus against the off-the-shelf BioCLIP v1 it
was meant to beat, recovering ~10% of the gap. Compressing the large encoder does
not work.

Picking a different one does. **PlantCLEF2024** — DINOv2 ViT-B/14 fine-tuned on
7,806 Pl@ntNet species, already in `features/pretrained.py` — was rejected in
`BAKEOFF_FINDINGS.md` as a *benchmark* because 71.3% of that test set was inside
its training data. That is the right call for a benchmark and says nothing about
its use as a *deployment encoder*, which had never been evaluated.

On Oregon iNaturalist photographs, split by observation:

| encoder | int4 | ms/image | 20 common Oregon | 27-way safety set | hemlock named edible @100% | @75% |
|---|---|---|---|---|---|---|
| BioCLIP-2 | 152.0 MB | 20.4 | **0.9769** | **0.9743** | 0.067 | 0.008 |
| **PlantCLEF2024** | **43.3 MB** | 38.6 | 0.9621 | 0.9670 | **0.042** | **0.000** |
| MobileCLIP2-S2 | 17.9 MB | 5.2 | 0.8557 | 0.8184 | 0.283 | 0.167 |

**At 3.5x fewer bytes it is 1.5pp behind on accuracy and *ahead* on safety** —
zero hemlock-named-edible at 75% coverage, where BioCLIP-2 still has 0.8%.

Two things make this less surprising than it looks. It is trained on plants
specifically rather than all of biology, and it runs at 518px — 5.3x the pixels —
which is exactly the resolution advantage you would expect to matter for
distinguishing umbels.

### The cost is latency, not size

**38.6 ms/image against BioCLIP-2's 20.4** (MPS, batch 8, M4 Max — indicative,
not ANE). A third of the parameters and nearly twice the time, because of the
input resolution. So:

- **storage-constrained, compute available** → PlantCLEF2024
- **phone** → BioCLIP-2; 152 MB fits and it is twice as fast
- **both constrained** → neither, and S2 is not safe at 0.283

Byte order is no longer speed order, so `encoders.choose` ranks storage only and
says so. A caller with a latency budget must pass `--encoder` explicitly.

### Contamination cuts the other way here

BioCLIP-2 trains on iNaturalist via GBIF and these *are* iNaturalist images.
PlantCLEF2024 trains on Pl@ntNet images of overlapping *taxa*. So BioCLIP-2's
numbers are inflated by image-level exposure while PlantCLEF2024's are inflated
only by taxon-level exposure — the weaker form. The true gap is plausibly smaller
than the 1.5pp measured, in PlantCLEF2024's favour. Neither is clean; only field
photographs are.

## Not yet measured

- **Fine-tuning.** Everything here is frozen. The frontier may make it
  unnecessary: S2 at 17.9 MB is 1.6pp behind BioCLIP-2 at K=10.
- **A genus-informativeness term in `UTILITY`.** The current constant 0.5 is what
  makes the HARD arm degenerate. Scoring a genus answer by how much it narrows
  the chosen set would fix it, and is a change to a declared utility — so it
  needs writing down before it is fitted, not after.
- **The congener warning.** The mechanism predicts near-OOD AUROC degrades with
  the count of outside-set congeners; regressing per-draw AUROC on that count
  would turn "expect false positives" into a named list.
- Subsets are drawn from a catalogue selected by image availability, median 77
  images/species. A user pulling their own species from iNaturalist would likely
  have more data. Treat as indicative.
