# Head-to-head with Pl@ntNet: we lose at naming, and win at not being wrong

Every competitive claim in this repo until now compared *their* published
aggregates against *our* measurements — different corpora, different label
spaces, different definitions of accuracy. This sends **the same photograph** to
both and scores them identically.

**465 observations, one per in-catalogue species**, so no two share a species and
the intervals need no cluster bootstrap. Pl@ntNet answered all 465 with zero
errors. Names are reconciled through synonym alias sets, because 9.5% of this
catalogue's names are a taxonomic generation behind (`INAT_FINDINGS.md`) and
string equality would measure nomenclature rather than identification — that
correction alone moved 22 of Pl@ntNet's answers from wrong to right.

## Three-way, same photograph: what we have ties the best incumbent, and it cannot ship

465 observations, one per in-catalogue species, the same single image to all
three. Zero errors after retrying iNaturalist's rate limit.

| system | label space | species top-1 | 95% CI | genus top-1 |
|---|---|---|---|---|
| **iNaturalist** (vision only) | 108,124 | **0.7871** | [0.748, 0.824] | 0.9140 |
| **ours — BioCLIP-2** *(cannot ship)* | 490 | **0.7849** | [0.746, 0.822] | **0.9398** |
| Pl@ntNet | ~50,000 | 0.7398 | [0.701, 0.781] | 0.8817 |
| ours — distilled student | 490 | 0.6753 | [0.632, 0.716] | 0.8430 |
| ours — Core ML int4 *(ships)* | 490 | 0.6559 | [0.613, 0.697] | 0.8473 |

Paired over the same observations:

| | Δ | 95% CI | |
|---|---|---|---|
| iNaturalist − Pl@ntNet | +0.047 | [+0.002, +0.088] | ✓ iNat is the stronger incumbent |
| **ours BioCLIP-2 − iNaturalist** | **−0.002** | **[−0.047, +0.043]** | **statistically tied** |
| ours int4 *(ships)* − iNaturalist | −0.131 | [−0.181, −0.084] | ✓ significant loss |

**The best configuration we have is indistinguishable from iNaturalist's server
model at species level, and ahead of it at genus (0.940 vs 0.914) — from a label
space 220x smaller.** It also cannot ship: it is the 304M ViT-L.

**What ships is 13.1pp behind.** So the entire competitive question reduces to
the encoder gap. Nothing else in the system is the problem — not the decision
rule, not the catalogue, not the fusion. The gap between 0.785 and 0.656 is the
whole distance between "competitive with the best free option" and "clearly
worse than two of them".

### Two categories, and only one of them is ours

The table above mixes systems that answer a different question. Split by whether
a network connection is required:

| | system | species top-1 | offline? |
|---|---|---|---|
| **server-side** | iNaturalist (108k taxa) | 0.7871 | ✗ |
| | Pl@ntNet (~50k) | 0.7398 | ✗ |
| **on-device** | **ours — BioCLIP-2, if the size budget rises** | **0.7849** | ✓ |
| | ours — Core ML int4, ships today | 0.6559 | ✓ |
| | iNaturalist Seek | ~0.66 on easier data, 2020 | ✓ |

**In the offline category the field is two systems: Seek and us.** Every number
that made this project look uncompetitive came from comparing it against
server-side services with unlimited compute and a permanent connection.

And the position within that category is strong. Our BioCLIP-2 configuration
scores **0.7849 — statistically tied with iNaturalist's *server* model** — which
would make the product claim *server-grade accuracy, offline*. Bridged through
Pl@ntNet as a common reference, Seek sits ~12pp further back (see below), and
what ships today at 0.6559 is already plausibly ahead of it.

That reframes the whole decision. It is not "can we beat Pl@ntNet" — we cannot,
with what ships. It is **"can BioCLIP-2 run on a phone", because if it can, the
offline category has a new best member by a wide margin.** That is a question
about app size and Core ML conversion, both answerable in an afternoon with the
pipeline already built.

### One caveat that runs in our favour

Our evaluation set is **iNaturalist research-grade observations, which is exactly
what iNaturalist trains its computer vision model on.** Their 0.787 is therefore
very likely inflated by having seen some of these photographs in training —
a far more direct contamination than BioCLIP-2's via TreeOfLife. Their published
vision-only figure is ~75%, and they score 78.7% here.

That makes the tie *understate* our position rather than overstate it: BioCLIP-2
matches a model that may have trained on the test images.

## Single photograph, like for like

| system | label space | species top-1 | 95% CI | genus top-1 |
|---|---|---|---|---|
| **Pl@ntNet** | ~50,000 | **0.740** | [0.701, 0.781] | 0.882 |
| ours — Core ML int4 *(ships)* | 490 | 0.656 | [0.613, 0.697] | 0.847 |
| ours — BioCLIP v1 fp32 | 490 | 0.637 | [0.591, 0.680] | 0.852 |
| ours — distilled student *(ships)* | 490 | 0.675 | [0.632, 0.716] | — |
| ours — BioCLIP-2 *(cannot ship)* | 490 | **0.785** | [0.746, 0.822] | 0.940 |

Paired over the same observations:

| | Δ vs Pl@ntNet | 95% CI | |
|---|---|---|---|
| Core ML int4 *(ships)* | **−0.084** | [−0.129, −0.039] | ✓ significant loss |
| BioCLIP v1 fp32 | −0.103 | [−0.148, −0.060] | ✓ significant loss |
| distilled student *(ships)* | −0.065 | [−0.110, −0.022] | ✓ significant loss |
| BioCLIP-2 *(cannot ship)* | **+0.045** | [+0.007, +0.084] | ✓ significant win |

Distillation was the attempt to move the first row up to the last. It did not:
the student lands at 0.675, still a significant loss, and is statistically
indistinguishable from BioCLIP v1 on the full evaluation
(`ONDEVICE_FINDINGS.md`). **Nothing that fits a phone beats Pl@ntNet at naming
plants.**

**The model that ships is significantly worse than Pl@ntNet at naming plants —
by 8.4 percentage points — while choosing from 100x fewer species.** Our
specialisation bought nothing on the identification task itself. Fusing all of an
observation's photos lifts the deployed model to 0.708, which is still below
Pl@ntNet's *single-photo* 0.740.

The approach is not the problem: BioCLIP-2 beats Pl@ntNet by 4.5pp. The
deployable version of it is the problem. That is now the second measurement
pointing at the encoder gap rather than anything else in the system.

They are also not strictly better — of 465, both are right on 263, only we are
right on 42, only Pl@ntNet on 81, both wrong on 79. There is complementary
signal, but they dominate it.

## The comparison that actually favours the product

Pl@ntNet always answers. We decline. On the identical 465 observations, at the
operating point already fitted in `REJECTION_FINDINGS.md`:

| | answers | correct when answering | **wrong answers per 100 captures** |
|---|---|---|---|
| Pl@ntNet | 100% | 0.740 | **26** |
| ours *(ships)* | 61% | **0.979** [0.961, 0.993] | **1** |

**Twenty-six times fewer wrong answers.**

> ### Retracted as a differentiator
>
> This was written as "the entire product thesis". It is not, and the test that
> kills it is one I should have run in the same sitting: **abstention is a
> technique, not a moat.** Applied to Pl@ntNet's own returned scores at matched
> coverage:
>
> | | coverage | precision | wrong per 100 |
> |---|---|---|---|
> | ours — three-way + reject class | 0.611 | 0.979 [0.961, 0.993] | 1.3 |
> | Pl@ntNet — naive species threshold | 0.611 | 0.919 [0.887, 0.951] | 3.1 |
> | **Pl@ntNet — through our exact cascade** | 0.613 | **0.996** [0.989, 1.000] | **0.2** |
>
> A single threshold on their top-1 score does *not* reproduce our behaviour —
> 0.919 against 0.979 — so the machinery is doing something real. But summing
> their scores by genus and running our two-threshold cascade beats us **six
> times over on wrong answers**, using nothing but data they already return.
> Their thresholds were tuned on this very data, so 0.996 is an upper bound; the
> direction is not in doubt.
>
> The reason is straightforward: the cascade needs calibrated scores over a
> label space, and theirs are calibrated over 50,000 species. We built the
> better decision rule and they have the better classifier to run it on. **A
> competitor could ship this in an afternoon.**

### State the cost honestly

Of our answers, 44% are at species and 56% at genus. So per 100 captures:

| | correct species names | correct genus-only | wrong | declined |
|---|---|---|---|---|
| Pl@ntNet | ~74 | — | ~26 | 0 |
| ours | ~26 | ~34 | ~1 | ~39 |

**Pl@ntNet delivers roughly 2.8x more correct species names than we do.** We
deliver roughly 1/26th as many errors. Neither is better in the abstract — the
answer depends entirely on what a wrong answer costs:

- **Curiosity** — "what's this flower?" — a wrong answer costs nothing and a
  ranked list is more useful than a decline. **Pl@ntNet wins, decisively.**
- **Consequence** — foraging, toxicity to a child or pet, an invasive species
  report — a confident wrong answer is the failure mode that matters.
  **Declining wins, and the 26:1 ratio is the size of the win.**

This is a real property of the system, and it is *not* a defensible niche —
see the retraction above. "Almost never confidently wrong" is valuable, and it
is also three hours of work for anyone holding a calibrated classifier.

## So what is actually left

Taking the differentiators in turn, against the evidence rather than against
intuition:

| claim | status |
|---|---|
| better identification | **false** — −8.4pp against Pl@ntNet, and distillation failed to fix it |
| calibrated abstention | **not defensible** — better on their scores than ours |
| genus fallback | **not defensible** — same cascade, same afternoon |
| runs offline / on-device | **true against Pl@ntNet** — its app needs a network. Contested against Seek; see below |
| small curated catalogue | a liability at identification; possibly an asset for a decision-specific product, untested |

**On the current evidence there is no demonstrated differentiator.** The honest
position is that this is a well-engineered system without a product case yet,
and that the case would have to come from a use context nobody here has
measured — one where the catalogue being 490 *curated* species is the point,
rather than an accident of image availability.

## Offline is a real edge over Pl@ntNet — and Seek already occupies it

Two corrections to the dismissal above, in opposite directions.

**Offline genuinely differentiates us from Pl@ntNet.** Their app requires a
network connection. Ours answers in ~9–19 ms on the Neural Engine with the radio
off — in a forest, abroad without roaming, in a basement greenhouse. That is a
real capability difference, not a marketing one.

*(A related point that is often confused: Pl@ntNet's 500/day limit is on the
**developer API**, not the consumer app, which is free, ad-supported and
unlimited. So it is not a consumer-facing weakness. It is, however, a strong
argument for owning the model rather than reselling their API — at
€0.005/identification, a product doing a million scans a month would pay
~€5,000/month for something our on-device model does at zero marginal cost.)*

**But the on-device niche is already occupied, and by something that does our
trick.** An independent peer-reviewed comparison of free plant-ID apps
([Hart et al. 2023, *Plants People Planet*](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1002/pan3.10460),
857 images) reports:

> iNaturalist Seek provided a binary classification on whether the user should
> have confidence in the identification. Of the 562 of 857 images (66%) where a
> confirmed identification was given, this was correct **in all cases**.

**Seek answers 66% of the time and was correct on every one.** That is calibrated
abstention, on-device, free, shipped — the thing this document called "the entire
product thesis" two days ago. Our comparable figures are 61% coverage at 0.979.
**The claim that nobody else ships on-device calibrated abstention is false.**

### Reading the paper properly changes the comparison

857 photographs of 277 species, taken by 16 ecological practitioners on their own
phones as "record shots", UK wild and naturalised plants with exotics
deliberately excluded, one photograph per plant. Five apps. Species top-1:
Pl@ntNet 86.6%, LeafSnap 86.9%, **Seek 66.0%**, Google Lens 57%, PlantSnap ~46%.

That test set is **much easier than ours** — and Pl@ntNet gives us the yardstick
to say by how much, since we measured it on our own images:

| | Pl@ntNet | the other system | gap |
|---|---|---|---|
| their set (UK, 277 spp) | 86.6 | Seek **66.0** | −20.6pp |
| our set (iNat, 465 spp) | 74.0 | ours, int4, 1 photo **65.6** | **−8.4pp** |

**Measured against a common reference, we sit 12.2pp closer to Pl@ntNet than
Seek does** — and Pl@ntNet scores 12.6pp lower on our images than on theirs,
which is the difficulty difference made explicit. At genus the two are level:
Seek is 4.0pp behind Pl@ntNet on their set, we are 3.5pp behind on ours.

So on species identification we appear to be *ahead* of Seek, not behind it. The
raw 66% vs 65.6% comparison is misleading because the test sets are not
comparable.

### Three things that cut the other way

1. **The study used apps as of 3 September 2020.** Seek's on-device model has
   been replaced repeatedly since — it was ~20k taxa then and a beta now carries
   ~80k. Today's Seek is very likely better than 66%, and this is the caveat that
   bites hardest.
2. **Seek was given location; we use none.** The authors entered county-level
   geolocation as metadata "to assist with automatic identification". On a
   UK-only test set that is a powerful prior — iNaturalist's own geomodel is
   worth +12pp of top-1 (`LOCATION_FINDINGS.md`). Part of Seek's 66% is
   geography, and none of our 65.6% is.
3. **Bridging two test sets through a third system is an assumption**, not a
   measurement. It assumes the difficulty gap scales similarly for both, which is
   plausible and unverified.

### What it is fair to conclude

Not "we beat Seek" — the bridge is too indirect and the model is six years stale.
But the earlier claim that Seek's existence settles the on-device question was
too strong in the other direction. **The honest position is that on-device
calibrated abstention is occupied territory, and how our system compares within
it is unresolved.** A direct measurement is worth an hour: install Seek, show it
50 photographs from our evaluation set, record the answer *and the rank it
answers at*.

## What has not been measured, and would change the picture

- **Seek's accuracy at species rank specifically, on our images.** The study
  above does not separate rank, and Seek's whole design is to answer coarser when
  unsure. If its 66% "confident" answers are mostly genus and family while ours
  are 44% species, the products differ more than the headline suggests. No API,
  so this needs manual comparison on a device — perhaps 50 photographs, an hour.
- **A decision-specific catalogue.** These 490 species were chosen by image
  availability, not by "plants where being wrong is expensive". Toxicity and
  edibility coverage is unknown.

## Caveats, in order of how much they bite

1. **Our catalogue was selected from PlantNet-300K** — every one of these 465
   species is one Pl@ntNet covers well, by construction. Favourable to them, and
   it means the comparison does not test their long tail at all.
2. **BioCLIP-2's win is confounded.** iNaturalist feeds GBIF feeds
   TreeOfLife-200M, so BioCLIP-2 may have trained on these exact photographs;
   Pl@ntNet almost certainly did not. Its +4.5pp is an upper bound.
3. **Thresholds were fitted on a calibration split containing about half these
   observations.** Fine for a descriptive comparison, but the 0.979 is not a
   clean out-of-sample figure.
4. **Single photograph is the fair comparison**; our product fuses several, which
   Pl@ntNet was not given here.
5. **iNaturalist is not yet measured** — its API needs a token. Given it scores
   ~75% vision-only across 108,124 taxa, expect it to land near Pl@ntNet.

## Reproduce

```bash
PLANTNET_API_KEY=... PYTHONPATH=. python -m plantid.eval.headtohead --n 465
```

Responses are cached per observation, so re-scoring costs no quota.
