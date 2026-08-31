# Hierarchical output: answer at the level you're confident about

Tests three related ideas: making leaf+flower required and bark optional;
whole-plant shape as a signal; and coarse-then-fine classification.

All on cached BioCLIP-2 embeddings, 87 species / 53 genera, seconds of CPU.

## 1. Organ protocol: leaf + flower required, bark optional — confirmed

Settled by data (`ROADMAP.md` §0): bark is 0.7% of PlantNet-300K, and only 17
species have ≥20 bark images versus 409 for flower and 382 for leaf. Bark
cannot be required at any useful catalog size.

It stays as an *opportunistic* input because it does earn its keep
in-distribution — leaf+flower open-set AUROC 0.919 → 0.941 with bark added
(`OPENSET_FINDINGS.md`) — but its cross-source rejection failure means the
product's guarantees must not depend on it.

## 2. Whole-plant shape ("a maple looks different from a pine")

The intuition is right, but as a *separate organ* it is data-starved. PlantNet
organ availability:

| organ | images | species ≥5 | species ≥20 |
|---|---|---|---|
| flower | 176,531 | 704 | 409 |
| leaf | 110,784 | 652 | 382 |
| fruit | 11,277 | 169 | 72 |
| **habit** (whole plant) | **4,445** | **201** | **54** |
| bark | 2,076 | 111 | 17 |

`habit` is thinner than fruit and only marginally better than bark. It cannot
be a required capture step.

**But the insight survives in a better form.** "Maple vs pine" is a *coarse*
distinction, and coarse distinctions are exactly what §3 captures — and they
are already recoverable from ordinary leaf and flower photos, without asking
the user for a whole-plant shot. Note also that v1 tested explicit shape
descriptors (Hu moments) and they failed badly (`PHASE2/3_FINDINGS.md`) — the
shape signal is better learned by the encoder than hand-engineered.

## 3. Coarse-then-fine — the strong idea, with a twist

### Genus-level accuracy is far higher than species-level

| organ | species top-1 | **genus top-1** | gain |
|---|---|---|---|
| leaf | 0.862 | **0.974** | +0.111 |
| bark | 0.738 | **0.820** | +0.082 |
| flower | 0.910 | **0.980** | +0.070 |

A genus-level answer on leaf or flower is right ~97–98% of the time.

### Graceful degradation: the key result for trust

For species the model has **never seen**, but whose genus has a congener in the
catalog, the correct genus is still predicted:

| organ | n | correct genus for an *unseen species* |
|---|---|---|
| leaf | 155 | 0.787 |
| bark | 40 | 0.650 |
| **flower** | 144 | **0.875** |

**This is a better answer to the open-set problem than rejection is.** Rather
than "I don't know", the app can say *"this is a Sedum — I'm not sure which
one"*, and be right ~88% of the time on a species it has never encountered.
That is more useful to the user *and* more often correct than a binary decline.

### Two-stage pipeline: no accuracy gain, so don't build it

Predicting genus then species within the predicted genus:

| organ | flat species | two-stage | oracle-genus ceiling |
|---|---|---|---|
| leaf | 0.862 | 0.861 | 0.883 |
| bark | 0.738 | 0.732 | 0.896 |
| flower | 0.910 | 0.910 | 0.930 |

Two-stage matches flat but never beats it — genus errors propagate, and the
oracle row shows even a *perfect* genus stage would add only ~2pp on
leaf/flower. **The value of hierarchy is output structure, not accuracy.**

### Implementation: marginalize, don't train a second head

Deriving genus by summing the flat species posterior within each genus matches
a dedicated genus classifier:

| organ | dedicated genus head | **marginalized from flat head** |
|---|---|---|
| leaf | 0.974 | 0.976 |
| bark | 0.820 | 0.809 |
| flower | 0.980 | 0.978 |

And marginalized genus confidence is a slightly *better* rejection signal than
species confidence on the strong organs:

| organ | OOD AUROC, species conf | OOD AUROC, genus conf |
|---|---|---|
| leaf | 0.872 | **0.882** |
| flower | 0.907 | **0.917** |
| bark | **0.865** | 0.854 |

## Recommended design

**One flat species head. Everything else falls out of it:**

- **species answer** = argmax of the species posterior
- **genus answer** = marginalize the same posterior over genera (~97–98% correct)
- **rejection signal** = marginalized genus confidence (better than species conf
  on leaf and flower)
- **UI behaviour** = report the most specific level clearing its confidence
  threshold: species if confident, else genus, else decline

No second model, no pipeline, no extra training. The coarse level is a free
by-product of a model we already have — and it converts the open-set failure
mode from "wrong species, confidently" into "right genus, honestly hedged".

### Open questions

- **Family level** would extend the ladder (species → genus → family) and should
  degrade even more gracefully. Needs a genus→family taxonomy table; POWO or
  World Flora Online (`DATA_STRATEGY.md`).
- **Does graceful degradation survive cross-source?** All numbers here are
  same-corpus, and that assumption has already been falsified once for bark
  (`OPENSET_FINDINGS.md`). Genus-level transfer must be tested against
  BarkVN-50 / Oxford Flowers before it is relied on.
- The 87-species set has 53 genera, so many genera are singletons. Genus-level
  advantage may look different at 261 species, where genera are better populated.
