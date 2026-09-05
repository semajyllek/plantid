# `plantid plan | build | card`

Turn a chosen species list into a small offline model, and say honestly what it
will and will not do.

```bash
plantid plan  --species my.txt --budget 20
plantid build --images ./photos --background-images ./other --out models/mine
plantid card  models/mine
```

(`PYTHONPATH=. .venv/bin/python -m plantid.tool.cli ...` until it is packaged.)

## It takes a dataset; it does not fetch one

Three ways in, and nothing here knows what a plant is:

| flag | shape |
|---|---|
| `--images DIR` | `DIR/<label>/*.jpg` |
| `--manifest FILE` | parquet/csv with `label`, `path` [, `group`, `cluster`] |
| `--embeddings FILE` | npz with `descriptor`, `label` [, `group`, `cluster`] |

`--background-*` takes the same three forms and supplies negatives. Without it
there is no reject class: the model is closed-set, cannot decline, and the card
says so rather than implying a rejection capability that was never fitted.

**`cluster` is the unit that must not straddle a split** — several photographs of
one plant, one specimen, one production run. Supply it whenever the data has that
structure. Without it every row is treated as independent and the card records
that its intervals are anticonservative.

**`group`** is the coarse rank the cascade falls back to. It defaults to the
first whitespace-delimited token of the label, which is exactly right for
Linnaean binomials and often right elsewhere; override it with a column.

Labels need not be binomials. A list that is *wholly* `Genus species` gets the
Linnaean join key applied; anything else passes through as written.

Fetching from iNaturalist, GBIF or anywhere else deliberately lives outside the
tool — see `analysis/` — because the choice of corpus, its licensing and its
taxonomy are domain decisions.

## Why `plan` comes first

Fetching, training and exporting are commodity — a dozen tools do them. What
none of them do is tell you what you are about to get. `EMBEDDED_FINDINGS.md`
shows why that matters: **the composition of a species list decides which
failure mode it has, and the failure mode is invisible in the headline metrics.**

Two lists of fourteen species from the same catalogue, same encoder:

| | genus-crowded (8 *Sedum*, 6 *Trifolium*) | well-separated (14 genera) |
|---|---|---|
| coverage | **80.6%** | 61.8% |
| precision | 97.8% | 98.5% |
| species-level share | **47.6%** | **76.1%** |
| closed-set top-1 | 81.7% | 97.0% |

The crowded list answers a third more queries at the same precision and is much
worse. It buys the coverage with genus answers that narrow nothing — "it is a
*Sedum*" when eight of your fourteen species are *Sedum* — and gets 15pp less
accurate when it does name a species. Report coverage alone and a user reads
their worst case as their best.

So `plan` runs before any compute, and every report carries the species-level
share.

## What each command does

**`plan`** — no training, no downloads, seconds. Analyses the list for crowded
genera and for relatives left *outside* it (the weakest rejection case, since no
correct label exists for them), picks an encoder for the byte budget, and
projects coverage / precision / species-share by interpolating the measured grid
for the domain (`profiles/`), or declines to project if there is none.

**`build`** — fits a logistic head over frozen embeddings, fits the
species/genus/decline thresholds from `eval/rejection.py` by expected-utility
maximisation on a clustered calibration split, evaluates on the held-out half
against three buckets (on-list, relatives you did not choose, unrelated), and
writes a bundle plus a card.

**`card`** — renders the bundle's manifest as a model card.

Personalisation is the head, not the encoder: at 20 species and 512 dimensions
that is ~40 KB against a 17.9 MB encoder, which is why this is CPU-seconds
rather than GPU-hours.

## Projection is not measurement

`plan` interpolates a grid measured on a *different* catalogue; `build` measures
the user's actual data. They disagree, and by design the card supersedes the
plan. Observed on the two lists above, the projection erred **in both
directions**: 33% projected against 47.6% measured (+15pp) on the crowded list,
84% against 76.1% (−8pp) on the varied one. So `plan` is an estimate with wide
error, not a bound — good enough to decide whether to proceed, not good enough
to report.

`build`'s own numbers carry cluster-bootstrapped intervals over **species**, not
rows, because six-odd observations share a species. At fourteen species those
intervals are wide (coverage 42–70% on the varied list), and that width is a
fact about the list rather than a presentation choice.

`plan --ood-rate` is restricted to the rates actually measured (0.5 / 0.2 / 0.1);
`build` accepts any value because it fits on your data.

### Projection is gated behind a measured profile

`profiles/plants-bioclip2.json` is a grid measured on 530 plant species through
BioCLIP-2. There is no fallback: ask for a domain without a profile and
projection raises rather than quoting plant numbers at you.

`BIRDS_FINDINGS.md` shows the coarse-answer trap reproducing on birds — species
level 0.958 → 0.718 for a genus-crowded set that scored *higher* on coverage.
That licenses `plan`'s structural warnings in any domain with a label/group
hierarchy. It does not license the numbers.

## Encoders

Image tower only — the text tower never ships. int4 is the default assumed
precision because it measures as indistinguishable from fp32 at every catalogue
size tested.

| encoder | params | int4 | notes |
|---|---|---|---|
| MobileCLIP2-S0 | 11.4 M | 5.7 MB | microcontroller-class |
| MobileCLIP2-S2 | 35.8 M | 17.9 MB | 1.6pp behind BioCLIP-2 at 10 species, but fragile |
| PlantCLEF2024 | 86.6 M | 43.3 MB | ≈BioCLIP-2, *safer* on hazards, but 38.6 ms/image |
| BioCLIP-2 | 304.0 M | 152 MB | reconciles with the shipped 160 MB build; 20.4 ms |

A phone is **not** a constrained target: 152 MB fits comfortably and is the
fastest option. **Byte order is not speed order** — PlantCLEF2024 is a third of
BioCLIP-2's parameters and nearly twice its latency, because it runs at 518px. So
`choose` ranks storage only, and a latency budget needs `--encoder` passed by
hand. The small encoders matter below ~20 MB — Coral, microcontrollers, cheap
drone payloads — and `CONTAMINATION_FINDINGS.md` shows they are also the ones
that degrade under any distribution change.

## Limits worth knowing before you build

- **Narrowing does not fix a confusable list.** Going 50 → 10 species buys ~4pp
  on a congener-dense set. The tool warns; it cannot solve.
- **The reference pool is a floor.** Relatives are checked against the local
  catalogue's 499 binomials, so a genus whose relatives it never included will
  look safer than it is. Widening it needs only a taxonomy, not images —
  `data/resolve_taxa.py` already resolves names against iNaturalist (475 of 497),
  but it is fuzzy matching rather than a GBIF/POWO backbone, which is what a pool
  of thousands would need.
- **`plan`'s reference pool is the local catalogue.** `build` accepts any source,
  but the composition warnings still check relatives against the bundled 499
  binomials, so `plan` is most useful for plants until a broader taxonomy is
  wired in.
- **Nothing here is verification.** Do not eat anything on the basis of a model
  this builds.
