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

## Single photograph, like for like

| system | label space | species top-1 | 95% CI | genus top-1 |
|---|---|---|---|---|
| **Pl@ntNet** | ~50,000 | **0.740** | [0.701, 0.781] | 0.882 |
| ours — Core ML int4 *(ships)* | 490 | 0.656 | [0.613, 0.697] | 0.847 |
| ours — BioCLIP v1 fp32 | 490 | 0.637 | [0.591, 0.680] | 0.852 |
| ours — BioCLIP-2 *(cannot ship)* | 490 | **0.785** | [0.746, 0.822] | 0.940 |

Paired over the same observations:

| | Δ vs Pl@ntNet | 95% CI | |
|---|---|---|---|
| Core ML int4 *(ships)* | **−0.084** | [−0.129, −0.039] | ✓ significant loss |
| BioCLIP v1 fp32 | −0.103 | [−0.148, −0.060] | ✓ significant loss |
| BioCLIP-2 *(cannot ship)* | **+0.045** | [+0.007, +0.084] | ✓ significant win |

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

**Twenty-six times fewer wrong answers.** That is the entire product thesis, and
it is now measured rather than asserted.

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

This is the first evidence that the product has a defensible niche, and it also
says precisely what that niche is. It is not "a better plant identifier." It is
"an identifier that is almost never confidently wrong," which is only valuable
where being wrong is expensive.

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
