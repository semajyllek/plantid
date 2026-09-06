# plantid — orientation for a new session

On-device plant identification for Europe / North America. A **490-class curated
catalogue** over frozen BioCLIP embeddings, with a three-way answer: name the
species, name the genus, or decline.

## Three repos, and which is which

| repo | is | note |
|---|---|---|
| **plantid** (here) | the research record | 28 findings/prereg docs, 129 tests. Evidence only. |
| [**narrowcast**](https://github.com/semajyllek/narrowcast) | the tool, pip-installable | domain-general, 65 tests, CI. `fit / plan / build / card / encoders` |
| [**narrowcast-kws**](https://github.com/semajyllek/narrowcast-kws) | audio demo | consumes the *installed package*; ESC-50 + Speech Commands |

narrowcast deliberately cannot read this project's catalogue caches — choosing a
corpus and reconciling its taxonomy are domain decisions. The seam is
`analysis/export_for_narrowcast.py`, which writes catalogue vectors in the
`--embeddings` format.

Read this first, then `ROADMAP.md` for the plan and the `*_FINDINGS.md` docs for
evidence. The newest four are `HEADROOM_FINDINGS`, `EMBEDDED_FINDINGS`,
`OREGON_SAFETY_FINDINGS` and `CONTAMINATION_FINDINGS`. **Git history is the chronological record** — commit messages carry the
reasoning, the numbers, and the retractions.

## The one thing that is easy to get wrong

**Every accuracy number is encoder-specific, and the encoders differ by a lot.**
Never quote a figure without naming which. On 5,534 real iNaturalist
observations:

| variant | what it is | genus | species | coverage @20% OOD |
|---|---|---|---|---|
| `bioclip2` | ViT-L fp32, reference only | 0.9747 | 0.8460 | 0.722 |
| **`bioclip2_cml4`** | **Core ML int4, 160 MB — deployable** | **0.9735** | **0.8370** | **0.692** |
| `bioclip1` | ViT-B fp32 | 0.9310 | 0.7604 | 0.531 |
| `bioclip1_cml4` | Core ML int4, 46 MB | 0.9179 | 0.7517 | 0.499 |
| `plantclef24` | ViT-B/14 @518, 43 MB — **the middle ground** | see below | see below | — |
| `bioclip1_distil` | distilled from BioCLIP-2 — **failed** | 0.9243 | 0.7706 | 0.550 |
| `bioclip_inat` | iNat-only ViT-B — **eliminated** | 0.9115 | 0.7438 | 0.515 |

Run anything with `--variant`, e.g.
`PYTHONPATH=. python -m plantid.eval.rejection --variant bioclip2_cml4`.

## Where it stands

Measured against the incumbents on **identical photographs**, 465 observations,
one per species (`COMPETITIVE_FINDINGS.md`):

| system | species top-1 | offline? |
|---|---|---|
| iNaturalist server (108k taxa) | 0.7871 | ✗ |
| **ours, `bioclip2_cml4`, 160 MB** | **0.7720** | ✓ |
| Pl@ntNet (~50k) | 0.7398 | ✗ |
| ours, `bioclip1_cml4`, 46 MB | 0.6559 | ✓ |

The 160 MB build is **statistically tied with iNaturalist's server model**
(paired −0.015, CI [−0.060, +0.030]) and ahead at genus (0.974 vs 0.914). That is
the product claim: *server-grade accuracy, offline*.

The open decision is app size: **160 MB and competitive, or 46 MB and 13pp
behind.** This is a product judgement, not a research question — but it is not
the only thing that could be done next, and the list under "Open" is not
exhaustive.

## Conventions that make the numbers trustworthy

These exist because things failed without them. Follow them.

- **Cluster bootstrap, never row-level.** Resample *species* (or *genera* for the
  near-OOD bucket), because ~6 observations share a species. Row-level intervals
  have twice produced effects here that failed to replicate.
- **Background rows cluster on their real species, not on `__OTHER__`.** They
  *score* as `__OTHER__`, but giving them that as a clustering identity leaves
  `make_splits` one cluster for `distant_ood`, puts every negative in test, and
  fits thresholds on a calibration set with no negatives in it. This silently
  broke the deployable-coverage table in `EMBEDDED_FINDINGS.md` (retracted in
  place) and would have broken `HEADROOM_FINDINGS.md`. Same trap in
  `deployment_weights`: an *absent* bucket leaves its share unclaimed and the
  whole weighting renormalises to a lower effective prevalence.
- **Declare utilities before fitting.** Thresholds are fitted by expected-utility
  maximisation with payoffs written down in advance (`eval/rejection.py:UTILITY`),
  never read off a test set.
- **Anchor to a stated prevalence.** `deployment_weights` fixes the assumed
  out-of-catalogue rate at 20%, so the evaluation set's incidental composition
  cannot choose the operating point. Adding data once moved `t_species` from
  0.897 to 0.552 and produced a completely different product.
- **The contamination caveat is now measured, not assumed.** iNaturalist is in
  BioCLIP-2's training data, and that qualified every number here as an upper
  bound. Tested against images uploaded after its training cutoff with a matched
  control: **−0.27pp, CI [−0.88, +0.35]** (`CONTAMINATION_FINDINGS.md`). Under
  1pp. Do not re-hedge on it. *Domain* shift — a different camera in different
  hands — is still untested and is a separate thing.
- **Cosine does not predict accuracy.** Checked three times: it under-predicted
  the cost of int4 (0.932 → −1.3pp), wildly over-predicted distillation (0.956 →
  nothing), and was right once (0.982 → −0.1pp). It bounds how much *could* have
  changed and says nothing about whether what changed mattered. **Measure the
  head.**
- **Retract in place.** When a claim dies, strike it through and record the
  measurement that killed it rather than deleting it. Several docs contain
  retractions of their own headline claims; that is deliberate.
- **Name reconciliation is required for any external comparison.** 9.5% of
  catalogue names are a taxonomic generation behind, so string equality measures
  nomenclature rather than identification. Use
  `eval/headtohead.py:alias_sets()` / `correct()`.

## Practical

**Two virtualenvs, and it matters which.**
`.venv` (py3.14) runs the tests and pandas work; it has **no torch**.
`.venv-mps` (py3.12) has torch, coremltools, open_clip — anything touching a
model. Tests: `PYTHONPATH=. .venv/bin/python -m pytest -q` (129 pass, 2 skip).

**`data/processed/` is gitignored and local-only** (12 GB): images, embedding
caches per encoder, and `headtohead/` holding 1,394 cached API responses
(465 Pl@ntNet, 465 iNaturalist, 464 iNaturalist+geo). Re-scoring
the competitor comparison costs **no API quota** — the responses are on disk.

**A real Core ML trap.** int4 per-grouped-channel is silently wrong on the Metal
GPU backend — cosine 0.204 for v1, 0.628 for BioCLIP-2, correct on ANE and CPU,
and it returns a plausible unit-norm embedding while being wrong. The iOS app
**must** pin `computeUnits` to `.cpuAndNeuralEngine`; `.all` is a request, not a
guarantee.

**`analysis/`** holds the scripts that produced the committed results — encoder
comparisons, ablations, the head-to-head. They reproduce claims in the findings
docs.

**Published write-ups** (private artifacts):
[experimental record](https://claude.ai/code/artifact/e1115322-7231-4726-956e-1ef9391cb0f0) ·
[project status](https://claude.ai/code/artifact/28638d03-1b2f-4956-be5e-7675a57d37e0) ·
[what the data looks like](https://claude.ai/code/artifact/e4a5ddab-765e-42c0-9891-320e1f198e99)

## Closed — do not redo these

- **Distillation.** Ran on an A100, 48k transfer images, held-out cosine 0.956 to
  the teacher. Recovered ~10% of the encoder gap; statistically indistinguishable
  from the encoder it was meant to beat.
- **`bioclip_inat`.** Worse on both corpora. Breadth of biological pretraining
  beats matching the deployment distribution.
- **Geographic prior**, for *this* catalogue. Worth ~0 at 490 regional species
  and **+4.5pp to iNaturalist's 108k-taxa model on the same photographs** — the
  value scales with label-space size. Re-open only if the catalogue grows a lot.
- **Abstention as a moat.** Our own cascade applied to Pl@ntNet's returned scores
  beats us. It is a technique, not a differentiator.
- **Background pool size.** 59 species performs as well as 589 on every rejection
  metric.
- **Cropping, ORB retrieval, classical descriptors.** All measured and lost.
- **Distillation, reconfirmed.** The middle ground between 17.9 MB and 152 MB is
  **not** a compressed BioCLIP-2 — it is `plantclef24`, an off-the-shelf ViT-B at
  43 MB that is 1.5pp behind on accuracy and *ahead* on hazard safety. It costs
  38.6 ms/image against BioCLIP-2's 20.4, because it runs at 518px. Byte order is
  not speed order.
- **Pruning, to reach a small budget** (`PRUNE_FINDINGS.md`). Depth-pruned
  BioCLIP-2 at 127 MB scores 0.9323; off-the-shelf `plantclef24` at 43 MB scores
  0.9621. Every pruned point is dominated. 99% of ViT-L is its 24 blocks, so
  reaching 17.9 MB needs ~3 of them. **Selection beats compression** — spend a
  GPU on evaluating breadth, not on compressing.
- **Reading an earlier layer** (`LAYER_FINDINGS.md`). Perception Encoder reports
  mid-stack embeddings beat the output for contrastive encoders. Not here: every
  intermediate layer is worse, monotonically, and concatenation adds +0.0000.
  BioCLIP-2's objective already matches the task.

## The finding the tool is built on

A label set crowded with siblings of one group buys **coverage** with coarse
answers that narrow nothing, so the headline metrics move the reassuring way
while the model gets worse. Replicated across domains and modalities:

| crowded vs varied | Δcoverage | Δlabel-level |
|---|---|---|
| plants (image) | +0.188 | **−0.285** |
| birds (image) | +0.018 | **−0.240** |
| text, 20 Newsgroups | +0.216 | **−0.326** |
| ESC-50 (audio) | −0.095 | −0.107 |
| Speech Commands (audio) | −0.027 | −0.077 |

**It does not always fire, and the governing quantity is now measured.**
*Headroom* — coarse-rank accuracy minus fine-rank accuracy — predicts retreat to
the group rank at **CV R² 0.883** over 1,409 arms, against 0.362 for fine
accuracy alone (`HEADROOM_FINDINGS.md`). Was a hypothesis at n=4; is now a
finding. As a rule of thumb, **group-answer share ≈ 1.8 × headroom**, measurable
on the calibration split.

The controlled comparison is the thing to remember: one fitted head, fine
accuracy pinned at 0.659, only the *group column* varying — group answers move
0.022 → 0.417 and coverage 0.200 → 0.500. The grouping alone moves the product.

Two honest qualifications. Headroom is *nearly* but not exactly the quantity
(`b + c = −0.038`, CI excludes zero — fine accuracy weighs ~19% more), and below
the 0.889 break-even retreat is suppressed rather than eliminated.

**Headroom predicts retreat, not harm.** `kws acoustic` is not a counterexample:
its group answers came out of *declines* so coverage inflated harmlessly, while
text's came out of *label* answers and quality collapsed. Same mechanism, two
shadows — so the crowded-set warning is still a warning about a *risk*, and no
report drops the label-level share.

## Open

- **Teach narrowcast the headroom rule.** `HEADROOM_FINDINGS.md` establishes it;
  the tool's crowded-set warning still fires on label-set *structure* rather than
  on measured headroom. Deliberately not changed by the run that measured it.
- **The size decision** above, now three-way: 17.9 MB (unsafe, and fragile to
  any distribution change), 43 MB (`plantclef24`, slower), 152 MB (fastest).
- **A clean domain-shift test.** The only untested axis left. Needs photographs
  taken on a different camera — Tier 1 in `DATA_STRATEGY.md`.
- **Oregon.** 4,570 research-grade species available, 1,175 with ≥100
  observations; the current 499-binomial catalogue overlaps it by 106. A regional
  catalogue is a fresh fetch, not an adaptation.
- **A real phone.** All latency is M4 Max ANE (121 ms for the 160 MB build), not
  A-series.
- **Seek's on-device accuracy at species rank**, on our images. The only
  offline competitor, no API, needs a manual comparison. Published figures are
  from a 2020-era model on easier data.
- **near-OOD** is the weakest bucket: 22% answered wrong, and it cannot be fixed
  by fetching — only 172 catalogue genera exist and 120 are covered. Needs a
  within-genus margin score rather than summed genus mass.
- **32 catalogue species still unevaluated**, mostly cultivated-only plants
  iNaturalist grades "casual".
- **The catalogue was selected by image availability**, not by any product
  criterion. If the product needs plants where being wrong is expensive
  (toxicity, foraging, invasives), the catalogue should be re-selected against
  that — different data, different project.
