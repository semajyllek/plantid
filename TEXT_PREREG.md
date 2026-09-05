# Pre-registration — does the coarse-answer trap survive a change of modality?

Written **before** any text was embedded or any head fitted.

## Why

`BIRDS_FINDINGS.md` established the coarse-answer trap across label hierarchy and
visual domain — plants to birds — but through **one encoder, on images**. The
tool claims to be modality-independent: the cascade, `UTILITY`, prevalence
anchoring, the clustered splits and the card do not know what an image is, and
the modality-specific parts sit behind seams (`encode.py`, `sources.py`).

That claim is untested. This tests it on text.

## Design

20 Newsgroups (`sklearn`, headers/footers/quotes removed), which has a genuine
two-level hierarchy: 20 leaf labels under 7 top-level groups. Two 5-label arms:

- **CROWDED** — all five `comp.*` groups: graphics, os.ms-windows.misc,
  sys.ibm.pc.hardware, sys.mac.hardware, windows.x. One group, and
  `sys.ibm.pc.hardware` vs `sys.mac.hardware` is a genuinely hard pair for a
  human skimming.
- **VARIED** — five labels from five different top-level groups: comp.graphics,
  rec.sport.hockey, sci.space, talk.politics.guns, soc.religion.christian.

`group` is supplied explicitly as the top-level token, since the default
first-whitespace-token rule is a biological-binomial convention that does not
apply here. Negatives are documents from the ten newsgroups in neither arm.

Text is embedded **outside** narrowcast and passed via `--embeddings`, which is
the boundary the tool is designed around: it takes a dataset and does not know
how the dataset was produced.

No cluster column: newsgroup documents are independent, there is no repeated
subject, and narrowcast will record that its intervals are anticonservative
rather than inventing a grouping.

## Prediction, declared now

If the finding is modality-independent, CROWDED will show:

1. **Higher or comparable coverage** than VARIED
2. **Comparable precision**
3. **Substantially lower label-level share**

Images gave 0.476 vs 0.761 (plants) and 0.718 vs 0.958 (birds). Direction is
what is being tested; magnitude on text is not predicted.

## What would falsify it

CROWDED showing lower coverage, or a label-level share close to VARIED's. Either
would mean the trap depends on something about visual encoders or visual label
hierarchies, and the tool's modality-independence claim should be restricted to
vision until shown otherwise.

## Known limitation, stated in advance

One text encoder, one corpus, five labels per arm. Establishes that the effect
is not vision-specific; does not establish a magnitude for text in general.
