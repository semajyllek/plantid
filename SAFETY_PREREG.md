# Pre-registration — Oregon safety-pair discrimination

Written **before** any images were fetched, embedded or fitted. The point of the
test is that a general-purpose plant identifier's mean accuracy says nothing
about whether it can be trusted on the specific pairs where being wrong is
expensive. Those pairs are named here, in advance, and reported individually.
No averaging across pairs — an average over these would hide exactly the case
that matters.

## The question

Can frozen BioCLIP-2 embeddings separate Oregon's dangerous lookalikes at a
precision that would justify shipping a consequence-framed product?

If not, the Oregon app is a demo of nothing and the tool's story is "narrow
classifiers over datasets you already have", not "safety-critical offline ID".

## The pairs, and which direction is fatal

`A → B` means: predicting **B** when the truth is **A** is the dangerous error.

### Apiaceae — the group that kills people
Lomatium roots are dug and eaten across the Pacific Northwest. Poison hemlock and
water hemlock are in the same family and grow in the same places.

| truth | mistaken for | why it matters |
|---|---|---|
| Conium maculatum | Daucus carota | the textbook fatal confusion |
| Conium maculatum | Lomatium nudicaule | forager digs hemlock root |
| Conium maculatum | Lomatium utriculatum | as above |
| Conium maculatum | Anthriscus caucalis | umbel/fern-leaf confusion |
| Conium maculatum | Osmorhiza berteroi | sweet cicely is foraged |
| Cicuta douglasii | Heracleum maximum | water hemlock vs cow parsnip |
| Cicuta douglasii | Lomatium dissectum | both wetland-adjacent, both dug |

### Foraged berries
| truth | mistaken for | why it matters |
|---|---|---|
| Sambucus racemosa | Sambucus cerulea | red elderberry is toxic raw, blue is not |
| Rubus armeniacus | Rubus ursinus | invasive vs native — management, not safety |

### Contact and cardiac
| truth | mistaken for | why it matters |
|---|---|---|
| Toxicodendron diversilobum | Rubus ursinus | poison oak vs trailing blackberry, both trifoliate |
| Digitalis purpurea | Verbascum-like rosettes | first-year foxglove rosette is eaten in error |
| Veratrum viride | Veratrum californicum | both toxic; a within-genus control |

## Metrics, declared now

For each pair, from a single head fitted over all 27 species:

1. **Fatal-direction rate** — P(predict B | truth A) for the pairs above. This is
   the number the product lives or dies on, not accuracy.
2. **Pairwise separability** — balanced accuracy on the two classes alone.
3. **Abstention behaviour** — what share of the dangerous species is declined
   rather than answered, under the existing three-way cascade. Declining hemlock
   is a *safe* outcome; naming it wild carrot is not. These must be counted
   separately and never summed into one "error rate".

## Splitting

Split by **observation**, never by image: an iNaturalist observation bundles
several photographs of one individual plant, and an image-level split puts the
same plant on both sides. Intervals bootstrap over observations.

## What would kill the idea

A fatal-direction rate above ~1% on the Conium pairs at any usable coverage. At
that point the honest product does not name Apiaceae at all — it says "umbel,
do not eat" and stops, which is a different and much smaller product.

## Known limitation, stated in advance

iNaturalist is inside BioCLIP-2's training data via GBIF (`DATA_STRATEGY.md`), so
these numbers are optimistic. They set an **upper bound**: if the pairs fail
here, they fail everywhere. Passing here does not establish they pass in the
field.
