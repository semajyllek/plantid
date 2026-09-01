"""Catalogue label curation: what counts as one nameable plant.

The catalogue inherits PlantNet's labels verbatim, and two things in them are
wrong for a plant-identification app.

**1. Hybrid names are truncated into junk classes.** Class labels were formed as
`" ".join(name.split()[:2])`, which turns `Fragaria × ananassa` into the class
`Fragaria ×` and collapses three different pelargoniums --- `x asperum`,
`x hortorum`, `x hybridum`, 187 images --- into one class `Pelargonium x`, while
`Pelargonium × hortorum` becomes a *separate* class because it is spelled with
`×` rather than `x`. `canonical_name` keeps the hybrid marker, which splits the
conflated hybrids apart and unifies the two spellings. This is a bug fix, not a
product decision: the garden strawberry and the common geranium are exactly the
plants an enthusiast wants named.

(Author citations need no such handling. `Sedum palmeri S.Watson` and
`Sedum palmeri S. Watson` are separate `species_id`s in the catalogue but
already collapse to one class, which is why 530 ids yield 498 species.)

**2. Some labels are distinctions no user wants and no photograph supports.**
`MERGE` folds those together. The criteria, applied in that order:

  - iNaturalist does not recognise the name as an active taxon, or resolves it
    to the target (verified with `resolve_taxa`, not assumed);
  - the plants are not separable from a photograph by a non-specialist;
  - a plant enthusiast would consider them the same plant.

The *Ophrys sphegodes* complex is the case that motivated this. The catalogue
carries eight segregate microspecies of it; iNat recognises none of the seven
merged here, and telling them apart is contested among orchid specialists, let
alone in a phone photo. `Ophrys araneola` is deliberately **not** merged --- it
resolves to itself as an active taxon, so it stays a species.

Merging is not free: it makes the label space easier, so accuracy gains on
merged species are partly definitional. The honest measurement is what happens
to the species left *behind* in a de-crowded genus --- see
`REJECTION_FINDINGS.md`.

`DROP` removes labels that are not species at all.
"""

# Segregates of the Ophrys sphegodes complex. iNat resolves `aranifera` to
# sphegodes and does not recognise the other six at all.
_OPHRYS = ("Ophrys arachnitiformis", "Ophrys aranifera", "Ophrys incubacea",
           "Ophrys lupercalis", "Ophrys occidentalis", "Ophrys passionis",
           "Ophrys virescens")

MERGE = {name: "Ophrys sphegodes" for name in _OPHRYS}
# iNat lumps both under Sedum adolphi; they are sold interchangeably.
MERGE["Sedum nussbaumerianum"] = "Sedum adolphii"

# Not a species: a genus-level placeholder. The model already answers at genus
# level when it cannot name a species, so this label can only ever be noise.
DROP = frozenset({"Pelargonium spp."})


def canonical_name(name: str) -> str:
    """Genus + epithet, keeping the hybrid marker and normalising `×` to `x`.

    'Sedum acre L.'                    -> 'Sedum acre'
    'Fragaria × ananassa (Duchesne)'   -> 'Fragaria x ananassa'
    'Pelargonium x hortorum L.H.Bailey'-> 'Pelargonium x hortorum'
    """
    tokens = str(name).replace("×", " x ").split()
    if not tokens:
        return ""
    out, i = [tokens[0]], 1
    if i < len(tokens) and tokens[i].lower() == "x":
        out.append("x")
        i += 1
    if i < len(tokens):
        out.append(tokens[i])
    return " ".join(out)


def curated_name(name: str, merge: bool = True) -> str | None:
    """Canonical name after curation, or None if the label should be dropped.

    `merge=False` gives the bug fix without the product decision, so the two can
    be measured apart.
    """
    canon = canonical_name(name)
    if canon in DROP:
        return None
    return MERGE.get(canon, canon) if merge else canon


def curated_names(names, merge: bool = True) -> set:
    """The label set a collection of raw catalogue names reduces to."""
    return {c for c in (curated_name(n, merge) for n in names) if c is not None}
