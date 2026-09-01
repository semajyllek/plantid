"""Label curation decides what counts as one nameable plant.

The bug being fixed is silent: two-token truncation produced class labels like
`Fragaria ×` and `Pelargonium x`, the latter covering three different hybrids at
once, and no error was raised anywhere.
"""

from plantid.data.curation import MERGE, canonical_name, curated_name, curated_names


def test_author_citation_is_stripped():
    assert canonical_name("Sedum acre L.") == "Sedum acre"
    assert canonical_name("Cirsium arvense (L.) Scop.") == "Cirsium arvense"


def test_author_spelling_variants_collapse():
    """'S.Watson' vs 'S. Watson' were separate species_ids in the catalogue."""
    assert canonical_name("Sedum palmeri S.Watson") == canonical_name("Sedum palmeri S. Watson")


def test_hybrid_epithet_survives():
    """The bug: truncation turned the garden strawberry into the class 'Fragaria ×'."""
    assert canonical_name("Fragaria × ananassa (Duchesne) Rozier") == "Fragaria x ananassa"
    assert canonical_name("Anemone x hybrida Paxton") == "Anemone x hybrida"


def test_distinct_hybrids_stay_distinct():
    """Three pelargoniums shared the single class 'Pelargonium x' (187 images)."""
    names = ["Pelargonium x asperum Ehrh. ex Willd.",
             "Pelargonium x hortorum L.H. Bailey",
             "Pelargonium x hybridum (L.) Aiton"]
    assert len({canonical_name(n) for n in names}) == 3


def test_hybrid_marker_spelling_unifies():
    """'x hortorum' and '× hortorum' were two classes for one plant."""
    assert (canonical_name("Pelargonium x hortorum L.H. Bailey")
            == canonical_name("Pelargonium × hortorum L.H. Bailey")
            == "Pelargonium x hortorum")


def test_ophrys_complex_merges_but_araneola_does_not():
    """araneola resolves to itself as an active iNat taxon, so it is a species."""
    assert curated_name("Ophrys passionis Sennen") == "Ophrys sphegodes"
    assert curated_name("Ophrys aranifera Huds.") == "Ophrys sphegodes"
    assert curated_name("Ophrys araneola sensu auct.plur.") == "Ophrys araneola"
    assert curated_name("Ophrys apifera Huds.") == "Ophrys apifera"


def test_merge_can_be_disabled_to_isolate_the_bug_fix():
    assert curated_name("Ophrys passionis Sennen", merge=False) == "Ophrys passionis"
    assert curated_name("Fragaria × ananassa L.", merge=False) == "Fragaria x ananassa"


def test_genus_placeholder_is_dropped():
    assert curated_name("Pelargonium spp.") is None
    assert "Pelargonium spp." not in curated_names(["Pelargonium spp.", "Sedum acre L."])


def test_merge_targets_are_themselves_stable():
    """A merge target must not itself be merged, or the map is order-dependent."""
    for target in set(MERGE.values()):
        assert curated_name(target) == target


def test_genus_is_recoverable_from_every_label():
    """`genus_matrix` splits on the first token; hybrids must not break it."""
    for raw, genus in [("Fragaria × ananassa L.", "Fragaria"),
                       ("Ophrys passionis Sennen", "Ophrys"),
                       ("Sedum acre L.", "Sedum")]:
        assert curated_name(raw).split()[0] == genus
