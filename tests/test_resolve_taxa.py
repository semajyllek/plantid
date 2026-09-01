"""The match rule is the whole safety argument of `resolve_taxa`.

iNat's fuzzy search happily returns a different plant for a name it does not
know, so accepting the first result would silently rewrite catalogue species
into their neighbours. These are the real API responses that motivated the rule.
"""

from plantid.data.resolve_taxa import pick


def taxon(name, matched_term, rank="species"):
    return {"name": name, "matched_term": matched_term, "rank": rank, "id": 1}


def test_exact_synonym_wins():
    """`matched_term` equal to the query means iNat lists our name as a synonym."""
    results = [taxon("Anemonoides nemorosa", "Anemone nemorosa"),
               taxon("Anemonoides ranunculoides", "Anemone nemorosa-lutea")]
    t, how = pick(results, "Anemone nemorosa")
    assert (t["name"], how) == ("Anemonoides nemorosa", "exact")


def test_exact_beats_earlier_epithet_match():
    """Order in the response is relevance, not correctness: Sedum kamtschaticum
    resolves to Phedimus kamtschaticus, which iNat returns third."""
    results = [taxon("Phedimus aizoon", "Sedum kamtschaticum viviparum"),
               taxon("Phedimus ellacombeanus", "Sedum kamtschaticum ellacombeanum"),
               taxon("Phedimus kamtschaticus", "Sedum kamtschaticum")]
    t, how = pick(results, "Sedum kamtschaticum")
    assert (t["name"], how) == ("Phedimus kamtschaticus", "exact")


def test_epithet_survives_genus_transfer():
    """No exact synonym is listed, but the epithet pins the species."""
    results = [taxon("Anemonoides blanda", "Anemone apennina blanda"),
               taxon("Anemonoides apennina", "Anemone apennina apennina")]
    t, how = pick(results, "Anemone apennina")
    assert (t["name"], how) == ("Anemonoides apennina", "epithet")


def test_rejects_a_merely_similar_species():
    """The failure this rule exists to prevent: `q=` returning a real but
    different plant, which a first-result match would have accepted."""
    results = [taxon("Acalypha rhomboidea", "Acalypha rhomboidea")]
    assert pick(results, "Acalypha virginica") == (None, None)


def test_ignores_non_species_ranks():
    results = [taxon("Pelargonium", "Pelargonium", rank="genus")]
    assert pick(results, "Pelargonium x") == (None, None)


def test_no_results():
    assert pick([], "Ophrys lupercalis") == (None, None)
