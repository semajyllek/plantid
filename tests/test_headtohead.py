"""Scoring the competitors must not quietly flatter or penalise them.

Two traps here. A genus-rank suggestion counted as a species answer would inflate
their species accuracy; a sampling scheme that draws several observations of one
species would make the interval too tight while looking fine.
"""

import pandas as pd
import pytest

from plantid.eval.headtohead import genus_of, sample_observations, score, top1


def test_genus_rank_suggestion_is_not_a_species_answer():
    """iNaturalist answers at genus when unsure; scoring that as a species
    prediction would credit it for an answer it declined to give."""
    resp = {"results": [{"name": "Sedum", "rank": "genus", "score": 0.9},
                        {"name": "Sedum acre", "rank": "species", "score": 0.4}]}
    assert top1(resp) == "Sedum acre"


def test_single_word_names_are_skipped_even_without_a_rank_field():
    """Pl@ntNet returns no rank field, so the arity of the name is the only signal."""
    assert top1({"results": [{"name": "Sedum"}, {"name": "Sedum acre"}]}) == "Sedum acre"


def test_author_citations_are_stripped_before_comparison():
    """Their names carry authorities; ours do not. Comparing raw strings would
    score every correct answer as wrong."""
    assert top1({"results": [{"name": "Sedum acre L.", "rank": "species"}]}) == "Sedum acre"


def test_hybrid_names_survive_scoring():
    got = top1({"results": [{"name": "Fragaria × ananassa Duchesne", "rank": "species"}]})
    assert got == "Fragaria x ananassa"


def test_no_species_suggestion_returns_none_rather_than_a_wrong_answer():
    assert top1({"results": [{"name": "Plantae", "rank": "kingdom"}]}) is None
    assert top1({"results": []}) is None
    assert top1({"error": 401}) is None


def test_genus_of_handles_missing():
    assert genus_of("Sedum acre") == "Sedum"
    assert genus_of(None) is None


def test_score_counts_only_answered_observations():
    """A service that fails on an image should not be scored as wrong for it —
    that would confuse an outage with an error."""
    rows = [{"truth": "Sedum acre", "plantnet_top1": "Sedum acre",
             "plantnet_top5": ["Sedum acre"], "inat_top1": None, "inat_top5": []},
            {"truth": "Bellis perennis", "plantnet_top1": "Bellis sylvestris",
             "plantnet_top5": ["Bellis sylvestris", "Bellis perennis"],
             "inat_top1": "Bellis perennis", "inat_top5": ["Bellis perennis"]}]
    t = score(rows)
    assert t.loc["plantnet", "n"] == 2 and t.loc["inat", "n"] == 1
    assert t.loc["plantnet", "species_top1"] == pytest.approx(0.5)
    assert t.loc["plantnet", "genus_top1"] == pytest.approx(1.0)   # Bellis both times
    assert t.loc["plantnet", "species_top5"] == pytest.approx(1.0)


def test_sampling_takes_one_observation_per_species(tmp_path):
    """Several observations of one species would break the independence the
    interval assumes."""
    df = pd.DataFrame({
        "obs_id": range(6),
        "species_name": ["Sedum acre L."] * 3 + ["Bellis perennis L."] * 3,
        "genus": ["Sedum"] * 3 + ["Bellis"] * 3,
        "bucket": ["in_catalog"] * 6,
        "local_paths": [["images/a.jpg"]] * 6,
    })
    df.to_parquet(tmp_path / "inat_observations.parquet")
    out = sample_observations(n=10, cache_dir=tmp_path)
    assert len(out) == 2
    assert set(out["truth"]) == {"Sedum acre", "Bellis perennis"}


def test_sampling_excludes_out_of_catalogue_observations(tmp_path):
    df = pd.DataFrame({
        "obs_id": [1, 2],
        "species_name": ["Sedum acre L.", "Banksia serrata L.f."],
        "genus": ["Sedum", "Banksia"],
        "bucket": ["in_catalog", "regional_ood"],
        "local_paths": [["images/a.jpg"], ["images/b.jpg"]],
    })
    df.to_parquet(tmp_path / "inat_observations.parquet")
    assert list(sample_observations(n=10, cache_dir=tmp_path)["truth"]) == ["Sedum acre"]


def test_a_current_name_is_not_scored_wrong_against_our_stale_one():
    """The bug the first trial run exposed: Pl@ntNet answered Leuenbergeria bleo
    to our Pereskia bleo -- the same plant -- and was marked incorrect. 9.5% of
    this catalogue's names are a taxonomic generation behind, so string equality
    measures nomenclature rather than identification."""
    from plantid.eval.headtohead import correct

    aliases = {"Pereskia bleo": {"Pereskia bleo", "Leuenbergeria bleo"}}
    assert correct("Leuenbergeria bleo", "Pereskia bleo", aliases)
    assert correct("Pereskia bleo", "Pereskia bleo", aliases)
    assert not correct("Punica granatum", "Pereskia bleo", aliases)


def test_alias_matching_also_applies_at_genus_level():
    """A renamed species usually moves genus too, so genus scoring needs the same
    treatment or it inherits the same bias."""
    from plantid.eval.headtohead import correct

    aliases = {"Pereskia bleo": {"Pereskia bleo", "Leuenbergeria bleo"}}
    assert correct("Leuenbergeria grandifolia", "Pereskia bleo", aliases, genus=True)
    assert not correct("Punica granatum", "Pereskia bleo", aliases, genus=True)


def test_species_with_no_alias_entry_falls_back_to_exact_match():
    from plantid.eval.headtohead import correct

    assert correct("Sedum acre", "Sedum acre", {})
    assert not correct("Sedum album", "Sedum acre", {})
