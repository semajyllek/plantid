"""The name rewrite in `_rows` is a silent-failure point.

The catalogue carries PlantNet's pre-split names, so observations of a renamed
species come back under a binomial the catalogue does not contain. Without the
rewrite they fail the in-catalogue membership test and the fetch returns zero
rows for a species with 84,000 observations -- with no error anywhere.
"""

from plantid.data.inat_eval import _rows

CATALOG_SPECIES = {"Anemone nemorosa", "Lactuca sativa"}
CATALOG_GENERA = {"Anemone", "Lactuca"}
RENAMES = {"Anemonoides nemorosa": "Anemone nemorosa"}


def observation(name, obs_id=1, n_photos=2):
    return {"id": obs_id, "taxon": {"name": name, "rank": "species"},
            "photos": [{"url": f"http://x/{i}/square.jpg"} for i in range(n_photos)]}


def test_renamed_species_is_kept_under_its_catalogue_name():
    out = _rows([observation("Anemonoides nemorosa")], "in_catalog",
                CATALOG_SPECIES, CATALOG_GENERA, name_map=RENAMES)
    assert len(out) == 1
    assert out[0]["species_name"] == "Anemone nemorosa"
    assert out[0]["genus"] == "Anemone"


def test_original_name_is_kept_for_provenance():
    out = _rows([observation("Anemonoides nemorosa")], "in_catalog",
                CATALOG_SPECIES, CATALOG_GENERA, name_map=RENAMES)
    assert out[0]["inat_name"] == "Anemonoides nemorosa"


def test_without_the_map_the_same_observation_is_discarded():
    """The bug this rewrite exists to fix."""
    assert _rows([observation("Anemonoides nemorosa")], "in_catalog",
                 CATALOG_SPECIES, CATALOG_GENERA) == []


def test_rewrite_moves_a_row_out_of_the_ood_buckets():
    """A renamed catalogue species must not be scored as a plant to reject."""
    obs = [observation("Anemonoides nemorosa")]
    assert _rows(obs, "distant_ood", CATALOG_SPECIES, CATALOG_GENERA) != []
    assert _rows(obs, "distant_ood", CATALOG_SPECIES, CATALOG_GENERA,
                 name_map=RENAMES) == []


def test_unmapped_names_pass_through_unchanged():
    out = _rows([observation("Lactuca sativa")], "in_catalog",
                CATALOG_SPECIES, CATALOG_GENERA, name_map=RENAMES)
    assert out[0]["species_name"] == out[0]["inat_name"] == "Lactuca sativa"


def test_near_ood_still_requires_genus_in_catalogue():
    out = _rows([observation("Lactuca serriola")], "near_ood",
                CATALOG_SPECIES, CATALOG_GENERA, name_map=RENAMES)
    assert len(out) == 1 and out[0]["genus"] == "Lactuca"
    assert _rows([observation("Bellis perennis")], "near_ood",
                 CATALOG_SPECIES, CATALOG_GENERA) == []


def test_single_photo_observations_are_dropped():
    assert _rows([observation("Anemone nemorosa", n_photos=1)], "in_catalog",
                 CATALOG_SPECIES, CATALOG_GENERA) == []
