import json

import numpy as np
import pytest

from plantid.tool import build, card, encoders, plan, projection, species


# ---- species list parsing -------------------------------------------------

def test_canonical_strips_authority_and_comments():
    assert species.canonical("Sedum acre L.") == "Sedum acre"
    assert species.canonical("  Trifolium repens  # in the lawn ") == "Trifolium repens"
    assert species.canonical("# just a comment") is None


def test_canonical_matches_the_repo_join_key():
    """Hybrids normalise to 'x', not '×' -- the tool must not spell the key its own way."""
    from plantid.data.curation import canonical_name
    for raw in ("Fragaria × ananassa Duchesne", "Pelargonium x hortorum L.H. Bailey",
                "Sedum acre L."):
        assert species.canonical(raw) == canonical_name(raw)


def test_read_list_dedupes_and_keeps_order(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("Sedum acre L.\n\n# comment\nTrifolium repens\nSedum acre\n")
    assert species.read_list(p) == ["Sedum acre", "Trifolium repens"]


def test_read_list_raises_rather_than_dropping(tmp_path):
    """A silently ignored species is a model that cannot see a plant the user asked for."""
    p = tmp_path / "s.txt"
    p.write_text("Sedum acre\nnot a binomial\n")
    with pytest.raises(ValueError, match="could not parse"):
        species.read_list(p)


# ---- composition ----------------------------------------------------------

POOL = ["Sedum acre", "Sedum album", "Sedum dasyphyllum", "Sedum rupestre",
        "Trifolium repens", "Trifolium pratense", "Bellis perennis"]


def test_analyse_congener_fraction_and_crowding():
    a = species.analyse(["Sedum acre", "Sedum album", "Bellis perennis"], pool=POOL)
    assert a["n_species"] == 3 and a["n_genera"] == 2
    assert a["in_set_congener_frac"] == pytest.approx(2 / 3)
    assert a["crowded_genera"] == {"Sedum": 2}


def test_analyse_groups_outside_congeners_by_genus():
    a = species.analyse(["Sedum acre", "Sedum album"], pool=POOL)
    assert a["outside_congeners"] == {"Sedum": ["Sedum dasyphyllum", "Sedum rupestre"]}
    assert a["n_species_exposed"] == 2


def test_analyse_no_congeners_when_set_is_separated():
    a = species.analyse(["Bellis perennis"], pool=POOL)
    assert a["in_set_congener_frac"] == 0.0
    assert a["crowded_genera"] == {} and a["outside_congeners"] == {}


# ---- encoder choice -------------------------------------------------------

def test_choose_takes_largest_that_fits():
    assert encoders.choose(20).variant == "mobileclip2_s2"
    assert encoders.choose(10).variant == "mobileclip2_s0"
    assert encoders.choose(1000).variant == "bioclip2"
    assert encoders.choose(None).variant == "bioclip2"


def test_choose_falls_back_to_smallest_when_nothing_fits():
    assert encoders.choose(1).variant == "mobileclip2_s0"


def test_bioclip2_int4_reconciles_with_shipped_artifact():
    """152 MB against the 160 MB build is the check that the counts are image-tower only."""
    assert encoders.BY_VARIANT["bioclip2"].size_mb(4) == pytest.approx(152.0)


# ---- projection -----------------------------------------------------------

def test_project_matches_measured_cell_at_an_anchor():
    anchors = projection.GRID["congener_anchors"]["20"]
    p = projection.project("mobileclip2_s2", 20, anchors["easy"], p_ood=0.2)
    assert p["coverage"] == pytest.approx(0.771, abs=1e-6)
    assert p["species_share"] == pytest.approx(0.833, abs=1e-6)


def test_congener_dense_sets_project_lower_species_share():
    a = projection.GRID["congener_anchors"]["20"]
    easy = projection.project("bioclip2", 20, a["easy"])
    hard = projection.project("bioclip2", 20, a["hard"])
    assert hard["species_share"] < easy["species_share"]
    # ...while coverage looks *better*, which is the trap the tool exists to flag
    assert hard["coverage"] > easy["coverage"]


def test_project_clamps_rather_than_extrapolating():
    p = projection.project("bioclip2", 3, 0.0)
    assert p["K_used"] == 10 and p["extrapolated"] == "below"
    p = projection.project("bioclip2", 400, 0.0)
    assert p["K_used"] == 50 and p["extrapolated"] == "above"


def test_project_rejects_unmeasured_prevalence():
    with pytest.raises(ValueError, match="not measured"):
        projection.project("bioclip2", 20, 0.2, p_ood=0.33)


# ---- plan -----------------------------------------------------------------

def test_plan_warns_on_crowded_genera():
    pl = plan.make_plan(["Sedum acre", "Sedum album", "Sedum dasyphyllum"],
                        budget_mb=20, pool=POOL)
    kinds = {w.kind for w in pl["warnings"]}
    assert "crowded" in kinds and "outside_congeners" in kinds
    assert "species-level" in plan.render(pl)


def test_plan_warns_about_species_it_has_no_images_for():
    """A confident projection for a list `build` will drop is the worst failure here."""
    pl = plan.make_plan(["Sedum acre", "Conium maculatum"], budget_mb=20, pool=POOL)
    w = {x.kind: x for x in pl["warnings"]}
    assert "missing" in w and "Conium maculatum" in w["missing"].detail
    assert pl["n_available"] == 1


def test_plan_refuses_to_project_when_most_of_the_list_is_absent():
    pl = plan.make_plan(["Conium maculatum", "Cicuta virosa", "Sedum acre"],
                        budget_mb=20, pool=POOL)
    assert pl["projection"] is None
    assert "unprojectable" in {x.kind for x in pl["warnings"]}
    assert "No projection" in plan.render(pl)


def test_plan_reports_budget_shortfall():
    pl = plan.make_plan(["Bellis perennis"], budget_mb=20, pool=POOL)
    assert "152.0 MB" in pl["budget_note"]
    assert pl["encoder"].variant == "mobileclip2_s2"


# ---- card -----------------------------------------------------------------

def _manifest(species_share):
    return {
        "bundle_version": 1, "created": "2026-01-01T00:00:00",
        "encoder": "mobileclip2_s2", "source": "local-catalogue",
        "species": ["Sedum acre", "Sedum album"],
        "counts": {"train": 100},
        "composition": {"n_species": 2, "crowded_genera": {"Sedum": 2}},
        "outside_congeners": {"Sedum": ["Sedum dasyphyllum"]},
        "utility": {"species_correct": 1.0},
        "metrics": {
            "t_genus": 0.5, "t_species": 0.9, "p_ood": 0.2,
            "coverage": 0.84, "precision": 0.97, "species_share": species_share,
            "closed_set_top1": 0.81, "n_calib": 10, "n_test": 20,
            "per_bucket": {"in_catalog": {"n": 20, "answered": 0.9,
                                          "correct_when_answered": 0.95}},
        },
    }


def test_card_flags_a_low_species_share():
    out = card.render(_manifest(0.31))
    assert "Read the species-level share, not the coverage" in out


def test_card_omits_the_flag_when_species_share_is_healthy():
    assert "Read the species-level share" not in card.render(_manifest(0.85))


def test_card_always_carries_the_safety_line():
    for share in (0.31, 0.85):
        assert "Do not eat anything" in card.render(_manifest(share))


def test_card_singularises_a_lone_relative():
    out = card.render(_manifest(0.85))
    assert "1 relative not on your list" in out


def test_card_shows_cluster_bootstrapped_intervals():
    m = _manifest(0.85)
    m["metrics"]["ci"] = {"species_share": [0.61, 0.96], "precision": [0.90, 0.99],
                          "closed_set_top1": [0.70, 0.92]}
    m["metrics"]["n_species_clusters"] = 7
    out = card.render(m)
    assert "61.0–96.0%" in out
    assert "over **species**, not rows" in out


def test_card_dashes_a_missing_interval_rather_than_inventing_one():
    out = card.render(_manifest(0.85))   # no "ci" key at all
    assert "—" in out and "None" not in out


# ---- interval computation -------------------------------------------------

def test_ci_needs_at_least_two_clusters():
    one = np.ones(2)
    assert build._ci(np.array([1.0, 0.0]), one, np.array(["a", "a"])) is None
    assert build._ci(np.array([]), np.array([]), np.array([])) is None


def test_ci_brackets_the_point_estimate():
    vals = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    clusters = np.array(["a", "a", "b", "b", "c", "c"])
    lo, hi = build._ci(vals, np.ones(6), clusters)
    assert lo <= vals.mean() <= hi


def test_ci_tracks_the_weighted_ratio_not_the_unweighted_mean():
    """The bug this replaced: precision 96% with a 22-77% interval around it."""
    correct = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    w = np.array([10.0, 10.0, 0.1, 0.1, 0.1, 0.1])   # in-list rows dominate
    clusters = np.array(["a", "b", "c", "d", "e", "f"])
    lo, hi = build._ci(correct * w, w, clusters)
    point = (correct * w).sum() / w.sum()
    assert lo <= point <= hi
    assert hi > 0.5   # nowhere near the 0.33 unweighted mean


# ---- bundle round trip ----------------------------------------------------

class _Clf:
    coef_ = np.zeros((2, 4))
    intercept_ = np.zeros(2)
    classes_ = np.array(["Sedum acre", "__OTHER__"])


def test_bundle_round_trip(tmp_path):
    m = _manifest(0.5)
    out = build.save_bundle(tmp_path / "b", _Clf(), m["species"], "mobileclip2_s2",
                            m["metrics"], species.analyse(m["species"], pool=POOL),
                            {"train": 100}, source="local-catalogue")
    loaded = build.load_bundle(out)
    assert loaded["species"] == m["species"]
    assert loaded["encoder"] == "mobileclip2_s2"
    assert loaded["metrics"]["coverage"] == 0.84
    assert json.loads((out / "manifest.json").read_text())["bundle_version"] == 1
    assert (out / "head.npz").exists()
