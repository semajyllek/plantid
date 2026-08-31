import numpy as np
import pandas as pd

from plantid.data.species_ranges import LocationPrior, haversine_km
from plantid.eval.location import decide_with_location, location_scores
from plantid.eval.rejection import DECLINE, GENUS, SPECIES


def _points():
    """Two species with disjoint ranges: one in Britain, one in California."""
    rows = []
    for lat, lon in [(51.5, -0.1), (51.6, -0.2), (52.0, -0.5)]:
        rows.append({"species": "Britanica testa", "obs_id": len(rows), "lat": lat, "lon": lon})
    for lat, lon in [(37.8, -122.4), (37.7, -122.5), (38.0, -122.2)]:
        rows.append({"species": "Californica testa", "obs_id": len(rows), "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def test_haversine_known_distance():
    # London to Paris is ~344 km
    d = haversine_km(51.507, -0.128, np.array([48.857]), np.array([2.352]))
    assert 330 < d[0] < 360


def test_prior_favours_the_local_species():
    p = LocationPrior(_points(), bandwidth_km=50)
    london = p.prior(51.5, -0.1)
    assert london[p.index["Britanica testa"]] > london[p.index["Californica testa"]]
    sf = p.prior(37.8, -122.4)
    assert sf[p.index["Californica testa"]] > sf[p.index["Britanica testa"]]


def test_prior_never_reaches_zero():
    """A species with no nearby records must stay possible — the catalogue is
    full of cultivated exotics growing far outside their range."""
    p = LocationPrior(_points(), bandwidth_km=50, alpha=0.5)
    london = p.prior(51.5, -0.1)
    assert london[p.index["Californica testa"]] > 0
    assert np.isclose(london.sum(), 1.0)


def test_plausible_set_is_local():
    p = LocationPrior(_points(), bandwidth_km=50)
    assert p.plausible(51.5, -0.1, min_weight=1.0) == {"Britanica testa"}
    assert p.plausible(37.8, -122.4, min_weight=1.0) == {"Californica testa"}


def test_unknown_species_gets_neutral_not_floor():
    """Species missing from the range data (33 of 248, reclassified by iNat)
    must not be penalised everywhere — that is a bias, not a location signal."""
    p = LocationPrior(_points(), bandwidth_km=50)
    df = pd.DataFrame([{"lat": 51.5, "lon": -0.1, "pred_species": "Missing species"}])
    neutral = location_scores(df, p, neutral=True)["loc_prq"].iloc[0]
    floored = location_scores(df, p, neutral=False)["loc_prq"].iloc[0]
    assert np.isclose(neutral, 0.5)   # mid-rank: no evidence either way
    assert floored < neutral


def test_location_gate_only_withholds_never_renames():
    """The gate may demote or decline, but must never change the named label."""
    n = 4
    sp = np.array([0.99, 0.99, 0.20, 0.20])
    gn = np.array([0.99, 0.99, 0.99, 0.99])
    loc = np.array([9.0, 0.01, 9.0, 0.01])
    got = decide_with_location(sp, gn, loc, t_genus=0.5, t_species=0.5, t_loc=1.0)
    assert got[0] == SPECIES          # confident and locally plausible
    assert got[1] == GENUS            # confident but locally implausible -> demoted
    assert got[2] == GENUS            # unchanged by location
    assert len(got) == n


def test_location_gate_can_only_reduce_specificity():
    rng = np.random.default_rng(0)
    sp, gn = rng.random(200), rng.random(200)
    loc = rng.random(200)
    rank = {SPECIES: 2, GENUS: 1, DECLINE: 0}
    base = np.array([rank[x] for x in
                     decide_with_location(sp, gn, loc, 0.4, 0.6, 0.0)])
    gated = np.array([rank[x] for x in
                      decide_with_location(sp, gn, loc, 0.4, 0.6, 0.5)])
    assert (gated <= base).all()


def test_rank_quantile_is_within_location_unlike_mass():
    """The admissible score must be invariant to how well-recorded a place is.

    Absolute mass is a function of the coordinates alone, so it cannot separate
    a catalogue plant from an unknown one arriving at the same coordinates.
    """
    pts = _points()
    dense = pd.concat([pts] + [pts[pts.species == "Britanica testa"]] * 4, ignore_index=True)
    dense["obs_id"] = range(len(dense))
    sparse_p = LocationPrior(pts, bandwidth_km=50)
    dense_p = LocationPrior(dense, bandwidth_km=50)
    # the local species still ranks top in both, despite very different mass
    assert sparse_p.rank_quantile(51.5, -0.1, "Britanica testa") == \
           dense_p.rank_quantile(51.5, -0.1, "Britanica testa")
    assert dense_p.weights(51.5, -0.1).sum() > sparse_p.weights(51.5, -0.1).sum()
