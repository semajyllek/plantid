import numpy as np
import pandas as pd

from plantid.eval.rejection import (
    DECLINE,
    GENUS,
    SPECIES,
    UTILITY,
    cluster_bootstrap,
    decide,
    fit_thresholds,
    genus_matrix,
    make_splits,
    scores,
    utility,
)


def _posteriors(n=40, n_species=6, seed=0):
    """Random valid posteriors over n_species catalog classes plus __OTHER__."""
    rng = np.random.default_rng(seed)
    P = rng.random((n, n_species + 1)) ** 3
    return P / P.sum(axis=1, keepdims=True)


def _classes(n_species=6):
    # two genera, three species each -> genus marginalisation has something to do
    names = [f"Alpha sp{i}" for i in range(n_species // 2)]
    names += [f"Beta sp{i}" for i in range(n_species - n_species // 2)]
    return np.array(sorted(names) + ["__OTHER__"])


def test_scores_are_nested():
    """max species <= max genus <= 1 - P(OTHER), exactly, by construction."""
    classes = _classes()
    mask = np.ones(len(classes), bool)
    oi = len(classes) - 1
    mask[oi] = False
    gmat, _ = genus_matrix(classes, mask)

    sc, gc, omo = scores(_posteriors(n_species=len(classes) - 1), mask, oi, gmat)
    assert np.all(sc <= gc + 1e-12)
    assert np.all(gc <= omo + 1e-12)


def test_decide_cascade_precedence():
    species_conf = np.array([0.9, 0.9, 0.1, 0.1])
    genus_conf = np.array([0.9, 0.2, 0.9, 0.2])
    got = decide(species_conf, genus_conf, t_genus=0.5, t_species=0.5)
    # low genus confidence declines regardless of species confidence
    assert list(got) == [SPECIES, DECLINE, GENUS, DECLINE]


def test_utility_scores_each_outcome():
    levels = np.array([SPECIES, SPECIES, GENUS, GENUS, DECLINE, DECLINE], dtype=object)
    species_ok = np.array([True, False, False, False, False, False])
    genus_ok = np.array([True, False, True, False, False, False])
    in_catalog = np.array([True, True, True, True, True, False])
    got = utility(levels, species_ok, genus_ok, in_catalog)
    assert list(got) == [
        UTILITY["species_correct"], UTILITY["wrong"],
        UTILITY["genus_correct"], UTILITY["wrong"],
        UTILITY["decline_in_catalog"], UTILITY["decline_ood"],
    ]


def test_fit_thresholds_declines_when_everything_is_wrong():
    """If no answer is ever right, declining out-of-catalogue is the best policy."""
    n = 60
    conf = np.linspace(0.2, 0.95, n)
    got, best_u = fit_thresholds(
        species_conf=conf, genus_conf=conf,
        species_ok=np.zeros(n, bool), genus_ok=np.zeros(n, bool),
        in_catalog=np.zeros(n, bool),
    )
    levels = decide(conf, conf, *got[::-1][::-1])
    assert best_u > 0  # declining OOD pays +1 each
    assert (levels == DECLINE).mean() > 0.9


def test_make_splits_keeps_clusters_whole():
    df = pd.DataFrame({
        "bucket": ["in_catalog"] * 8 + ["near_ood"] * 8,
        "species": [f"Sp{i // 2}" for i in range(8)] + [f"Other{i // 2}" for i in range(8)],
        "genus": ["G"] * 8 + [f"NG{i // 4}" for i in range(8)],
    })
    fold = make_splits(df, seed=1)
    # in_catalog splits by species, near_ood by genus: no cluster spans both folds
    for key, sub in (("species", df.bucket == "in_catalog"), ("genus", df.bucket == "near_ood")):
        folds_per_cluster = df[sub].assign(fold=fold[sub]).groupby(key)["fold"].nunique()
        assert (folds_per_cluster == 1).all()


def test_cluster_bootstrap_brackets_the_mean():
    values = np.repeat([0.0, 1.0], 30)
    clusters = np.repeat([f"c{i}" for i in range(12)], 5)
    lo, hi = cluster_bootstrap(values, clusters, n=200, seed=0)
    assert lo <= values.mean() <= hi
