import numpy as np
import pandas as pd

from plantid.eval.match_eval import evaluate, per_species_top1, sweep_k
from plantid.matching.classical import ClassicalMatcher


def _separable_data():
    rng = np.random.default_rng(2)
    image_ids, species_ids, species_names, splits, descs = [], [], [], [], []
    centers = {"sp1": np.array([0.0, 0.0]), "sp2": np.array([10.0, 10.0])}
    i = 0
    for sp, center in centers.items():
        for split, n in (("train", 20), ("val", 5), ("test", 5)):
            for _ in range(n):
                descs.append(rng.normal(loc=center, scale=0.2, size=2))
                image_ids.append(f"img{i}")
                species_ids.append(sp)
                species_names.append(sp)
                splits.append(split)
                i += 1
    return {
        "image_id": np.array(image_ids),
        "species_id": np.array(species_ids),
        "species_name": np.array(species_names),
        "split": np.array(splits),
        "descriptor": np.stack(descs),
    }


def _matcher_from_train(data):
    mask = data["split"] == "train"
    raw = data["descriptor"][mask]
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std < 1e-12] = 1.0
    return ClassicalMatcher((raw - mean) / std, data["species_id"][mask], mean, std)


def test_evaluate_separable_clusters_perfect_accuracy():
    data = _separable_data()
    matcher = _matcher_from_train(data)

    metrics = evaluate(matcher, data, split="test", k=5, top_ks=(1, 2))
    assert metrics["top1"] == 1.0
    assert metrics["top2"] == 1.0
    assert metrics["n"] == 10


def test_sweep_k_returns_all_requested_ks():
    data = _separable_data()
    matcher = _matcher_from_train(data)

    sweep = sweep_k(matcher, data, split="val", ks=(1, 5))
    assert set(sweep.keys()) == {1, 5}
    assert sweep[1] == 1.0


def test_per_species_top1_identifies_perfect_accuracy():
    data = _separable_data()
    matcher = _matcher_from_train(data)

    df = per_species_top1(matcher, data, split="test", k=5)
    assert set(df["species_id"]) == {"sp1", "sp2"}
    assert (df["top1_accuracy"] == 1.0).all()
    assert (df["n"] == 5).all()
