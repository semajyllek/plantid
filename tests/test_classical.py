import numpy as np
import pandas as pd

from plantid.features import store
from plantid.matching.classical import ClassicalMatcher, build_matcher


def test_rank_species_recovers_nearby_cluster():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(10, 4))
    cluster_b = rng.normal(loc=5.0, scale=0.1, size=(10, 4))
    gallery = np.vstack([cluster_a, cluster_b])
    species_ids = np.array(["A"] * 10 + ["B"] * 10)

    mean = gallery.mean(axis=0)
    std = gallery.std(axis=0)
    matcher = ClassicalMatcher((gallery - mean) / std, species_ids, mean, std)

    query_a = np.full((1, 4), 0.0)
    query_b = np.full((1, 4), 5.0)

    [ranked_a] = matcher.rank_species(query_a, k=5)
    [ranked_b] = matcher.rank_species(query_b, k=5)

    assert ranked_a[0][0] == "A"
    assert ranked_b[0][0] == "B"
    assert np.isclose(sum(score for _, score in ranked_a), 1.0)


def test_rank_species_handles_k_larger_than_gallery():
    gallery = np.zeros((3, 2))
    species_ids = np.array(["A", "B", "B"])
    matcher = ClassicalMatcher(gallery, species_ids, mean=np.zeros(2), std=np.ones(2))

    [ranked] = matcher.rank_species(np.zeros((1, 2)), k=100)
    assert {sp for sp, _ in ranked} == {"A", "B"}


def _fake_descriptors(n_per_species=10, dim=4, splits=("train", "test")):
    rng = np.random.default_rng(1)
    image_ids, species_ids, species_names, split_arr, descs = [], [], [], [], []
    centers = {"sp1": 0.0, "sp2": 10.0}
    i = 0
    for sp, center in centers.items():
        for split in splits:
            for _ in range(n_per_species):
                descs.append(rng.normal(loc=center, scale=0.1, size=dim))
                image_ids.append(f"img{i}")
                species_ids.append(sp)
                species_names.append(sp)
                split_arr.append(split)
                i += 1
    return {
        "image_id": np.array(image_ids),
        "species_id": np.array(species_ids),
        "species_name": np.array(species_names),
        "split": np.array(split_arr),
        "descriptor": np.stack(descs),
    }


def test_build_matcher_excludes_outliers_and_non_train(tmp_path):
    data = _fake_descriptors()
    store.save_descriptors(data, "leaf", cache_dir=tmp_path)

    # flag one train image of sp1 as an outlier
    train_sp1 = [
        iid
        for iid, sp, sp_split in zip(data["image_id"], data["species_id"], data["split"])
        if sp == "sp1" and sp_split == "train"
    ]
    outliers = pd.DataFrame(
        {
            "image_id": data["image_id"],
            "organ": "leaf",
            "is_outlier": [iid == train_sp1[0] for iid in data["image_id"]],
        }
    )
    outliers.to_parquet(tmp_path / "outlier_scores.parquet")

    matcher = build_matcher("leaf", cache_dir=tmp_path)

    # gallery should be train-only minus the one flagged outlier: 10 + 10 - 1 = 19
    assert matcher.gallery.shape[0] == 19

    [ranked] = matcher.rank_species(np.full((1, 4), 0.0), k=5)
    assert ranked[0][0] == "sp1"
