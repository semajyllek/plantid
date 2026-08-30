import numpy as np
import pandas as pd

from plantid.features import store
from plantid.matching.fusion import evaluate_fusion, fuse_rankings, query_confidence_weights


def test_fuse_rankings_combines_across_organs():
    leaf_ranking = [("A", 0.6), ("B", 0.4)]
    bark_ranking = [("B", 0.7), ("A", 0.3)]
    flower_ranking = [("A", 0.5), ("B", 0.5)]

    fused = fuse_rankings({"leaf": leaf_ranking, "bark": bark_ranking, "flower": flower_ranking})
    fused_species = [sp for sp, _ in fused]

    # A is rank-1 in two of three organs -> should come out on top
    assert fused_species[0] == "A"
    assert set(fused_species) == {"A", "B"}


def test_fuse_rankings_handles_missing_organ():
    fused = fuse_rankings({"leaf": [("A", 1.0)]})
    assert fused == [("A", 1.0 / 61)]


def test_fuse_rankings_respects_weights():
    # B is rank-1 in leaf and bark, A is rank-1 only in flower (high weight).
    rankings = {
        "leaf": [("B", 0.6), ("A", 0.4)],
        "bark": [("B", 0.6), ("A", 0.4)],
        "flower": [("A", 0.6), ("B", 0.4)],
    }

    unweighted = fuse_rankings(rankings)
    assert unweighted[0][0] == "B"

    weighted = fuse_rankings(rankings, weights={"leaf": 0.1, "bark": 0.1, "flower": 1.0})
    assert weighted[0][0] == "A"


def test_query_confidence_weights_uses_top1_score():
    rankings = {
        "leaf": [("A", 0.9), ("B", 0.1)],
        "bark": [("B", 0.34), ("A", 0.33), ("C", 0.33)],
    }

    weights = query_confidence_weights(rankings)

    assert weights["leaf"] == 0.9
    assert weights["bark"] == 0.34


def test_query_confidence_weights_empty_ranking_is_zero():
    assert query_confidence_weights({"leaf": []}) == {"leaf": 0.0}


def _build_dataset(tmp_path, n_per_species=20, dim=3):
    rng = np.random.default_rng(3)
    centers = {"sp1": np.zeros(dim), "sp2": np.full(dim, 10.0)}
    splits = ("train", "val", "test")

    rows = []
    for organ in ("leaf", "bark", "flower"):
        image_ids, species_ids, species_names, split_arr, descs = [], [], [], [], []
        i = 0
        for sp, center in centers.items():
            for split in splits:
                for _ in range(n_per_species):
                    image_id = f"{organ}_{sp}_{split}_{i}"
                    descs.append(rng.normal(loc=center, scale=0.2, size=dim))
                    image_ids.append(image_id)
                    species_ids.append(sp)
                    species_names.append(sp)
                    split_arr.append(split)
                    rows.append(
                        {
                            "image_id": image_id,
                            "species_id": sp,
                            "species_name": sp,
                            "organ": organ,
                            "split": split,
                            "local_path": f"images/{sp}/{organ}/{image_id}.jpg",
                        }
                    )
                    i += 1
        data = {
            "image_id": np.array(image_ids),
            "species_id": np.array(species_ids),
            "species_name": np.array(species_names),
            "split": np.array(split_arr),
            "descriptor": np.stack(descs),
        }
        store.save_descriptors(data, organ, cache_dir=tmp_path)

    pd.DataFrame(rows).to_parquet(tmp_path / "plantnet_index.parquet")

    all_image_ids = [r["image_id"] for r in rows]
    outliers = pd.DataFrame(
        {
            "image_id": all_image_ids,
            "organ": [r["organ"] for r in rows],
            "is_outlier": [False] * len(rows),
        }
    )
    outliers.to_parquet(tmp_path / "outlier_scores.parquet")


def test_evaluate_fusion_perfect_on_separable_clusters(tmp_path):
    _build_dataset(tmp_path)

    metrics = evaluate_fusion(split="test", n_per_split=2, k_match=5, cache_dir=tmp_path)

    assert metrics["n"] > 0
    assert metrics["fused_top1"] == 1.0
    for organ in ("leaf", "bark", "flower"):
        assert metrics[f"{organ}_top1"] == 1.0
