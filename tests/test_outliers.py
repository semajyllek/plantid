import numpy as np
import pandas as pd

from plantid.features.outliers import detect_outliers


def _group(species_id, organ, n=10, dim=5, seed=0, outlier=False):
    rng = np.random.default_rng(seed)
    descs = [rng.normal(0, 0.1, size=dim) for _ in range(n)]
    if outlier:
        descs[-1] = np.full(dim, 50.0)
    return pd.DataFrame(
        {
            "image_id": [f"{species_id}_{organ}_{i}" for i in range(n)],
            "species_id": species_id,
            "species_name": f"species_{species_id}",
            "organ": organ,
            "split": "train",
            "descriptor": descs,
        }
    )


def test_flags_clear_outlier_in_large_group():
    df = _group("sp1", "leaf", n=10, outlier=True)
    results = detect_outliers(df)

    flagged = results[results["is_outlier"]]
    assert len(flagged) == 1
    assert flagged.iloc[0]["image_id"] == "sp1_leaf_9"


def test_no_outliers_in_homogeneous_group():
    df = _group("sp1", "leaf", n=10, outlier=False)
    results = detect_outliers(df)
    assert results["is_outlier"].sum() == 0


def test_small_groups_are_not_flagged():
    df = _group("sp1", "leaf", n=3, outlier=True)
    results = detect_outliers(df)
    assert results["is_outlier"].sum() == 0
    assert results["distance"].isna().all()


def test_multiple_groups_scored_independently():
    df = pd.concat(
        [
            _group("sp1", "leaf", n=10, seed=1, outlier=True),
            _group("sp2", "flower", n=10, seed=2, outlier=False),
        ],
        ignore_index=True,
    )
    results = detect_outliers(df)
    assert results["is_outlier"].sum() == 1
    flagged = results[results["is_outlier"]].iloc[0]
    assert flagged["species_id"] == "sp1"
    assert flagged["organ"] == "leaf"


def test_output_preserves_row_count_and_drops_descriptor_column():
    df = _group("sp1", "leaf", n=6)
    results = detect_outliers(df)
    assert len(results) == 6
    assert "descriptor" not in results.columns
