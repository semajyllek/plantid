import pandas as pd

from plantid.data.build_dataset import cap_and_split


def _synthetic_df(counts):
    """counts: dict of (species_id, organ) -> n rows"""
    rows = []
    for (species_id, organ), n in counts.items():
        for i in range(n):
            rows.append(
                {
                    "image_id": f"{species_id}_{organ}_{i}",
                    "species_id": species_id,
                    "organ": organ,
                }
            )
    return pd.DataFrame(rows)


def test_every_group_has_val_and_test():
    counts = {
        ("sp1", "leaf"): 5,
        ("sp1", "flower"): 12,
        ("sp1", "bark"): 60,
        ("sp2", "leaf"): 7,
        ("sp2", "flower"): 5,
        ("sp2", "bark"): 5,
    }
    df = _synthetic_df(counts)
    out = cap_and_split(df, cap=60)

    sizes = out.groupby(["species_id", "organ", "split"]).size().unstack(fill_value=0)
    assert (sizes["val"] >= 1).all()
    assert (sizes["test"] >= 1).all()
    assert (sizes["train"] >= 1).all()


def test_cap_is_respected():
    counts = {("sp1", "bark"): 100}
    df = _synthetic_df(counts)
    out = cap_and_split(df, cap=60)
    assert len(out) == 60


def test_no_duplicate_images_across_splits():
    counts = {("sp1", "leaf"): 20, ("sp2", "flower"): 8}
    df = _synthetic_df(counts)
    out = cap_and_split(df, cap=60)
    assert out["image_id"].is_unique


def test_split_proportions_roughly_15_percent():
    counts = {("sp1", "leaf"): 100}
    df = _synthetic_df(counts)
    out = cap_and_split(df, cap=100)
    sizes = out["split"].value_counts()
    assert sizes["val"] == 15
    assert sizes["test"] == 15
    assert sizes["train"] == 70
