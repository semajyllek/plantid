import pandas as pd

from plantid.data.groups import sample_groups


def _index(rows):
    return pd.DataFrame(rows, columns=["image_id", "species_id", "species_name", "organ", "split"])


def test_sample_groups_basic():
    rows = []
    for organ, n in [("leaf", 5), ("flower", 5), ("bark", 1)]:
        for i in range(n):
            rows.append((f"sp1_{organ}_{i}", "sp1", "Species One", organ, "train"))
    index = _index(rows)

    groups = sample_groups(index, n_per_split=10)

    assert len(groups) == 10
    assert (groups["plant_id"] == "sp1").all()
    # bark only has 1 image, so it must be reused across all groups
    assert groups["bark_image_id"].nunique() == 1
    # leaf/flower have 5 images each, sampled with replacement up to 10
    assert groups["leaf_image_id"].nunique() <= 5


def test_sample_groups_skips_species_missing_an_organ():
    rows = []
    for organ, n in [("leaf", 5), ("flower", 5)]:  # no bark
        for i in range(n):
            rows.append((f"sp1_{organ}_{i}", "sp1", "Species One", organ, "train"))
    index = _index(rows)

    groups = sample_groups(index, n_per_split=10)
    assert len(groups) == 0


def test_sample_groups_per_split():
    rows = []
    for split in ["train", "val", "test"]:
        for organ in ["leaf", "flower", "bark"]:
            for i in range(5):
                rows.append((f"sp1_{organ}_{split}_{i}", "sp1", "Species One", organ, split))
    index = _index(rows)

    groups = sample_groups(index, n_per_split=4)
    assert set(groups["split"]) == {"train", "val", "test"}
    assert (groups.groupby("split").size() == 4).all()
