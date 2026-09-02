"""The transfer set must never contain an image we evaluate on.

Distilling on the catalogue test split or on the iNaturalist observations would
fit the student to the exact photographs it is then scored with. It would raise
nothing, look like a large win, and be worse than any leak found in this project
so far. This is the guard.
"""

import numpy as np
import pandas as pd
import pytest

from plantid.train.distil import build_transfer_set, cosine_loss


@pytest.fixture
def cache(tmp_path):
    """A miniature processed-data directory with all three splits present."""
    rows, npz = [], {}
    for organ in ("leaf", "flower", "bark"):
        ids, splits = [], []
        for split in ("train", "val", "test"):
            for k in range(2):
                ids.append(f"{organ}_{split}_{k}")
                splits.append(split)
        rows += [{"image_id": i, "species_id": 1, "species_name": "Sedum acre L.",
                  "organ": organ, "split": s, "local_path": f"images/{i}.jpg"}
                 for i, s in zip(ids, splits)]
        npz[organ] = (np.asarray(ids, dtype=str), np.asarray(splits, dtype=str))
    pd.DataFrame(rows).to_parquet(tmp_path / "catalog_index.parquet")

    bg_rows = [{"image_id": f"bg_{organ}_{k}", "species_id": 9, "species_name": "Bellis perennis L.",
                "organ": organ, "local_path": f"images_background/bg_{organ}_{k}.jpg"}
               for organ in ("leaf", "flower", "bark") for k in range(2)]
    pd.DataFrame(bg_rows).to_parquet(tmp_path / "plantnet_background.parquet")

    for organ in ("leaf", "flower", "bark"):
        ids, splits = npz[organ]
        np.savez(tmp_path / f"catalog_{organ}_bioclip2.npz",
                 descriptor=np.random.rand(len(ids), 768).astype(np.float32),
                 image_id=ids, species_id=np.array(["1"] * len(ids)),
                 species_name=np.array(["Sedum acre L."] * len(ids)), split=splits)
        bids = np.asarray([f"bg_{organ}_{k}" for k in range(2)], dtype=str)
        np.savez(tmp_path / f"background_{organ}_bioclip2.npz",
                 descriptor=np.random.rand(2, 768).astype(np.float32),
                 image_id=bids, species_id=np.array(["9", "9"]),
                 species_name=np.array(["Bellis perennis L."] * 2))
    return tmp_path


def test_excludes_the_catalogue_test_split(cache):
    """Test images are what the per-organ accuracy numbers are computed on."""
    paths = set(build_transfer_set(cache)["local_path"])
    assert not any("_test_" in p for p in paths)


def test_excludes_the_catalogue_val_split(cache):
    """calibration.py fits temperature on val; training on it leaks into that."""
    paths = set(build_transfer_set(cache)["local_path"])
    assert not any("_val_" in p for p in paths)


def test_includes_catalogue_train_and_all_background(cache):
    paths = set(build_transfer_set(cache)["local_path"])
    assert sum("_train_" in p for p in paths) == 6      # 3 organs x 2
    assert sum(p.startswith("images_background/") for p in paths) == 6


def test_never_reaches_for_inaturalist(cache):
    """The eval set is not even a candidate: no iNat manifest exists in `cache`,
    and its construction must not depend on one."""
    assert not (cache / "inat_observations.parquet").exists()
    assert len(build_transfer_set(cache)) == 12


def test_teacher_targets_align_with_their_images(cache):
    """A shuffled join here would train the student on mismatched pairs and still
    converge to something plausible-looking."""
    df = build_transfer_set(cache)
    z = np.load(cache / "catalog_leaf_bioclip2.npz")
    want = {i: d for i, d in zip(z["image_id"], z["descriptor"]) if "_train_" in i}
    for _, row in df.iterrows():
        stem = row["local_path"].split("/")[-1].removesuffix(".jpg")
        if stem in want:
            assert np.allclose(row["teacher"], want[stem])


def test_cosine_loss_is_zero_on_a_perfect_match():
    torch = pytest.importorskip("torch")

    v = torch.nn.functional.normalize(torch.randn(4, 768), dim=-1)
    assert float(cosine_loss(v, v)) == pytest.approx(0.0, abs=1e-6)


def test_cosine_loss_is_one_on_orthogonal_embeddings():
    torch = pytest.importorskip("torch")

    a = torch.zeros(1, 4); a[0, 0] = 1
    b = torch.zeros(1, 4); b[0, 1] = 1
    assert float(cosine_loss(a, b)) == pytest.approx(1.0, abs=1e-6)
