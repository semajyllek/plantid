import numpy as np

from plantid.features import store


def _synthetic_data(n=4, dim=6):
    rng = np.random.default_rng(0)
    return {
        "image_id": np.array([f"img{i}" for i in range(n)]),
        "species_id": np.array([f"sp{i % 2}" for i in range(n)]),
        "species_name": np.array([f"Species {i % 2}" for i in range(n)]),
        "split": np.array(["train"] * n),
        "descriptor": rng.normal(size=(n, dim)),
    }


def test_save_and_load_roundtrip(tmp_path):
    data = _synthetic_data()
    store.save_descriptors(data, "leaf", cache_dir=tmp_path)

    loaded = store.load_descriptors("leaf", cache_dir=tmp_path)

    assert set(loaded.keys()) == set(data.keys())
    np.testing.assert_array_equal(loaded["image_id"], data["image_id"])
    np.testing.assert_array_equal(loaded["species_id"], data["species_id"])
    np.testing.assert_allclose(loaded["descriptor"], data["descriptor"])


def test_compute_and_cache_uses_existing_file(tmp_path, monkeypatch):
    data = _synthetic_data()
    store.save_descriptors(data, "leaf", cache_dir=tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("compute_descriptors should not be called when cache exists")

    monkeypatch.setattr(store, "compute_descriptors", boom)

    loaded = store.compute_and_cache(index=None, organ="leaf", cache_dir=tmp_path)
    np.testing.assert_allclose(loaded["descriptor"], data["descriptor"])


def test_compute_and_cache_force_recomputes(tmp_path, monkeypatch):
    old = _synthetic_data()
    store.save_descriptors(old, "leaf", cache_dir=tmp_path)

    new = _synthetic_data(n=2, dim=6)
    monkeypatch.setattr(store, "compute_descriptors", lambda index, organ: new)

    loaded = store.compute_and_cache(index=None, organ="leaf", cache_dir=tmp_path, force=True)
    assert loaded["descriptor"].shape == (2, 6)
