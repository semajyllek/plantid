"""The npz payload must match what the PyTorch embedding path writes.

A missing key here does not fail here. It fails much later, inside `build_heads`
or `load_background`, with nothing pointing back at the writer — the same shape
of silent failure as the hybrid-name truncation and the `name_map` rewrite.
"""

import numpy as np
import pandas as pd
import pytest

from plantid.deploy.embed_coreml import pack


@pytest.fixture
def sub():
    return pd.DataFrame({
        "image_id": ["a", "b"],
        "species_id": [1355936, 1355978],
        "species_name": ["Sedum acre L.", "Pelargonium zonale (L.) L'Hér."],
        "split": ["train", "test"],
        "organ": ["leaf", "leaf"],
    })


@pytest.fixture
def emb():
    return np.zeros((2, 512), np.float32)


def test_catalog_keys_match_the_pytorch_path(sub, emb):
    """`build_heads` reads split to select training rows; `load_background`
    filters on species_id and species_name."""
    assert set(pack(sub, emb, with_split=True)) == {
        "descriptor", "image_id", "species_id", "species_name", "split"}


def test_background_omits_split(sub, emb):
    """The background pool has no splits — its images are training negatives only."""
    assert "split" not in pack(sub, emb, with_split=False)


def test_ids_are_unicode_not_object(sub, emb):
    """np.load(allow_pickle=False) rejects object arrays, and pandas string
    columns become object arrays unless cast."""
    data = pack(sub, emb, with_split=True)
    for key in ("image_id", "species_id", "species_name", "split"):
        assert data[key].dtype.kind == "U", f"{key} is {data[key].dtype}"


def test_survives_a_savez_roundtrip_without_pickle(sub, emb, tmp_path):
    out = tmp_path / "c.npz"
    np.savez_compressed(out, **pack(sub, emb, with_split=True))
    loaded = np.load(out, allow_pickle=False)
    assert loaded["descriptor"].shape == (2, 512)
    assert loaded["species_name"][0] == "Sedum acre L."


def test_species_name_is_not_truncated(sub, emb):
    """Curation derives labels from the full name; truncating here would silently
    change the label space."""
    assert pack(sub, emb, with_split=True)["species_name"][1] == \
        "Pelargonium zonale (L.) L'Hér."


def test_row_count_matches_embeddings(sub, emb):
    data = pack(sub, emb, with_split=True)
    assert len(data["image_id"]) == len(data["descriptor"]) == 2
