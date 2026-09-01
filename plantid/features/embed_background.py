"""Embed the `__OTHER__` background pool with a frozen encoder.

Separate entry point from `pretrained.py` because the background manifest has no
splits and no test role — these images exist only as negatives for the reject
class. Resumable: skips any organ whose cache already exists.

Usage:
    PYTHONPATH=. .venv-mps/bin/python -m plantid.features.embed_background
"""

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.data.curation import canonical_name

BACKGROUND_MANIFEST = "plantnet_background.parquet"


def cache_path(organ: str, variant: str = "bioclip2", cache_dir=DATA_PROCESSED):
    return cache_dir / f"background_{organ}_{variant}.npz"


def binomial(name: str) -> str:
    """'Sedum acre L.' -> 'Sedum acre'. iNaturalist carries no species_id, so
    name is the only key that joins the eval set to the PlantNet pools.

    Delegates to `curation.canonical_name`, which additionally keeps the hybrid
    marker: plain truncation turned `Fragaria × ananassa` into `Fragaria ×` and
    merged three different pelargoniums into one key. This is the *join* key, so
    it deliberately does not apply `curation.MERGE` — that belongs where class
    labels are formed.
    """
    return canonical_name(name)


def load_background(
    organ: str,
    exclude_species=None,
    exclude_names=None,
    variant: str = "bioclip2",
    cache_dir=DATA_PROCESSED,
) -> dict:
    """Background embeddings, with excluded species dropped.

    Two distinct exclusions, both mandatory in practice:

    `exclude_species` (by species_id) removes species that are in the *catalogue*.
    The pool was built against an earlier, smaller catalogue, so some of its
    species are now ones we want to recognise — training the reject class on
    those teaches the model to reject them.

    `exclude_names` (by binomial) removes species that appear in the
    *evaluation* set. Without it the reject class is trained on the very species
    rejection is then measured on: 47 of 183 near-OOD species, covering 32% of
    near-OOD observations, were in this pool. That made the near-OOD score
    partly in-sample. iNat rows have no species_id, hence the name-level match.

    Filtering at load time avoids re-embedding the pool.
    """
    npz = np.load(cache_path(organ, variant, cache_dir))
    data = {k: npz[k] for k in npz.files}
    keep = np.ones(len(data["species_id"]), bool)
    if exclude_species:
        keep &= ~np.isin(data["species_id"], list(exclude_species))
    if exclude_names:
        names = np.array([binomial(n) for n in data["species_name"]])
        keep &= ~np.isin(names, [binomial(n) for n in exclude_names])
    return {k: v[keep] for k, v in data.items()}


def catalog_species(cache_dir=DATA_PROCESSED, manifest="catalog_index.parquet") -> set:
    path = cache_dir / manifest
    if not path.exists():
        return set()
    return set(pd.read_parquet(path)["species_id"].unique())


def eval_species_names(cache_dir=DATA_PROCESSED, manifest="inat_observations.parquet") -> set:
    """Binomials of every species in the iNat evaluation set, for `exclude_names`."""
    path = cache_dir / manifest
    if not path.exists():
        return set()
    return {binomial(n) for n in pd.read_parquet(path)["species_name"].unique()}


def main(variant: str = "bioclip2", cache_dir=DATA_PROCESSED, batch_size: int = 64):
    from plantid.features.pretrained import embed_images, load_encoder

    bg = pd.read_parquet(cache_dir / BACKGROUND_MANIFEST)
    bg = bg[bg["local_path"].notna()].reset_index(drop=True)

    todo = [o for o in ORGANS if not cache_path(o, variant, cache_dir).exists()]
    if not todo:
        print("all organs already embedded")
        return
    print(f"to embed: {todo}")

    model, preprocess, device = load_encoder(variant)
    for organ in todo:
        sub = bg[bg["organ"] == organ].reset_index(drop=True)
        if sub.empty:
            continue
        paths = [str(cache_dir / p) for p in sub["local_path"]]
        emb = embed_images(paths, model, preprocess, device, batch_size=batch_size, desc=f"bg[{organ}]")
        np.savez_compressed(
            cache_path(organ, variant, cache_dir),
            descriptor=emb,
            image_id=np.asarray(sub["image_id"], dtype=str),
            species_id=np.asarray(sub["species_id"], dtype=str),
            species_name=np.asarray(sub["species_name"], dtype=str),
        )
        print(f"{organ}: {emb.shape} from {sub['species_id'].nunique()} background species", flush=True)


if __name__ == "__main__":
    main()
