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

BACKGROUND_MANIFEST = "plantnet_background.parquet"


def cache_path(organ: str, variant: str = "bioclip2", cache_dir=DATA_PROCESSED):
    return cache_dir / f"background_{organ}_{variant}.npz"


def load_background(organ: str, exclude_species=None, variant: str = "bioclip2", cache_dir=DATA_PROCESSED) -> dict:
    """Background embeddings, with `exclude_species` dropped.

    The pool was built against an earlier, smaller catalog, so some of its
    species are now *in* the catalog. Those must not be used as negatives —
    training the reject class on a species you also want to recognise teaches
    the model to reject it. Filtering here avoids re-embedding: pass the current
    catalog's species ids and the overlap is dropped at load time.
    """
    npz = np.load(cache_path(organ, variant, cache_dir))
    data = {k: npz[k] for k in npz.files}
    if exclude_species:
        keep = ~np.isin(data["species_id"], list(exclude_species))
        data = {k: v[keep] for k, v in data.items()}
    return data


def catalog_species(cache_dir=DATA_PROCESSED, manifest="catalog_index.parquet") -> set:
    path = cache_dir / manifest
    if not path.exists():
        return set()
    return set(pd.read_parquet(path)["species_id"].unique())


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
