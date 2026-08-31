"""Embed the v2 catalog with a frozen encoder.

Mirrors `embed_background.py`: resumable per organ, writes
`catalog_{organ}_{variant}.npz` carrying the split labels the background pool
doesn't need.

Usage:
    PYTHONPATH=. .venv-mps/bin/python -m plantid.features.embed_catalog
"""

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED, ORGANS

CATALOG_MANIFEST = "catalog_index.parquet"


def cache_path(organ: str, variant: str = "bioclip2", cache_dir=DATA_PROCESSED):
    return cache_dir / f"catalog_{organ}_{variant}.npz"


def load_catalog(organ: str, variant: str = "bioclip2", cache_dir=DATA_PROCESSED) -> dict:
    npz = np.load(cache_path(organ, variant, cache_dir))
    return {k: npz[k] for k in npz.files}


def main(variant: str = "bioclip2", cache_dir=DATA_PROCESSED, batch_size: int = 64):
    from plantid.features.pretrained import embed_images, load_encoder

    idx = pd.read_parquet(cache_dir / CATALOG_MANIFEST)
    idx = idx[idx["local_path"].notna()].reset_index(drop=True)

    todo = [o for o in ORGANS if not cache_path(o, variant, cache_dir).exists()]
    if not todo:
        print("all organs already embedded")
        return
    print(f"to embed: {todo}", flush=True)

    model, preprocess, device = load_encoder(variant)
    for organ in todo:
        sub = idx[idx["organ"] == organ].reset_index(drop=True)
        if sub.empty:
            continue
        paths = [str(cache_dir / p) for p in sub["local_path"]]
        emb = embed_images(paths, model, preprocess, device, batch_size=batch_size, desc=f"cat[{organ}]")
        np.savez_compressed(
            cache_path(organ, variant, cache_dir),
            descriptor=emb,
            image_id=np.asarray(sub["image_id"], dtype=str),
            species_id=np.asarray(sub["species_id"], dtype=str),
            species_name=np.asarray(sub["species_name"], dtype=str),
            split=np.asarray(sub["split"], dtype=str),
        )
        print(f"{organ}: {emb.shape} from {sub['species_id'].nunique()} species", flush=True)


if __name__ == "__main__":
    main()
