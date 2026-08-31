"""Embed the iNaturalist evaluation observations with a frozen encoder.

Mirrors `embed_catalog.py` / `embed_background.py`. Unlike those, the unit here
is a *photo* rather than a catalogue row: an observation contributes several
photos of the same individual plant, and the observation-level posterior is
formed downstream by `eval/combiners.py`. The npz is therefore keyed by file
path, which is what `eval/rejection.py` and `eval/inat_fusion.py` join on.

Resumable: skips the cache if it already covers every photo in the manifest.

Usage:
    PYTHONPATH=. .venv-mps/bin/python -m plantid.features.embed_inat
"""

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED

MANIFEST = "inat_observations.parquet"


def cache_path(variant: str = "bioclip2", cache_dir=DATA_PROCESSED):
    return cache_dir / f"inat_{variant}.npz"


def load_inat(variant: str = "bioclip2", cache_dir=DATA_PROCESSED) -> dict:
    npz = np.load(cache_path(variant, cache_dir))
    return {k: npz[k] for k in npz.files}


def photo_paths(cache_dir=DATA_PROCESSED) -> list[str]:
    df = pd.read_parquet(cache_dir / MANIFEST)
    return [p for paths in df["local_paths"] for p in paths]


def main(variant: str = "bioclip2", cache_dir=DATA_PROCESSED, batch_size: int = 64):
    """Incremental: embeds only photos not already in the cache, then merges.

    Buckets get added over time (a region-restricted OOD set, say), and
    re-embedding thousands of unchanged photos to add a few hundred is pure
    waste — and would make the cache depend on when it was built.
    """
    from plantid.features.pretrained import embed_images, load_encoder

    paths = photo_paths(cache_dir)
    path = cache_path(variant, cache_dir)

    known_paths, known_emb = [], None
    if path.exists():
        cached = np.load(path)
        known_paths, known_emb = list(cached["path"]), cached["descriptor"]

    have = set(known_paths)
    todo = [p for p in paths if p not in have]
    if not todo:
        print(f"{path.name} already covers all {len(paths)} photos")
        return
    print(f"{len(have & set(paths))}/{len(paths)} cached; embedding {len(todo)} new photos", flush=True)

    model, preprocess, device = load_encoder(variant)
    new_emb = embed_images(todo, model, preprocess, device, batch_size=batch_size, desc="inat")

    all_paths = known_paths + todo
    all_emb = np.vstack([known_emb, new_emb]) if known_emb is not None else new_emb
    np.savez_compressed(path, descriptor=all_emb, path=np.asarray(all_paths, dtype=str))
    print(f"{path.name}: {all_emb.shape} covering {len(all_paths)} photos", flush=True)


if __name__ == "__main__":
    main()
