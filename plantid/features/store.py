"""Compute and cache per-organ descriptors for the full working dataset.

Descriptor extraction takes a few minutes for the full dataset; this module
computes them once per organ and caches the result to
`data/processed/descriptors_{organ}.npz` so matching/eval code can load them
without recomputing.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.features import bark, flower, leaf

DESCRIBE_FNS = {"leaf": leaf.describe, "bark": bark.describe, "flower": flower.describe}


def _cache_path(organ: str, variant: str | None = None, cache_dir: Path = DATA_PROCESSED) -> Path:
    suffix = f"_{variant}" if variant else ""
    return cache_dir / f"descriptors_{organ}{suffix}.npz"


def compute_descriptors(index: pd.DataFrame, organ: str) -> dict:
    """Compute descriptors for every image of the given organ in `index`.

    Returns a dict of parallel arrays: image_id, species_id, species_name,
    split, descriptor (2D, n_images x descriptor_dim).
    """
    describe_fn = DESCRIBE_FNS[organ]
    sub = index[(index["organ"] == organ) & index["local_path"].notna()]

    image_ids, species_ids, species_names, splits, descs = [], [], [], [], []
    for row in tqdm(sub.itertuples(), total=len(sub), desc=f"descriptors[{organ}]"):
        img = cv2.imread(str(DATA_PROCESSED / row.local_path))
        if img is None:
            continue
        descs.append(describe_fn(img))
        image_ids.append(row.image_id)
        species_ids.append(row.species_id)
        species_names.append(row.species_name)
        splits.append(row.split)

    return {
        "image_id": np.array(image_ids),
        "species_id": np.array(species_ids),
        "species_name": np.array(species_names),
        "split": np.array(splits),
        "descriptor": np.stack(descs),
    }


def save_descriptors(data: dict, organ: str, variant: str | None = None, cache_dir: Path = DATA_PROCESSED) -> Path:
    path = _cache_path(organ, variant, cache_dir)
    np.savez_compressed(path, **data)
    return path


def load_descriptors(organ: str, variant: str | None = None, cache_dir: Path = DATA_PROCESSED) -> dict:
    npz = np.load(_cache_path(organ, variant, cache_dir))
    return {k: npz[k] for k in npz.files}


def compute_and_cache(
    index: pd.DataFrame, organ: str, cache_dir: Path = DATA_PROCESSED, force: bool = False
) -> dict:
    path = _cache_path(organ, cache_dir=cache_dir)
    if path.exists() and not force:
        return load_descriptors(organ, cache_dir=cache_dir)
    data = compute_descriptors(index, organ)
    save_descriptors(data, organ, cache_dir=cache_dir)
    return data


def main():
    index = pd.read_parquet(DATA_PROCESSED / "plantnet_index.parquet")
    for organ in ORGANS:
        data = compute_and_cache(index, organ, force=True)
        print(organ, data["descriptor"].shape)


if __name__ == "__main__":
    main()
