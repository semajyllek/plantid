"""Build the `__OTHER__` background pool: PlantNet species *outside* the catalog.

The open-set experiments (`OPENSET_FINDINGS.md`) showed an explicit reject class
helps, but tested it with only 14 background species. This builds the full pool
— every PlantNet-300K species not in the current catalog, capped per
(species, organ) — so the lever can be exercised properly.

Deliberately kept separate from `build_dataset.py`, which selects the *catalog*
and requires all three organs per species. Background species need no such
requirement: their only job is to teach the model what "a plant, but not one of
ours" looks like, so any organ coverage is useful.

Usage:
    python -m plantid.data.build_background --cap 20 --min-per-organ 5
"""

import argparse

import pandas as pd

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.data.build_dataset import download_images
from plantid.data.plantnet_audit import load_metadata, load_species_names

BACKGROUND_MANIFEST = "plantnet_background.parquet"


def select_background(catalog_species: set, cap: int = 20, min_per_organ: int = 5, seed: int = 42) -> pd.DataFrame:
    """Every species outside `catalog_species` with >= min_per_organ of an organ."""
    df = load_metadata()
    names = load_species_names()
    df = df[df.organ.isin(ORGANS) & ~df.species_id.isin(catalog_species)].copy()

    counts = df.groupby(["species_id", "organ"]).size()
    keep = counts[counts >= min_per_organ].index
    df = df.set_index(["species_id", "organ"]).loc[df.set_index(["species_id", "organ"]).index.isin(keep)].reset_index()

    rows, rng_seed = [], seed
    for _, group in df.groupby(["species_id", "organ"]):
        rows.append(group.sample(min(len(group), cap), random_state=rng_seed))
        rng_seed += 1
    out = pd.concat(rows, ignore_index=True)
    out["species_name"] = out["species_id"].map(names)
    # Background images are training-only negatives; they never form a test set.
    out["split"] = "train"
    out["role"] = "background"
    return out[["image_id", "species_id", "species_name", "organ", "split", "role"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--min-per-organ", type=int, default=5)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--max-workers", type=int, default=32)
    args = ap.parse_args()

    # `catalog_index.parquet`, not `plantnet_index.parquet`: the latter is the
    # 87-species v1 working set, so building against it left 443 species in the
    # pool that the catalogue has since claimed — species the reject class would
    # then be taught to reject. `load_background` filters them at load time, so
    # nothing downstream was ever wrong, but the manifest on disk was.
    catalog = set(pd.read_parquet(DATA_PROCESSED / "catalog_index.parquet")["species_id"].unique())
    index = select_background(catalog, cap=args.cap, min_per_organ=args.min_per_organ)
    print(f"catalog species (excluded): {len(catalog)}")
    print(f"background: {index.species_id.nunique()} species, {len(index)} images")
    print(index.groupby("organ").size().to_string())

    if not args.no_download:
        index = download_images(index, DATA_PROCESSED / "images_background", max_workers=args.max_workers)
        print(f"failed downloads: {int(index['local_path'].isna().sum())}")

    path = DATA_PROCESSED / BACKGROUND_MANIFEST
    index.to_parquet(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
