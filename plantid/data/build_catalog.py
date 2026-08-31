"""Build the v2 catalog: species with adequate leaf AND flower coverage.

Supersedes `build_dataset.py`, which required leaf+bark+flower and was therefore
capped at 87 species by bark scarcity (`ROADMAP.md` §0). Here leaf and flower are
required; bark is included opportunistically wherever a species has enough of it.

Writes into the same `data/processed/images/` tree as v1, keyed by
species_id/organ/image_id, so images already fetched are reused rather than
re-downloaded.

Usage:
    python -m plantid.data.build_catalog --min-leaf-flower 20 --cap 60
"""

import argparse

import pandas as pd

from plantid.config import DATA_PROCESSED
from plantid.data.build_dataset import cap_and_split, download_images
from plantid.data.plantnet_audit import load_metadata, load_species_names

CATALOG_MANIFEST = "catalog_index.parquet"
REQUIRED_ORGANS = ("leaf", "flower")
OPPORTUNISTIC_ORGANS = ("bark",)


def select_catalog(min_leaf_flower: int = 20, min_opportunistic: int = 5, cap: int = 60, seed: int = 42):
    """Species with >= min_leaf_flower of BOTH leaf and flower; bark added where available."""
    df = load_metadata()
    names = load_species_names()
    counts = df.groupby(["species_id", "organ"]).size().unstack(fill_value=0)

    ok = pd.Series(True, index=counts.index)
    for organ in REQUIRED_ORGANS:
        ok &= counts.get(organ, 0) >= min_leaf_flower
    species = set(counts.index[ok])

    keep_organs = list(REQUIRED_ORGANS) + list(OPPORTUNISTIC_ORGANS)
    sub = df[df.species_id.isin(species) & df.organ.isin(keep_organs)].copy()

    # drop opportunistic organs that are too thin for the species to be useful
    pair_counts = sub.groupby(["species_id", "organ"]).size()
    thin = {
        (s, o)
        for (s, o), n in pair_counts.items()
        if o in OPPORTUNISTIC_ORGANS and n < min_opportunistic
    }
    if thin:
        mask = [(s, o) not in thin for s, o in zip(sub.species_id, sub.organ)]
        sub = sub[mask]

    sub["species_name"] = sub["species_id"].map(names)
    out = cap_and_split(sub, cap=cap, seed=seed)
    out["role"] = "catalog"
    return out[["image_id", "species_id", "species_name", "organ", "split", "role"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-leaf-flower", type=int, default=20)
    ap.add_argument("--min-opportunistic", type=int, default=5)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--max-workers", type=int, default=32)
    args = ap.parse_args()

    index = select_catalog(args.min_leaf_flower, args.min_opportunistic, args.cap)
    print(f"catalog: {index.species_id.nunique()} species, {len(index)} images")
    print(index.groupby("organ").size().to_string())
    print(index.groupby(["organ", "split"]).size().unstack(fill_value=0).to_string())
    n_bark = index[index.organ == "bark"].species_id.nunique()
    print(f"species with opportunistic bark: {n_bark}")

    if not args.no_download:
        index = download_images(index, DATA_PROCESSED / "images", max_workers=args.max_workers)
        print(f"failed downloads: {int(index['local_path'].isna().sum())}")

    path = DATA_PROCESSED / CATALOG_MANIFEST
    index.to_parquet(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
