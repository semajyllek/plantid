"""Build the working {leaf, bark, flower, plant_id} dataset from PlantNet-300K.

Selects species with adequate per-organ coverage, caps images per
(species, organ) to keep the working set manageable, re-derives train/val/test
splits per (species, organ) so that every combination has at least one val and
one test example, and downloads the images via the Pl@ntNet image API.

Usage:
    python -m plantid.data.build_dataset --min-per-organ 5 --cap 60
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.data.plantnet_audit import (
    load_metadata,
    load_species_names,
    species_organ_pivot,
    species_with_min_per_organ,
)

IMAGE_URL = "https://bs.plantnet.org/image/m/{image_id}"


def cap_and_split(df: pd.DataFrame, cap: int = 60, seed: int = 42) -> pd.DataFrame:
    """Cap each (species_id, organ) group at `cap` rows and assign a
    train/val/test split, guaranteeing at least one val and one test row per
    group (given the group has at least 3 rows).
    """
    rows = []
    rng_seed = seed
    for (_species_id, _organ), group in df.groupby(["species_id", "organ"]):
        group = group.sample(frac=1.0, random_state=rng_seed)
        rng_seed += 1
        group = group.head(cap)

        n = len(group)
        if n < 3:
            # too few to give every split a row; keep them for training only.
            # `require="either"` admits species covered by one organ, which can
            # leave a handful of images of the other — enough to learn from,
            # not enough to evaluate on.
            split = ["train"] * n
        else:
            n_val = max(1, round(0.15 * n))
            n_test = max(1, round(0.15 * n))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train, n_val, n_test = 1, 1, n - 2
            split = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        assert len(split) == n, f"split {len(split)} != group {n}"
        group = group.iloc[: len(split)].copy()
        group["split"] = split
        rows.append(group)

    return pd.concat(rows, ignore_index=True)


def select_dataset(min_per_organ: int = 5, cap: int = 60, seed: int = 42) -> pd.DataFrame:
    df = load_metadata()
    names = load_species_names()
    pivot = species_organ_pivot(df)

    species_ids = species_with_min_per_organ(pivot, list(ORGANS), min_per_organ)
    sub = df[df.species_id.isin(species_ids) & df.organ.isin(ORGANS)].copy()
    sub["species_name"] = sub["species_id"].map(names)

    out = cap_and_split(sub, cap=cap, seed=seed)
    return out[["image_id", "species_id", "species_name", "organ", "split"]]


def download_images(index: pd.DataFrame, out_dir, max_workers: int = 16) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def fetch(row):
        species_dir = out_dir / row.species_id / row.organ
        species_dir.mkdir(parents=True, exist_ok=True)
        dest = species_dir / f"{row.image_id}.jpg"
        if not dest.exists():
            for attempt in range(3):
                try:
                    resp = requests.get(IMAGE_URL.format(image_id=row.image_id), timeout=20)
                    resp.raise_for_status()
                    dest.write_bytes(resp.content)
                    break
                except requests.RequestException:
                    time.sleep(1.0 * (attempt + 1))
            else:
                return row.image_id, None
        return row.image_id, str(dest.relative_to(out_dir.parent))

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch, row) for row in index.itertuples()]
        for i, fut in enumerate(as_completed(futures)):
            image_id, path = fut.result()
            results[image_id] = path
            if (i + 1) % 1000 == 0:
                print(f"  downloaded {i + 1}/{len(index)}")

    index = index.copy()
    index["local_path"] = index["image_id"].map(results)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-per-organ", type=int, default=5)
    parser.add_argument("--cap", type=int, default=60)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    index = select_dataset(min_per_organ=args.min_per_organ, cap=args.cap)
    print(f"Selected {index.species_id.nunique()} species, {len(index)} images")
    print(index.groupby("organ").size())
    print(index.groupby(["organ", "split"]).size().unstack(fill_value=0))

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_PROCESSED / "plantnet_index.parquet"

    if not args.no_download:
        images_dir = DATA_PROCESSED / "images"
        index = download_images(index, images_dir)
        n_failed = index["local_path"].isna().sum()
        print(f"Failed downloads: {n_failed}")

    index.to_parquet(manifest_path, index=False)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
