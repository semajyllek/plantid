"""Audit the Pl@ntNet-300K metadata: organ coverage, class balance, and
observation structure, ahead of building the {leaf, bark, flower, plant_id}
grouped dataset.

Source: https://github.com/plantnet/PlantNet-300K
Metadata downloaded to data/raw/plantnet300k/.
"""

import json

import pandas as pd

from plantid.config import DATA_RAW

PLANTNET_DIR = DATA_RAW / "plantnet300k"


def load_metadata() -> pd.DataFrame:
    with open(PLANTNET_DIR / "plantnet300K_metadata.json") as f:
        raw = json.load(f)

    df = pd.DataFrame.from_dict(raw, orient="index")
    df.index.name = "image_id"
    df = df.reset_index()
    return df


def load_species_names() -> dict:
    with open(PLANTNET_DIR / "plantnet300K_species_id_2_name.json") as f:
        return json.load(f)


def organ_counts(df: pd.DataFrame) -> pd.Series:
    return df["organ"].value_counts()


def species_organ_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = species_id, columns = organ, values = image count."""
    return pd.crosstab(df["species_id"], df["organ"])


def species_image_counts(df: pd.DataFrame) -> pd.Series:
    return df["species_id"].value_counts()


def observation_stats(df: pd.DataFrame) -> dict:
    obs_sizes = df.groupby("obs_id").size()
    obs_organ_counts = df.groupby("obs_id")["organ"].nunique()
    return {
        "n_observations": df["obs_id"].nunique(),
        "images_per_obs_mean": obs_sizes.mean(),
        "images_per_obs_median": obs_sizes.median(),
        "images_per_obs_max": obs_sizes.max(),
        "obs_with_multiple_organs": int((obs_organ_counts > 1).sum()),
        "obs_with_multiple_organs_pct": 100 * (obs_organ_counts > 1).mean(),
    }


def species_with_min_per_organ(pivot: pd.DataFrame, organs: list[str], min_count: int) -> pd.Index:
    """Species ids with at least `min_count` images for every organ in `organs`."""
    missing = [o for o in organs if o not in pivot.columns]
    if missing:
        return pd.Index([])
    mask = (pivot[organs] >= min_count).all(axis=1)
    return pivot.index[mask]


def main():
    df = load_metadata()
    names = load_species_names()

    print(f"Total images: {len(df)}")
    print(f"Total species: {df['species_id'].nunique()}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print()

    print("Organ distribution:")
    oc = organ_counts(df)
    for organ, count in oc.items():
        print(f"  {organ:>10}: {count:>7}  ({100 * count / len(df):.1f}%)")
    print()

    pivot = species_organ_pivot(df)
    print(f"Species x organ pivot shape: {pivot.shape}")
    print(pivot.sum(axis=0))
    print()

    counts = species_image_counts(df)
    print("Per-species image count distribution:")
    print(counts.describe())
    print(f"  top 10% of species hold {100 * counts.sort_values(ascending=False).head(len(counts) // 10).sum() / counts.sum():.1f}% of images")
    print()

    obs = observation_stats(df)
    print("Observation stats:")
    for k, v in obs.items():
        print(f"  {k}: {v}")
    print()

    organs_present = list(oc.index)
    for min_count in (1, 5, 10, 20, 50):
        n_species = len(species_with_min_per_organ(pivot, organs_present, min_count))
        print(f"Species with >= {min_count} images for ALL of {organs_present}: {n_species}")


if __name__ == "__main__":
    main()
