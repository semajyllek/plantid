"""Per-(species, organ) outlier / label-noise detection.

For each (species_id, organ) group, every descriptor dimension is z-scored
within the group, then each image's Euclidean distance to the group centroid
is measured in that normalized space. Distances are converted to a robust
"modified z-score" using the median and median absolute deviation (MAD)
rather than the mean/stdev, since a handful of badly mislabeled images
should not be allowed to drag the reference statistics around (Iglewicz &
Hoaglin, 1993). Images with modified z-score above MAD_THRESHOLD are flagged
as likely mislabeled or off-target — e.g. the Sedum sediforme case found
during the Phase 1 audit.

Groups smaller than MIN_GROUP_SIZE are left unflagged: with only a handful of
samples, both the centroid and the MAD are too noisy to support a reliable
call.
"""

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from plantid.config import DATA_PROCESSED
from plantid.features import bark, flower, leaf

DESCRIBE_FNS = {"leaf": leaf.describe, "bark": bark.describe, "flower": flower.describe}

MAD_THRESHOLD = 3.5
MIN_GROUP_SIZE = 5


def compute_descriptors(index: pd.DataFrame) -> pd.DataFrame:
    """Compute the per-organ descriptor for every image in `index`.

    Returns a DataFrame with one row per successfully-read image, carrying
    the original metadata columns plus a `descriptor` column (1D np.ndarray).
    """
    records = []
    for row in tqdm(index.itertuples(), total=len(index), desc="descriptors"):
        describe_fn = DESCRIBE_FNS.get(row.organ)
        if describe_fn is None or pd.isna(row.local_path):
            continue
        img = cv2.imread(str(DATA_PROCESSED / row.local_path))
        if img is None:
            continue
        records.append(
            {
                "image_id": row.image_id,
                "species_id": row.species_id,
                "species_name": row.species_name,
                "organ": row.organ,
                "split": row.split,
                "descriptor": describe_fn(img),
            }
        )
    return pd.DataFrame.from_records(records)


def _group_outlier_scores(group: pd.DataFrame) -> pd.DataFrame:
    out = group.drop(columns=["descriptor"]).copy()

    if len(group) < MIN_GROUP_SIZE:
        out["distance"] = np.nan
        out["modified_z"] = np.nan
        out["is_outlier"] = False
        return out

    X = np.stack(group["descriptor"].to_numpy())
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0
    z = (X - mean) / std
    dist = np.sqrt((z**2).sum(axis=1))

    median = np.median(dist)
    mad = np.median(np.abs(dist - median))
    modified_z = np.zeros_like(dist) if mad < 1e-12 else 0.6745 * (dist - median) / mad

    out["distance"] = dist
    out["modified_z"] = modified_z
    out["is_outlier"] = modified_z > MAD_THRESHOLD
    return out


def detect_outliers(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Flag per-image outliers within each (species_id, organ) group."""
    groups = [
        _group_outlier_scores(group)
        for _, group in descriptors.groupby(["species_id", "organ"], sort=False)
    ]
    return pd.concat(groups, ignore_index=True)


def main():
    index = pd.read_parquet(DATA_PROCESSED / "plantnet_index.parquet")
    print(f"Computing descriptors for {len(index)} images...")
    descriptors = compute_descriptors(index)

    print("Scoring outliers within each (species, organ) group...")
    results = detect_outliers(descriptors)

    out_path = DATA_PROCESSED / "outlier_scores.parquet"
    results.to_parquet(out_path, index=False)
    print(f"Wrote {len(results)} rows to {out_path}")

    n_flagged = int(results["is_outlier"].sum())
    print(f"\nFlagged {n_flagged} / {len(results)} images as outliers (modified_z > {MAD_THRESHOLD})")
    print(results.groupby("organ")["is_outlier"].sum())

    print("\nTop 15 outliers overall:")
    top = results.sort_values("modified_z", ascending=False).head(15)
    print(top[["species_name", "organ", "image_id", "modified_z"]].to_string(index=False))


if __name__ == "__main__":
    main()
