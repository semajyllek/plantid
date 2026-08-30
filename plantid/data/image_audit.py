"""Audit downloaded images: resolution, aspect ratio, corrupt files, per organ.

Usage:
    python -m plantid.data.image_audit
"""

import cv2
import pandas as pd

from plantid.config import DATA_PROCESSED


def audit_images(manifest_path=None) -> pd.DataFrame:
    manifest_path = manifest_path or (DATA_PROCESSED / "plantnet_index.parquet")
    index = pd.read_parquet(manifest_path)

    rows = []
    for row in index.itertuples():
        path = DATA_PROCESSED / row.local_path if pd.notna(row.local_path) else None
        record = {
            "image_id": row.image_id,
            "species_id": row.species_id,
            "organ": row.organ,
            "split": row.split,
            "readable": False,
            "width": None,
            "height": None,
            "channels": None,
        }
        if path is not None and path.exists():
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                record["readable"] = True
                record["height"], record["width"] = img.shape[:2]
                record["channels"] = 1 if img.ndim == 2 else img.shape[2]
        rows.append(record)

    return pd.DataFrame(rows)


def main():
    df = audit_images()

    n_total = len(df)
    n_unreadable = (~df["readable"]).sum()
    print(f"Total images in manifest: {n_total}")
    print(f"Unreadable / missing: {n_unreadable}")
    if n_unreadable:
        print(df[~df["readable"]].groupby(["species_id", "organ"]).size())
    print()

    ok = df[df["readable"]]
    print("Resolution stats by organ:")
    print(ok.groupby("organ")[["width", "height"]].describe().T)
    print()

    ok = ok.copy()
    ok["aspect_ratio"] = ok["width"] / ok["height"]
    print("Aspect ratio stats by organ:")
    print(ok.groupby("organ")["aspect_ratio"].describe())
    print()

    print("Channel counts by organ:")
    print(ok.groupby(["organ", "channels"]).size())


if __name__ == "__main__":
    main()
