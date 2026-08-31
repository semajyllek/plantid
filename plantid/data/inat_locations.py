"""Backfill observation coordinates onto the iNaturalist evaluation manifest.

The original fetch kept only taxonomy and photos. Coordinates are what a
location prior needs — both to build it (which species occur where) and to test
it (was the true species plausible at the point the photo was taken).

iNat exposes up to 200 observations per request by id, so 2,600 observations is
~13 calls. Coordinates are absent or deliberately fuzzed for threatened species
(`obscured`), which is recorded rather than silently treated as missing.

Usage:
    python -m plantid.data.inat_locations
"""

import time

import pandas as pd
import requests

from plantid.config import DATA_PROCESSED
from plantid.data.inat_eval import API, HEADERS, MANIFEST, SLEEP

BATCH = 200


def fetch_locations(obs_ids, sleep=SLEEP):
    """obs_id -> {lat, lon, obscured, place_guess}. Missing ids are simply absent."""
    out = {}
    ids = [str(i) for i in obs_ids]
    for start in range(0, len(ids), BATCH):
        chunk = ids[start : start + BATCH]
        try:
            r = requests.get(API, params={"id": ",".join(chunk), "per_page": BATCH},
                             headers=HEADERS, timeout=45)
            results = r.json().get("results", []) if r.status_code == 200 else []
        except requests.RequestException:
            results = []
        for o in results:
            loc = o.get("location")
            if not loc:
                continue
            lat, lon = (float(v) for v in loc.split(","))
            out[o["id"]] = {"lat": lat, "lon": lon,
                            "obscured": bool(o.get("obscured")),
                            "place_guess": o.get("place_guess") or ""}
        time.sleep(sleep)
    return out


def main(cache_dir=DATA_PROCESSED):
    path = cache_dir / MANIFEST
    df = pd.read_parquet(path)
    todo = df if "lat" not in df.columns else df[df["lat"].isna()]
    print(f"{len(df)} observations, {len(todo)} needing coordinates")

    loc = fetch_locations(todo["obs_id"].tolist())
    print(f"resolved {len(loc)} of {len(todo)}")

    for col, key in (("lat", "lat"), ("lon", "lon"),
                     ("obscured", "obscured"), ("place_guess", "place_guess")):
        mapped = df["obs_id"].map(lambda i: loc.get(i, {}).get(key))
        df[col] = mapped if col not in df.columns else df[col].combine_first(mapped)

    df.to_parquet(path, index=False)
    have = df["lat"].notna()
    print(f"wrote {path}")
    print(f"  with coordinates: {have.sum()}/{len(df)} ({have.mean():.1%})")
    print(f"  obscured (coords fuzzed by iNat): {int(df['obscured'].fillna(False).sum())}")
    print(df[have].groupby("bucket").size().to_string())


if __name__ == "__main__":
    main()
