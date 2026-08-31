"""Fetch real multi-organ observations from iNaturalist for cross-source evaluation.

Every fusion number in this repo rests on *synthetic* groups: one leaf, one bark
and one flower drawn independently from the same species in the same corpus
(`data/groups.py`). That assumes the organ images are conditionally independent
given species, which is exactly what the sampler enforces and not what a user
does. A real user photographs one individual plant, so errors are correlated.

iNaturalist observations bundle several photos of the *same individual*, which
is the structure PlantNet-300K structurally cannot provide (`obs_id` is 1:1 with
images). This module pulls those observations into three buckets:

  in_catalog   - species is in the 261-species catalog       -> should be accepted
  near_ood     - species not in catalog, but its genus is    -> hard rejection
  distant_ood  - neither species nor genus in the catalog    -> easy rejection

**Contamination caveat**: iNaturalist feeds GBIF, which feeds TreeOfLife-200M,
which BioCLIP-2 was trained on (`DATA_STRATEGY.md`). So this tests source-shift
for the *head* while the *encoder* may well have seen these images. It is a real
test of the fusion and rejection logic, not of the embedding's novelty.

Usage:
    python -m plantid.data.inat_eval --per-bucket 150
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from plantid.config import DATA_PROCESSED

API = "https://api.inaturalist.org/v1/observations"
PLANTAE = 47126
HEADERS = {"User-Agent": "plantid-research/0.1 (species identification evaluation)"}
MANIFEST = "inat_observations.parquet"
IMAGES_DIR = "images_inat"
MIN_PHOTOS = 2
MAX_PHOTOS = 4
SLEEP = 1.1  # be polite: iNat asks for <= 60 requests/minute


def _get(params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2.0 * (attempt + 1))
    return {"results": []}


def _rows(results, bucket, catalog_species, catalog_genera):
    out = []
    for o in results:
        taxon = o.get("taxon") or {}
        if taxon.get("rank") != "species":
            continue
        name = taxon.get("name") or ""
        if len(name.split()) < 2:
            continue
        genus = name.split()[0]
        in_sp, in_gen = name in catalog_species, genus in catalog_genera
        if bucket == "in_catalog" and not in_sp:
            continue
        if bucket == "near_ood" and (in_sp or not in_gen):
            continue
        if bucket == "distant_ood" and (in_sp or in_gen):
            continue
        photos = [p["url"].replace("square", "medium") for p in o.get("photos", [])][:MAX_PHOTOS]
        if len(photos) < MIN_PHOTOS:
            continue
        out.append({"obs_id": o["id"], "species_name": name, "genus": genus,
                    "bucket": bucket, "photo_urls": photos})
    return out


def fetch(catalog_species, catalog_genera, per_bucket=150, seed=0):
    rows = []

    # in-catalog: query species by species, two observations each
    species = sorted(catalog_species)
    rng = pd.Series(species).sample(min(len(species), per_bucket), random_state=seed)
    for name in rng:
        j = _get({"taxon_name": name, "quality_grade": "research", "photos": "true",
                  "per_page": 10, "locale": "en"})
        rows += _rows(j.get("results", []), "in_catalog", catalog_species, catalog_genera)[:2]
        time.sleep(SLEEP)
        if len([r for r in rows if r["bucket"] == "in_catalog"]) >= per_bucket:
            break

    # OOD buckets: sample plants at random and filter into near / distant
    page = 1
    while True:
        counts = {b: len([r for r in rows if r["bucket"] == b]) for b in ("near_ood", "distant_ood")}
        if min(counts.values()) >= per_bucket or page > 40:
            break
        j = _get({"taxon_id": PLANTAE, "quality_grade": "research", "photos": "true",
                  "per_page": 200, "page": page, "order_by": "random", "locale": "en"})
        res = j.get("results", [])
        if not res:
            break
        for b in ("near_ood", "distant_ood"):
            if counts[b] < per_bucket:
                rows += _rows(res, b, catalog_species, catalog_genera)[: per_bucket - counts[b]]
        page += 1
        time.sleep(SLEEP)

    return pd.DataFrame(rows).drop_duplicates(subset=["obs_id"])


def download(df, out_dir, max_workers=16):
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r.obs_id, i, url) for r in df.itertuples() for i, url in enumerate(r.photo_urls)]

    def fetch_one(job):
        obs_id, i, url = job
        dest = out_dir / f"{obs_id}_{i}.jpg"
        if dest.exists():
            return obs_id, i, str(dest)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return obs_id, i, str(dest)
        except requests.RequestException:
            return obs_id, i, None

    with ThreadPoolExecutor(max_workers) as ex:
        got = list(ex.map(fetch_one, jobs))
    ok = {(o, i): p for o, i, p in got if p}
    df = df.copy()
    df["local_paths"] = [[ok[(r.obs_id, i)] for i in range(len(r.photo_urls)) if (r.obs_id, i) in ok]
                         for r in df.itertuples()]
    return df[df["local_paths"].map(len) >= MIN_PHOTOS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=150)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    cat = pd.read_parquet(DATA_PROCESSED / "catalog_index.parquet")
    species = {" ".join(n.split()[:2]) for n in cat["species_name"].unique()}
    genera = {n.split()[0] for n in species}
    print(f"catalog: {len(species)} species, {len(genera)} genera")

    df = fetch(species, genera, per_bucket=args.per_bucket)
    print(df.groupby("bucket").size().to_string())

    if not args.no_download:
        df = download(df, DATA_PROCESSED / IMAGES_DIR)
        print(f"after download: {len(df)} observations, {df['local_paths'].map(len).sum()} photos")
        print(df.groupby("bucket").size().to_string())

    df.to_parquet(DATA_PROCESSED / MANIFEST, index=False)
    print(f"wrote {DATA_PROCESSED / MANIFEST}")


if __name__ == "__main__":
    main()
