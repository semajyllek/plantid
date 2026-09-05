"""Temporal holdout: iNaturalist observations uploaded after BioCLIP-2's training data.

BioCLIP-2 was released June 2025 and trained on TreeOfLife-200M, whose GBIF
snapshot necessarily predates that. PlantCLEF2024 is a 2024 model and MobileCLIP2
trains on pre-2025 web data. So observations *uploaded* after 2025-07-01 are
image-level novel to all three encoders while being drawn from the same platform,
the same photographers and the same species.

That isolates memorisation, which is the contamination question. It does not test
domain shift -- same photographic culture -- so it is not a substitute for
photographs taken on a different camera by a different person.

`created_at` is recorded rather than assumed, so the cutoff can be verified after
the fact instead of trusted.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

H = {"User-Agent": "plantid-research/0.1 (species identification evaluation)"}
OBS = "https://api.inaturalist.org/v1/observations"
BASE = Path(__file__).parent
OUT = BASE / "holdout"
CUTOFF = "2025-07-01"
MAX_OBS = 90
MAX_PHOTOS = 2
SLEEP = 1.1

SPECIES = [
    "Pseudotsuga menziesii", "Gaultheria shallon", "Berberis aquifolium",
    "Polystichum munitum", "Acer circinatum", "Arbutus menziesii",
    "Thuja plicata", "Tsuga heterophylla", "Trillium ovatum", "Camassia quamash",
    "Achillea millefolium", "Oxalis oregana", "Symphoricarpos albus",
    "Holodiscus discolor", "Physocarpus capitatus", "Cornus sericea",
    "Lonicera involucrata", "Ribes sanguineum", "Salix scouleriana",
    "Quercus garryana",
]


def observations(name, seen):
    rows, page = [], 1
    while len(rows) < MAX_OBS and page <= 5:
        try:
            r = requests.get(OBS, params={
                "place_id": 10, "taxon_name": name, "quality_grade": "research",
                "photos": "true", "rank": "species", "per_page": 100, "page": page,
                "created_d1": CUTOFF, "order_by": "created_at", "order": "desc",
            }, headers=H, timeout=60)
            res = r.json().get("results", [])
        except Exception as e:
            print(f"    retry {name} p{page}: {e}", flush=True)
            time.sleep(5)
            page += 1
            continue
        if not res:
            break
        for o in res:
            if (o.get("taxon") or {}).get("name") != name or o["id"] in seen:
                continue
            urls = [p["url"].replace("/square", "/medium")
                    for p in (o.get("photos") or [])[:MAX_PHOTOS] if p.get("url")]
            if urls:
                rows.append({"obs_id": o["id"], "species_name": name, "urls": urls,
                             "created_at": o.get("created_at")})
        page += 1
        time.sleep(SLEEP)
    return rows[:MAX_OBS]


def grab(job):
    path, url = job
    if path.exists():
        return True
    try:
        r = requests.get(url, headers=H, timeout=45)
        if r.status_code == 200 and len(r.content) > 2000:
            path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # never reuse an observation that is already in the training corpus
    seen = set(pd.read_parquet(BASE / "common" / "manifest.parquet")["obs_id"].astype(int))
    print(f"excluding {len(seen)} observations already fetched\n", flush=True)

    manifest = []
    for i, name in enumerate(SPECIES, 1):
        rows = observations(name, seen)
        manifest += rows
        print(f"[{i:2d}/{len(SPECIES)}] {name:26s} {len(rows):4d} post-{CUTOFF}", flush=True)

    jobs, records = [], []
    for r in manifest:
        for k, url in enumerate(r["urls"]):
            p = OUT / r["species_name"].replace(" ", "_") / f"{r['obs_id']}_{k}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((p, url))
            records.append({"obs_id": r["obs_id"], "species_name": r["species_name"],
                            "local_path": str(p), "created_at": r["created_at"]})
    print(f"\ndownloading {len(jobs)} images", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        ok = list(ex.map(grab, jobs))
    df = pd.DataFrame(records)[pd.Series(ok)].drop_duplicates("local_path")
    df.to_parquet(OUT / "manifest.parquet")
    print(f"kept {len(df)} images / {df.obs_id.nunique()} observations / "
          f"{df.species_name.nunique()} species")
    print(f"upload dates {df.created_at.min()[:10]} .. {df.created_at.max()[:10]}")


if __name__ == "__main__":
    main()
