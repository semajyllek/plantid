"""Fetch Oregon research-grade observations for the pre-registered safety species.

Observation-level throughout: an iNat observation is several photographs of one
individual plant, and it is the unit that must not straddle a train/test split.
Metadata comes from the API at its requested rate; images come from the static
CDN in parallel, which is not the rate-limited path.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

H = {"User-Agent": "plantid-research/0.1 (species identification evaluation)"}
OBS = "https://api.inaturalist.org/v1/observations"
OUT = Path("/private/tmp/claude-501/-Users-jameskelly-Documents-plantid/"
           "8f28d0ec-b6fd-4bd9-9ee3-a9ec10992c6b/scratchpad/safety")
MAX_OBS = 160          # per species
MAX_PHOTOS = 2         # per observation
SLEEP = 1.1

SPECIES = [
    "Conium maculatum", "Cicuta douglasii", "Daucus carota", "Anthriscus caucalis",
    "Heracleum maximum", "Lomatium nudicaule", "Lomatium dissectum",
    "Lomatium utriculatum", "Lomatium triternatum", "Osmorhiza berteroi",
    "Torilis arvensis", "Foeniculum vulgare", "Sanicula crassicaulis",
    "Sambucus racemosa", "Sambucus cerulea",
    "Rubus armeniacus", "Rubus ursinus", "Rubus parviflorus", "Rubus spectabilis",
    "Rubus laciniatus",
    "Vaccinium parvifolium", "Vaccinium ovatum", "Vaccinium membranaceum",
    "Toxicodendron diversilobum", "Digitalis purpurea",
    "Veratrum viride", "Veratrum californicum",
]


def observations(name):
    rows, page = [], 1
    while len(rows) < MAX_OBS and page <= 6:
        r = requests.get(OBS, params={
            "place_id": 10, "taxon_name": name, "quality_grade": "research",
            "photos": "true", "rank": "species", "per_page": 100, "page": page,
            "order_by": "votes",
        }, headers=H, timeout=60)
        res = r.json().get("results", [])
        if not res:
            break
        for o in res:
            tx = (o.get("taxon") or {}).get("name")
            if tx != name:            # taxon_name search can drift to relatives
                continue
            urls = [p["url"].replace("/square", "/medium")
                    for p in (o.get("photos") or [])[:MAX_PHOTOS] if p.get("url")]
            if urls:
                rows.append({"obs_id": o["id"], "species_name": name, "urls": urls})
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
    manifest = []
    for i, name in enumerate(SPECIES, 1):
        rows = observations(name)
        manifest += rows
        print(f"[{i:2d}/{len(SPECIES)}] {name:30s} {len(rows):4d} observations", flush=True)

    jobs, records = [], []
    for r in manifest:
        for k, url in enumerate(r["urls"]):
            slug = r["species_name"].replace(" ", "_")
            p = OUT / slug / f"{r['obs_id']}_{k}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((p, url))
            records.append({"obs_id": r["obs_id"], "species_name": r["species_name"],
                            "local_path": str(p)})
    print(f"\ndownloading {len(jobs)} images", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        ok = list(ex.map(grab, jobs))
    df = pd.DataFrame(records)[pd.Series(ok)]
    df.to_parquet(OUT / "manifest.parquet")
    print(f"kept {len(df)} images over {df.species_name.nunique()} species, "
          f"{df.obs_id.nunique()} observations")
    print(df.groupby("species_name").agg(images=("local_path", "size"),
                                         obs=("obs_id", "nunique")).to_string())


if __name__ == "__main__":
    main()
