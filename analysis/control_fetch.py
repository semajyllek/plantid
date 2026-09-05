"""Matched pre-cutoff control.

The holdout was drawn with order_by=created_at desc; the training corpus was
drawn with order_by=votes, which selects heavily-faved observations and may
select unusual or striking specimens. So "post-cutoff scores higher" could be a
sampling artifact rather than an absence of memorisation.

This draws the SAME way -- created_at desc -- from the window immediately BEFORE
the cutoff. Sampling now matches and only the time period differs, so any
remaining gap is attributable to the encoder having seen the images.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import holdout_fetch as hf

hf.OUT = Path(__file__).parent / "control"
_orig = hf.observations
def observations(name, seen):
    import requests, time
    rows, page = [], 1
    while len(rows) < hf.MAX_OBS and page <= 5:
        try:
            r = requests.get(hf.OBS, params={
                "place_id": 10, "taxon_name": name, "quality_grade": "research",
                "photos": "true", "rank": "species", "per_page": 100, "page": page,
                "created_d2": hf.CUTOFF, "order_by": "created_at", "order": "desc",
            }, headers=hf.H, timeout=60)
            res = r.json().get("results", [])
        except Exception as e:
            print(f"    retry {name} p{page}: {e}", flush=True); time.sleep(5); page += 1; continue
        if not res: break
        for o in res:
            if (o.get("taxon") or {}).get("name") != name or o["id"] in seen: continue
            urls = [p["url"].replace("/square","/medium")
                    for p in (o.get("photos") or [])[:hf.MAX_PHOTOS] if p.get("url")]
            if urls:
                rows.append({"obs_id": o["id"], "species_name": name, "urls": urls,
                             "created_at": o.get("created_at")})
        page += 1; time.sleep(hf.SLEEP)
    return rows[:hf.MAX_OBS]
hf.observations = observations

if __name__ == "__main__":
    hf.main()
