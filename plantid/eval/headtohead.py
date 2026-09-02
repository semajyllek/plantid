"""Compare this system against Pl@ntNet and iNaturalist on identical photographs.

Every competitive claim in this repo so far rests on *their* published aggregates
against *our* measurements — different corpora, different label spaces, different
definitions of "accuracy". This sends one photograph to all three and scores them
the same way.

## The comparison is unfair to us, deliberately

Pl@ntNet chooses from tens of thousands of species and iNaturalist from 108,124
taxa; we choose from 490. **They are solving a much harder problem on the same
image, so a tie is a loss for us.** That is the point: if we cannot beat them
comfortably on the small set we specialise in, specialising bought nothing.

## Design

- **One observation per species.** With ~465 in-catalogue species and a 500/day
  Pl@ntNet free tier, one each maximises species coverage *and* makes every
  observation an independent cluster, so the interval needs no cluster bootstrap.
- **One photograph, the same one, to all three.** Our own fused answer is
  reported alongside but flagged, because fusing several photos is an advantage
  the others are not given here.
- **Responses are cached** by (service, obs_id). Re-runs cost no quota, and a
  crash halfway through loses nothing.

## Credentials

    PLANTNET_API_KEY   https://my.plantnet.org/  (free tier, 500 identifications/day)
    INAT_API_TOKEN     https://www.inaturalist.org/users/api_token  (JWT, ~24h)

Note this uploads photographs to third-party services. They are public
iNaturalist photos, so nothing private leaves the machine, but it is an outbound
transfer and the services will log it.

Usage:
    PYTHONPATH=. python -m plantid.eval.headtohead --n 200
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from plantid.config import DATA_PROCESSED
from plantid.data.curation import canonical_name, curated_name

PLANTNET_URL = "https://my-api.plantnet.org/v2/identify/all"
INAT_URL = "https://api.inaturalist.org/v1/computervision/score_image"
CACHE = DATA_PROCESSED / "headtohead"
HEADERS = {"User-Agent": "plantid-research/0.1 (encoder comparison)"}
SLEEP = 1.2


def sample_observations(n=200, seed=0, cache_dir=DATA_PROCESSED):
    """One in-catalogue observation per species, up to `n` species.

    One per species keeps observations independent — no two share a species — so
    a plain bootstrap is valid and species difficulty cannot be double-counted.
    """
    df = pd.read_parquet(cache_dir / "inat_observations.parquet")
    df = df[df["bucket"] == "in_catalog"].copy()
    df["truth"] = df["species_name"].map(lambda s: curated_name(s) or canonical_name(s))
    one = df.groupby("truth", group_keys=False).head(1)
    if n and n < len(one):
        one = one.sample(n, random_state=seed)
    return one.reset_index(drop=True)


def _cached(service, obs_id, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{service}_{obs_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    result = fn()
    path.write_text(json.dumps(result))
    time.sleep(SLEEP)
    return result


def query_plantnet(image_path, api_key):
    with open(image_path, "rb") as fh:
        r = requests.post(PLANTNET_URL, params={"api-key": api_key},
                          files=[("images", (Path(image_path).name, fh))],
                          data={"organs": ["auto"]}, headers=HEADERS, timeout=60)
    if r.status_code != 200:
        return {"error": r.status_code, "body": r.text[:300]}
    return {"results": [
        {"name": x["species"]["scientificNameWithoutAuthor"], "score": x["score"]}
        for x in r.json().get("results", [])[:10]]}


def query_inat(image_path, token):
    with open(image_path, "rb") as fh:
        r = requests.post(INAT_URL, headers={**HEADERS, "Authorization": token},
                          files={"image": fh}, timeout=60)
    if r.status_code != 200:
        return {"error": r.status_code, "body": r.text[:300]}
    out = []
    for x in r.json().get("results", [])[:10]:
        taxon = x.get("taxon") or {}
        out.append({"name": taxon.get("name"), "rank": taxon.get("rank"),
                    "score": x.get("combined_score", x.get("vision_score"))})
    return {"results": out}


def top1(resp):
    """Leading *species* name, or None. A genus-rank suggestion is not a species
    answer and must not be scored as one."""
    for r in (resp or {}).get("results", []):
        name = r.get("name")
        if not name:
            continue
        if r.get("rank") and r["rank"] != "species":
            continue
        if len(name.split()) >= 2:
            return canonical_name(name)
    return None


def genus_of(name):
    return name.split()[0] if name else None


def score(rows):
    """Per-service species and genus top-1, and top-5 species containment."""
    out = []
    for service in ("plantnet", "inat"):
        got = [r for r in rows if r.get(f"{service}_top1") is not None]
        sp = np.array([r[f"{service}_top1"] == r["truth"] for r in got], float)
        gn = np.array([genus_of(r[f"{service}_top1"]) == genus_of(r["truth"]) for r in got], float)
        t5 = np.array([r["truth"] in r[f"{service}_top5"] for r in got], float)
        out.append({"system": service, "n": len(got), "species_top1": sp.mean(),
                    "genus_top1": gn.mean(), "species_top5": t5.mean()})
    return pd.DataFrame(out).set_index("system")


def boot(x, n=4000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, float)
    m = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--plantnet-key", default=os.environ.get("PLANTNET_API_KEY"))
    ap.add_argument("--inat-token", default=os.environ.get("INAT_API_TOKEN"))
    ap.add_argument("--out", default=str(DATA_PROCESSED / "headtohead.parquet"))
    args = ap.parse_args()

    obs = sample_observations(args.n)
    print(f"{len(obs)} observations, one per species", flush=True)
    if not args.plantnet_key:
        print("! no PLANTNET_API_KEY — skipping Pl@ntNet")
    if not args.inat_token:
        print("! no INAT_API_TOKEN — skipping iNaturalist")

    rows = []
    for i, r in enumerate(obs.itertuples(), 1):
        img = str(DATA_PROCESSED / r.local_paths[0])
        row = {"obs_id": r.obs_id, "truth": r.truth, "image": img}
        for service, key, fn in (("plantnet", args.plantnet_key, query_plantnet),
                                 ("inat", args.inat_token, query_inat)):
            if not key:
                continue
            resp = _cached(service, r.obs_id, lambda: fn(img, key))
            row[f"{service}_top1"] = top1(resp)
            row[f"{service}_top5"] = [canonical_name(x["name"])
                                      for x in resp.get("results", [])[:5] if x.get("name")]
            row[f"{service}_error"] = resp.get("error")
        rows.append(row)
        if i % 25 == 0:
            print(f"  {i}/{len(obs)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    errs = {s: int(df.get(f"{s}_error", pd.Series(dtype=object)).notna().sum())
            for s in ("plantnet", "inat")}
    print(f"\nwrote {args.out}   errors: {errs}")
    print(score(rows).round(4).to_string())
    print("\nNote: they choose from tens of thousands of taxa, we choose from 490 —")
    print("a tie is a loss for us. Compare against our own numbers accordingly.")


if __name__ == "__main__":
    main()
