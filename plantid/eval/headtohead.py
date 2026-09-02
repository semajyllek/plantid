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

## Names must be reconciled before anything is compared

The first trial run scored Pl@ntNet wrong for answering *Leuenbergeria bleo* to
our *Pereskia bleo* — the same plant under its current name. 9.5% of this
catalogue's names are a taxonomic generation behind (`INAT_FINDINGS.md`), so raw
string comparison systematically penalises whichever system uses current
nomenclature, which is not a fact about identification at all.

So each truth species carries an **alias set** — its catalogue name plus whatever
`resolve_taxa` maps it to — and a prediction matching any alias is correct. The
share of hits that needed an alias is reported, since it is the size of the
effect being corrected for.

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
# iNaturalist throttles the computer-vision endpoint harder than their general
# API: a 1.2s cadence ran clean for 99 requests and then returned 429 for the
# remaining 366. Back off and retry rather than discarding those observations —
# dropping them silently would have left a result computed on whichever subset
# happened to arrive before the limit.
INAT_SLEEP = 2.5
INAT_RETRIES = 5


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


def alias_sets(cache_dir=DATA_PROCESSED):
    """truth name -> {accepted names}, from `resolve_taxa` output.

    Regenerate with:
        python -m plantid.data.resolve_taxa --names <binomials> \
            --out data/processed/catalog_taxonomy.json --no-counts
    """
    path = cache_dir / "catalog_taxonomy.json"
    if not path.exists():
        return {}
    out = {}
    for r in json.loads(path.read_text()):
        name, resolved = r.get("catalog_name"), r.get("resolved")
        out.setdefault(name, {name})
        if resolved:
            out[name].add(resolved)
            out.setdefault(resolved, {resolved}).add(name)
    return out


def correct(pred, truth, aliases, genus=False):
    """Does `pred` name the same plant as `truth`, allowing for synonymy?"""
    if pred is None:
        return False
    accepted = aliases.get(truth, {truth})
    if genus:
        return genus_of(pred) in {genus_of(a) for a in accepted}
    return pred in accepted


def _cached(service, obs_id, fn):
    """Cache successes only.

    Caching an error would be permanent, and on a 500/day free tier a transient
    failure that poisons the cache costs a whole day's quota to discover.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{service}_{obs_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    result = fn()
    if "error" not in result:
        path.write_text(json.dumps(result))
    time.sleep(INAT_SLEEP if service.startswith("inat") else SLEEP)
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


def query_inat(image_path, token, lat=None, lng=None):
    """iNaturalist's *server* model. Seek's on-device model has no API.

    `lat`/`lng` switch on their geomodel, which their own figures put at +12pp of
    top-1 (75% vision-only -> 87% with it). Sending nothing is the fair
    comparison against Pl@ntNet and against us, since neither uses location;
    sending coordinates measures what a geographic prior is worth on *our*
    images, which is the claim `LOCATION_FINDINGS.md` scopes.
    """
    data = {}
    if lat is not None and lng is not None and lat == lat and lng == lng:
        data = {"lat": str(lat), "lng": str(lng)}
    r = None
    for attempt in range(INAT_RETRIES):
        with open(image_path, "rb") as fh:
            r = requests.post(INAT_URL, headers={**HEADERS, "Authorization": token},
                              files={"image": fh}, data=data, timeout=60)
        if r.status_code != 429:
            break
        time.sleep(INAT_SLEEP * 2 ** (attempt + 1))   # 5s, 10s, 20s, 40s, 80s
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


def score(rows, aliases=None):
    """Per-service species and genus top-1, and top-5 species containment."""
    aliases = aliases if aliases is not None else {}
    out = []
    for service in ("plantnet", "inat", "inat_geo"):
        got = [r for r in rows if r.get(f"{service}_top1") is not None]
        if not got:
            continue
        sp = np.array([correct(r[f"{service}_top1"], r["truth"], aliases) for r in got], float)
        gn = np.array([correct(r[f"{service}_top1"], r["truth"], aliases, genus=True)
                       for r in got], float)
        t5 = np.array([any(correct(p, r["truth"], aliases) for p in r[f"{service}_top5"])
                       for r in got], float)
        # how many hits needed synonymy — the size of the naming correction
        strict = np.array([r[f"{service}_top1"] == r["truth"] for r in got], float)
        lo, hi = boot(sp)
        out.append({"system": service, "n": len(got), "species_top1": sp.mean(),
                    "sp_lo": lo, "sp_hi": hi, "genus_top1": gn.mean(),
                    "species_top5": t5.mean(), "via_alias": sp.sum() - strict.sum()})
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
    ap.add_argument("--inat-token", default=os.environ.get("INAT_API_TOKEN"),
                    help="JWT from https://www.inaturalist.org/users/api_token (~24h)")
    ap.add_argument("--inat-geo", action="store_true",
                    help="also send coordinates, enabling their geomodel")
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
        geo = (r.lat, r.lon) if args.inat_geo else (None, None)
        calls = [("plantnet", args.plantnet_key, lambda: query_plantnet(img, args.plantnet_key)),
                 ("inat" + ("_geo" if args.inat_geo else ""), args.inat_token,
                  lambda: query_inat(img, args.inat_token, *geo))]
        for service, key, fn in calls:
            if not key:
                continue
            resp = _cached(service, r.obs_id, fn)
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
            for s in ("plantnet", "inat", "inat_geo") if f"{s}_error" in df}
    print(f"\nwrote {args.out}   errors: {errs}")
    print(score(rows, alias_sets()).round(4).to_string())
    print("\nNote: they choose from tens of thousands of taxa, we choose from 490 —")
    print("a tie is a loss for us. Compare against our own numbers accordingly.")


if __name__ == "__main__":
    main()
