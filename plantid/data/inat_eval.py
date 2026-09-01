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
  regional_ood - same as distant_ood, but restricted to Europe / North America

`distant_ood` is drawn at random from global Plantae, so it is dominated by
mosses, ferns and tropical flora that a Europe/NA app would never be shown.
`regional_ood` is the deployment-realistic version: temperate plants a user
could plausibly photograph that are simply not in the catalogue. It is the
honest test of the reject decision.

**Contamination caveat**: iNaturalist feeds GBIF, which feeds TreeOfLife-200M,
which BioCLIP-2 was trained on (`DATA_STRATEGY.md`). So this tests source-shift
for the *head* while the *encoder* may well have seen these images. It is a real
test of the fusion and rejection logic, not of the embedding's novelty.

Usage:
    python -m plantid.data.inat_eval --per-bucket 150
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from plantid.config import DATA_PROCESSED
from plantid.data.curation import canonical_name, curated_name, curated_names

API = "https://api.inaturalist.org/v1/observations"
PLANTAE = 47126
# iNat continental places, for the deployment-realistic OOD bucket.
REGION_PLACE_IDS = (97391, 97394)  # Europe, North America
HEADERS = {"User-Agent": "plantid-research/0.1 (species identification evaluation)"}
MANIFEST = "inat_observations.parquet"
IMAGES_DIR = "images_inat"
MIN_PHOTOS = 2
MAX_PHOTOS = 6
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


def _rows(results, bucket, catalog_species, catalog_genera, min_photos=MIN_PHOTOS,
          name_map=None):
    """`name_map` rewrites iNat's current binomial to the catalogue's own.

    The catalogue carries PlantNet's pre-split names, so an observation of
    *Anemonoides nemorosa* is the plant `Anemone nemorosa` names. Without the
    rewrite it fails the `in_sp` test below and is discarded as out-of-catalogue
    — and, downstream, `build_observations` would score it against a head class
    that spells the species differently. The original name is kept as
    `inat_name` so the substitution stays auditable.
    """
    out = []
    for o in results:
        taxon = o.get("taxon") or {}
        if taxon.get("rank") != "species":
            continue
        inat_name = taxon.get("name") or ""
        if len(inat_name.split()) < 2:
            continue
        name = (name_map or {}).get(inat_name, inat_name)
        genus = name.split()[0]
        in_sp, in_gen = name in catalog_species, genus in catalog_genera
        if bucket == "in_catalog" and not in_sp:
            continue
        if bucket == "near_ood" and (in_sp or not in_gen):
            continue
        if bucket in ("distant_ood", "regional_ood") and (in_sp or in_gen):
            continue
        photos = [p["url"].replace("square", "medium") for p in o.get("photos", [])][:MAX_PHOTOS]
        if len(photos) < min_photos:
            continue
        out.append({"obs_id": o["id"], "species_name": name, "genus": genus,
                    "inat_name": inat_name, "bucket": bucket, "photo_urls": photos})
    return out


def fetch(catalog_species, catalog_genera, per_bucket=150, seed=0, min_photos=MIN_PHOTOS,
          per_species=3, per_genus=4, buckets=None, query_species=None, query_taxa=None,
          name_map=None):
    """Collect observations per bucket. `min_photos` filters at fetch time, so
    raising it means querying more broadly rather than discarding later.

    `per_species` / `per_genus` cap how many observations each query contributes,
    which is what limits the in-catalog and near-OOD bucket sizes: raise them to
    grow those buckets without issuing more requests.

    `query_taxa` queries by iNat `taxon_id` instead of `taxon_name`, which
    matters more than it sounds: `taxon_name` is a *fuzzy* match on the
    observations endpoint and returns related taxa, so a species with 56,684
    observations can come back as 100 pages of a commoner congener
    (`Lactuca sativa` -> *Lactuca serriola*). `taxon_id` is exact."""
    rows = []
    have = lambda b: len([r for r in rows if r["bucket"] == b])  # noqa: E731
    want = (lambda b: buckets is None or b in buckets)

    # in-catalog: query species by species. `query_species` / `query_taxa` narrow
    # which ones are asked for without changing what counts as in-catalogue, so
    # an existing evaluation set can be topped up for species it does not cover.
    if query_taxa is not None:
        to_query = [{"taxon_id": t} for t in sorted(query_taxa)]
    else:
        names = sorted(query_species) if query_species is not None else sorted(catalog_species)
        to_query = [{"taxon_name": n} for n in names]
    for params in pd.Series(to_query).sample(len(to_query), random_state=seed):
        if not want("in_catalog") or have("in_catalog") >= per_bucket:
            break
        j = _get({**params, "quality_grade": "research", "photos": "true",
                  "per_page": 100, "locale": "en"})
        rows += _rows(j.get("results", []), "in_catalog", catalog_species,
                      catalog_genera, min_photos, name_map)[:per_species]
        time.sleep(SLEEP)

    # near-OOD is rare under random sampling (genus in catalog, species not), so
    # query the catalog's *genera* directly and drop the catalog species.
    for genus in pd.Series(sorted(catalog_genera)).sample(len(catalog_genera), random_state=seed):
        if not want("near_ood") or have("near_ood") >= per_bucket:
            break
        j = _get({"taxon_name": genus, "quality_grade": "research", "photos": "true",
                  "per_page": 100, "locale": "en"})
        rows += _rows(j.get("results", []), "near_ood", catalog_species,
                      catalog_genera, min_photos, name_map)[:per_genus]
        time.sleep(SLEEP)

    # OOD buckets, both "neither species nor genus in the catalogue", sampled
    # from random plants. `regional_ood` adds a place filter and is the
    # deployment-realistic one: temperate plants a Europe/NA user could
    # plausibly photograph, rather than global mosses, ferns and tropical flora.
    for bucket, extra in (("distant_ood", {}),
                          ("regional_ood", {"place_id": ",".join(map(str, REGION_PLACE_IDS))})):
        if not want(bucket):
            continue
        page = 1
        while have(bucket) < per_bucket and page <= 60:
            j = _get({"taxon_id": PLANTAE, "quality_grade": "research", "photos": "true",
                      "per_page": 200, "page": page, "order_by": "random",
                      "locale": "en", **extra})
            res = j.get("results", [])
            if not res:
                break
            rows += _rows(res, bucket, catalog_species, catalog_genera, min_photos,
                          name_map)[: per_bucket - have(bucket)]
            page += 1
            time.sleep(SLEEP)

    return pd.DataFrame(rows).drop_duplicates(subset=["obs_id"])


def download(df, out_dir, max_workers=16, min_photos=MIN_PHOTOS):
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
    return df[df["local_paths"].map(len) >= min_photos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=150)
    ap.add_argument("--min-photos", type=int, default=MIN_PHOTOS)
    ap.add_argument("--per-species", type=int, default=3)
    ap.add_argument("--per-genus", type=int, default=4)
    ap.add_argument("--append", action="store_true",
                    help="add to the existing manifest instead of replacing the fetched buckets")
    ap.add_argument("--species-file", default=None,
                    help="newline-separated binomials to query for in_catalog; "
                         "does not change what counts as in-catalogue")
    ap.add_argument("--taxa-file", default=None,
                    help="JSON from plantid.data.resolve_taxa: queries in_catalog by "
                         "taxon_id and maps each resolved name back to its catalogue "
                         "binomial. Use for species whose catalogue name is a synonym.")
    ap.add_argument("--buckets", default=None,
                    help="comma-separated subset to fetch; existing rows for other buckets are kept")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    cat = pd.read_parquet(DATA_PROCESSED / "catalog_index.parquet")
    species = curated_names(cat["species_name"].unique())
    genera = {n.split()[0] for n in species}
    print(f"catalog: {len(species)} species, {len(genera)} genera")

    wanted = set(args.buckets.split(",")) if args.buckets else None
    qs = None
    if args.species_file:
        qs = [ln.strip() for ln in open(args.species_file) if ln.strip()]
        print(f"querying {len(qs)} specific species for in_catalog")

    taxa, name_map = None, None
    if args.taxa_file:
        recs = [r for r in json.load(open(args.taxa_file)) if r.get("taxon_id")]
        taxa = [r["taxon_id"] for r in recs]
        name_map = {r["resolved"]: r["catalog_name"] for r in recs
                    if r["resolved"] != r["catalog_name"]}
        print(f"querying {len(taxa)} taxon_ids for in_catalog, "
              f"{len(name_map)} of them under a renamed taxon")

    df = fetch(species, genera, per_bucket=args.per_bucket, min_photos=args.min_photos,
               per_species=args.per_species, per_genus=args.per_genus, buckets=wanted,
               query_species=qs, query_taxa=taxa, name_map=name_map)
    print(df.groupby("bucket").size().to_string())

    if not args.no_download:
        df = download(df, DATA_PROCESSED / IMAGES_DIR, min_photos=args.min_photos)
        print(f"after download: {len(df)} observations, {df['local_paths'].map(len).sum()} photos")
        print(df.groupby("bucket").size().to_string())

    path = DATA_PROCESSED / MANIFEST
    if (wanted or args.append) and path.exists():
        existing = pd.read_parquet(path)
        if args.append:
            # topping up: every existing row survives, new observations are added.
            # Replacing a bucket instead would silently discard the observations
            # already fetched for it.
            kept = existing
            print(f"append: keeping all {len(kept)} existing rows")
        else:
            # keep buckets we did not refetch, so their frozen calib/test splits
            # and already-embedded photos stay exactly as they were
            kept = existing[~existing["bucket"].isin(wanted)]
            print(f"merged: kept {len(kept)} rows from buckets {sorted(set(kept.bucket))}")
        df = pd.concat([kept, df], ignore_index=True).drop_duplicates(subset=["obs_id"])

    df.to_parquet(path, index=False)
    print(f"wrote {path} ({len(df)} observations)")
    print(df.groupby("bucket").size().to_string())


if __name__ == "__main__":
    main()


def relabel_buckets(cache_dir=DATA_PROCESSED, manifest=MANIFEST,
                    catalog="catalog_index.parquet"):
    """Recompute every observation's bucket against the *current* catalogue.

    Bucket membership is a property of the catalogue, not of the observation, so
    any change to the catalogue silently invalidates it. Growing from 261 to 530
    species moved 18.5% of the near-OOD bucket into the catalogue — those
    observations would otherwise have been scored as plants the model should
    reject while being plants it is supposed to name.

    Returns the relabelled frame and a summary of what moved.
    """
    df = pd.read_parquet(cache_dir / manifest)
    cat = pd.read_parquet(cache_dir / catalog)
    species = curated_names(cat["species_name"].unique())
    genera = {s.split()[0] for s in species}

    binom = df["species_name"].map(lambda n: curated_name(n) or canonical_name(n))
    in_sp = binom.isin(species)
    in_gen = binom.map(lambda n: n.split()[0] in genera)

    was = df["bucket"].copy()
    # a regional OOD observation stays regional when it is still out of catalogue
    regional = was == "regional_ood"
    new = pd.Series("distant_ood", index=df.index)
    new[regional] = "regional_ood"
    new[in_gen & ~in_sp] = "near_ood"
    new[in_sp] = "in_catalog"

    df = df.copy()
    df["bucket"] = new
    moved = pd.crosstab(was, new)
    return df, moved
