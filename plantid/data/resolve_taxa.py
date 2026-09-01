"""Resolve catalogue binomials against iNaturalist's active taxonomy.

The catalogue inherits PlantNet's names, which are in places a taxonomic
generation behind: `Anemone nemorosa` is now *Anemonoides nemorosa*,
`Sedum kamtschaticum` is *Phedimus kamtschaticus*, `Perovskia atriplicifolia`
is *Salvia yangii*. Two things break as a result.

1. `inat_eval.fetch` queries by `taxon_name`, so those species return nothing
   and look like plants iNaturalist has no data on. 57 of the 86 catalogue
   species with no observation are this, not absence.
2. Buckets are assigned by binomial string match, so an observation filed under
   the modern name lands in an OOD bucket while being a plant the model is
   supposed to name.

iNat's `/taxa?q=` search is fuzzy and will return a *different* species
(`Anemone apennina` -> *Anemonoides blanda*), so a match is accepted only when:

  exact    the result's `matched_term` is the queried binomial, i.e. iNat lists
           our name as a synonym of the returned active taxon
  epithet  the specific epithet survives a genus transfer
           (`Anemone nemorosa` -> *Anemonoides nemorosa*)

Anything else is left unresolved rather than guessed at. Names that resolve to
nothing are usually not lookup failures: hybrid placeholders (`Pelargonium x`),
contested microspecies (the *Ophrys sphegodes* complex), or cultivated-only
plants whose iNat observations are all "casual" grade.

Usage:
    python -m plantid.data.resolve_taxa --names missing.txt --out resolved.json
"""

import argparse
import json
import time

import requests

TAXA_API = "https://api.inaturalist.org/v1/taxa"
OBS_API = "https://api.inaturalist.org/v1/observations"
HEADERS = {"User-Agent": "plantid-research/0.1 (species identification evaluation)"}
SLEEP = 1.1  # iNat asks for <= 60 requests/minute


def pick(results, query):
    """Strictest defensible match for `query`, or (None, None).

    Returns (taxon, how) where `how` is "exact" or "epithet"; see module
    docstring for why a plain first-result match is not safe.
    """
    q = query.lower()
    for t in results:
        if t.get("rank") == "species" and str(t.get("matched_term", "")).lower() == q:
            return t, "exact"
    epithet = q.split()[-1]
    for t in results:
        if t.get("rank") == "species" and str(t.get("name", "")).lower().split()[-1] == epithet:
            return t, "epithet"
    return None, None


def _get(url, params):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        return r.json() if r.status_code == 200 else {}
    except (requests.RequestException, ValueError):
        return {}


def resolve(names, count_observations=True, min_photos=2, verbose=True):
    """Resolve each binomial, optionally counting observations we could use.

    `n_multiphoto` counts, within the first page, observations with at least
    `min_photos` photos — the same filter `inat_eval` applies — so it says
    whether a targeted refetch would actually yield anything.
    """
    out = []
    for i, name in enumerate(names, 1):
        rec = {"catalog_name": name, "resolved": None, "taxon_id": None,
               "match": None, "matched_term": None, "n_research": 0, "n_multiphoto": 0}
        j = _get(TAXA_API, {"q": name, "is_active": "true", "per_page": 10})
        taxon, how = pick(j.get("results", []), name)
        if taxon:
            rec.update(resolved=taxon.get("name"), taxon_id=taxon.get("id"),
                       match=how, matched_term=taxon.get("matched_term"))
        time.sleep(SLEEP)

        if taxon and count_observations:
            j = _get(OBS_API, {"taxon_id": rec["taxon_id"], "quality_grade": "research",
                               "photos": "true", "per_page": 100, "locale": "en"})
            rec["n_research"] = j.get("total_results", 0)
            rec["n_multiphoto"] = sum(1 for o in j.get("results", [])
                                      if len(o.get("photos", [])) >= min_photos)
            time.sleep(SLEEP)

        out.append(rec)
        if verbose:
            print(f"[{i}/{len(names)}] {name!r} -> {rec['resolved']!r} [{rec['match']}] "
                  f"research={rec['n_research']} multi={rec['n_multiphoto']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="newline-separated binomials")
    ap.add_argument("--out", required=True, help="where to write the JSON result")
    ap.add_argument("--no-counts", action="store_true",
                    help="resolve names only; skip the observation-count request")
    args = ap.parse_args()

    names = [ln.strip() for ln in open(args.names) if ln.strip()]
    recs = resolve(names, count_observations=not args.no_counts)
    json.dump(recs, open(args.out, "w"), indent=1)

    n_res = sum(r["resolved"] is not None for r in recs)
    n_use = sum(r["n_multiphoto"] > 0 for r in recs)
    print(f"\n{n_res}/{len(recs)} resolved, {n_use} with usable observations")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
