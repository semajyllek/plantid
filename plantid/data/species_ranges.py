"""Where each catalogue species actually occurs — the raw material for a location prior.

A user's coordinates are free at inference and enormously informative: most of
the catalogue cannot grow where they are standing. Measured against iNaturalist,
a 50 km radius cuts the 248-species catalogue to 83–92 candidates in European
cities and 26–55 in North American ones, i.e. a 2.7x–9.5x reduction in the label
space before the model looks at a single pixel.

Rather than asking "what grows near this point" once per query location, this
fetches the converse — a sample of observed coordinates per species — which is
the prior itself, computed once and reusable for any location.

Coordinates are an empirical sample, not a range map: a species observed 200
times near cities is biased toward where people walk. That is arguably the right
bias here, since the app is used where people walk.

Usage:
    python -m plantid.data.species_ranges --per-species 200
"""

import argparse
import time

import numpy as np
import pandas as pd
import requests

from plantid.config import DATA_PROCESSED
from plantid.data.inat_eval import API, HEADERS, SLEEP

RANGES = "species_ranges.parquet"
EARTH_KM = 6371.0


def fetch_species_points(binomials, per_species=200, sleep=SLEEP):
    rows = []
    for i, name in enumerate(binomials, 1):
        try:
            r = requests.get(API, params={
                "taxon_name": name, "quality_grade": "research", "geo": "true",
                "per_page": min(per_species, 200), "order_by": "random", "locale": "en",
            }, headers=HEADERS, timeout=45)
            results = r.json().get("results", []) if r.status_code == 200 else []
        except requests.RequestException:
            results = []
        for o in results:
            loc = o.get("location")
            taxon = (o.get("taxon") or {}).get("name", "")
            # the API matches loosely; keep only exact binomial hits
            if not loc or " ".join(taxon.split()[:2]) != name:
                continue
            lat, lon = (float(v) for v in loc.split(","))
            # obs_id is what lets the evaluation observations be excluded: the
            # range sample is drawn from the same pool the eval set came from,
            # and 15% of in-catalogue eval observations landed in it.
            rows.append({"species": name, "obs_id": o["id"], "lat": lat, "lon": lon})
        if i % 25 == 0:
            print(f"  {i}/{len(binomials)} species, {len(rows)} points", flush=True)
        time.sleep(sleep)
    return pd.DataFrame(rows)


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance from one point to many."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


class LocationPrior:
    """P(species | location) from a kernel-weighted sample of observations.

    Three design choices, each fixing a way the naive version misleads.

    **A distance kernel, not a hard radius.** Every species is capped at 200
    sampled points, so a hard cutoff leaves ~25% of genuine in-catalogue
    observations with *zero* records of their own species inside it — an
    artefact of the sampling cap, not of biology. A 0/1 radius would penalise a
    quarter of true positives and produce a null result that looks like "geography
    doesn't help". An exponential kernel degrades smoothly instead: 88% of
    observations have a same-species record within 250 km, 95% within 500 km.

    **Prevalence rescaling.** Capping every species at 200 points estimates
    P(location | species), not P(species | location); normalising across species
    then silently imposes a uniform species prior, so a globally rare plant with
    20 nearby points outranks an abundant one with 5. `prevalence` (total
    observations per species) rescales each species' sampled points back to its
    true weight.

    **A pseudo-count floor.** No species is ever ruled out — the catalogue is
    full of cultivated exotics growing far outside their natural range, and a
    hard geographic filter would turn every one of them into a guaranteed error.
    """

    def __init__(self, points: pd.DataFrame, bandwidth_km=250.0, alpha=0.5, prevalence=None):
        self.species = np.array(sorted(points["species"].unique()))
        self.index = {s: i for i, s in enumerate(self.species)}
        self.lat = points["lat"].to_numpy()
        self.lon = points["lon"].to_numpy()
        self.owner = np.array([self.index[s] for s in points["species"]])
        self.bandwidth_km, self.alpha = bandwidth_km, alpha
        sampled = np.bincount(self.owner, minlength=len(self.species)).astype(float)
        if prevalence:
            total = np.array([float(prevalence.get(s, np.nan)) for s in self.species])
            scale = np.where(np.isfinite(total) & (sampled > 0), total / np.maximum(sampled, 1), 1.0)
        else:
            scale = np.ones(len(self.species))
        self.scale = scale

    def weights(self, lat, lon):
        """Kernel-weighted, prevalence-rescaled mass per species at a point."""
        d = haversine_km(lat, lon, self.lat, self.lon)
        w = np.exp(-d / self.bandwidth_km)
        return np.bincount(self.owner, weights=w, minlength=len(self.species)) * self.scale

    def prior(self, lat, lon):
        m = self.weights(lat, lon) + self.alpha
        return m / m.sum()

    def rank_quantile(self, lat, lon, species):
        """Where this species ranks among the catalogue *at this location*, in [0,1].

        A within-location statistic, and that is the point. Absolute prior mass
        is largely a function of the coordinates alone — how well-recorded the
        area is — so it separates our evaluation buckets by their differing
        geography rather than by anything about the plant. In deployment a
        catalogue plant and an unknown one arrive from the same user at the same
        coordinates, where any location-only score has no discriminative power
        whatsoever. A rank quantile is invariant to that.
        """
        p = self.prior(lat, lon)
        i = self.index.get(species)
        if i is None:
            return float("nan")  # caller substitutes a neutral value
        return float((p < p[i]).mean())

    def plausible(self, lat, lon, min_weight=1.0):
        return set(self.species[self.weights(lat, lon) >= min_weight])


def load_prevalence(cache_dir=DATA_PROCESSED, name="species_prevalence.parquet"):
    path = cache_dir / name
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    return dict(zip(d["species"], d["total"]))


def load_prior(cache_dir=DATA_PROCESSED, exclude_obs_ids=None, **kw):
    """Load the prior, dropping any range point that is itself an evaluation
    observation. Without this the prior is partly built from the test point."""
    pts = pd.read_parquet(cache_dir / RANGES)
    if exclude_obs_ids is not None and "obs_id" in pts.columns:
        before = len(pts)
        pts = pts[~pts["obs_id"].isin(list(exclude_obs_ids))]
        print(f"location prior: dropped {before - len(pts)} range points that are evaluation observations")
    kw.setdefault("prevalence", load_prevalence(cache_dir))
    return LocationPrior(pts, **kw)


def eval_obs_ids(cache_dir=DATA_PROCESSED, manifest="inat_observations.parquet") -> set:
    path = cache_dir / manifest
    return set(pd.read_parquet(path)["obs_id"].unique()) if path.exists() else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species", type=int, default=200)
    args = ap.parse_args()

    cat = pd.read_parquet(DATA_PROCESSED / "catalog_index.parquet")
    names = sorted({" ".join(str(n).split()[:2]) for n in cat["species_name"].unique()})
    print(f"fetching up to {args.per_species} located observations for each of {len(names)} species")

    pts = fetch_species_points(names, per_species=args.per_species)
    pts.to_parquet(DATA_PROCESSED / RANGES, index=False)
    per = pts.groupby("species").size()
    print(f"\nwrote {DATA_PROCESSED / RANGES}: {len(pts)} points over {per.size} species")
    print(f"  points per species: median {per.median():.0f}, min {per.min()}, max {per.max()}")
    print(f"  species with no points: {len(names) - per.size}")


if __name__ == "__main__":
    main()
