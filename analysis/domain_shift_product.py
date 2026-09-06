"""Secondary arm: the production head, scored on both acquisition sources.

The primary arm (`analysis/domain_shift.py`) fits its own matched heads to make
the 2x2 symmetric. This one takes the *shipped* head -- organ-routed, fitted by
`build_heads` on Pl@ntNet-300K `split=='train'` with an `__OTHER__` class -- and
scores it photo-level on held-out Pl@ntNet photographs and on the iNaturalist
observations, over the species both corpora share.

Departure from `DOMAIN_SHIFT_PREREG.md`, recorded rather than quietly dropped:
the pre-registration asked for the full three-way cascade at `p_ood = 0.20` on
both sides. It cannot be run symmetrically. The out-of-catalogue buckets are
iNaturalist-only, and the one Pl@ntNet-side negative pool on disk
(`background_*`) is what `__OTHER__` was *fitted* on, so using it as test
negatives would be in-sample. What is reported instead is leak-free and
symmetric: identification with `__OTHER__` masked out of the argmax, and
separately how much probability mass the source change moves onto `__OTHER__`.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.domain_shift_product --variants bioclip2
"""

import argparse

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.data.curation import curated_name
from plantid.eval.inat_fusion import OTHER, _l2, build_heads, build_router, photo_posteriors
from plantid.features.embed_catalog import load_catalog
from plantid.features.embed_inat import load_inat

N_BOOT = 2000
SEED = 0


def plantnet_test(variant: str):
    frames, embs = [], []
    for organ in ORGANS:
        d = load_catalog(organ, variant=variant)
        keep = d["split"] == "test"
        frames.append(pd.DataFrame({"species_name": d["species_name"][keep]}))
        embs.append(d["descriptor"][keep])
    df = pd.concat(frames, ignore_index=True)
    df["cn"] = [curated_name(n) for n in df["species_name"]]
    ok = df["cn"].notna().to_numpy()
    return df[ok].reset_index(drop=True), _l2(np.vstack(embs)[ok].astype(np.float32))


def inat_photos(variant: str):
    obs = pd.read_parquet(DATA_PROCESSED / "inat_observations.parquet")
    obs = obs[obs["bucket"] == "in_catalog"]
    df = pd.DataFrame([{"species_name": r.species_name, "path": p}
                       for r in obs.itertuples() for p in r.local_paths])
    df["cn"] = [curated_name(n) or n for n in df["species_name"]]
    cache = load_inat(variant)
    pos = {p: i for i, p in enumerate(cache["path"])}
    df = df[df["path"].isin(pos)].reset_index(drop=True)
    rows = np.array([pos[p] for p in df["path"]])
    return df, _l2(cache["descriptor"].astype(np.float32))[rows]


def score(E, df, heads, proj, router, classes, oi, species):
    """Per-species species/genus top-1 with __OTHER__ masked, and __OTHER__ mass."""
    P, _, _ = photo_posteriors(E, heads, proj, router, len(classes))
    other = P[:, oi].copy()
    P[:, oi] = -1.0                      # masked out of the argmax, not renormalised
    pred = classes[P.argmax(1)]
    truth = df["cn"].to_numpy()
    hit = pd.DataFrame({
        "cn": truth,
        "species": (pred == truth).astype(float),
        "genus": np.array([p.split()[0] for p in pred]) == np.array([t.split()[0] for t in truth]),
        "other": other,
    })
    g = hit.groupby("cn").mean(numeric_only=True)
    return {k: g[k].astype(float).reindex(species) for k in ("species", "genus", "other")}


def run(variant: str) -> pd.DataFrame:
    heads, proj, classes = build_heads(variant=variant)
    router, _ = build_router(variant=variant)
    oi = list(classes).index(OTHER)

    pn, Epn = plantnet_test(variant)
    ina, Eina = inat_photos(variant)
    species = sorted(set(pn["cn"]) & set(ina["cn"]))

    cells = {"pn": score(Epn, pn, heads, proj, router, classes, oi, species),
             "inat": score(Eina, ina, heads, proj, router, classes, oi, species)}

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(species), (N_BOOT, len(species)))
    rows = []
    for metric in ("species", "genus", "other"):
        a, b = cells["pn"][metric].to_numpy(), cells["inat"][metric].to_numpy()
        d = (b[draws].mean(1) - a[draws].mean(1))
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"variant": variant, "metric": metric,
                     "plantnet": round(np.nanmean(a), 4), "inat": round(np.nanmean(b), 4),
                     "shift": round(np.nanmean(b) - np.nanmean(a), 4),
                     "lo": round(lo, 4), "hi": round(hi, 4)})
    return pd.DataFrame(rows).assign(n_species=len(species))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["bioclip2"])
    ap.add_argument("--out", default=str(DATA_PROCESSED / "domain_shift_product.csv"))
    a = ap.parse_args()
    out = []
    for v in a.variants:
        t = run(v)
        print(f"\n== {v}"); print(t.to_string(index=False), flush=True)
        out.append(t)
    pd.concat(out).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
