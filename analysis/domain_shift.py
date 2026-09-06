"""The source-shift 2x2: fit a head per acquisition source, score it on both.

Pre-registered in `DOMAIN_SHIFT_PREREG.md`. The reframing that makes this
runnable is that `build_heads()` fits on Pl@ntNet-300K and every headline number
in this repo is scored on iNaturalist, so the published figures are already the
cross-source arm and what was missing is the within-source control.

Primary arm only: closed-set, organ-free, photo-level, macro-averaged over the
shared species, paired species-cluster bootstrap. No `__OTHER__` class and no
thresholds -- the only iNaturalist negatives on disk are the OOD evaluation
buckets, and fitting a reject class on them would leak into the rejection
numbers this repo publishes.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.domain_shift --variant bioclip2
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.data.curation import curated_name
from plantid.features.embed_catalog import load_catalog
from plantid.features.embed_inat import load_inat

B = 10          # training photographs per species per source (declared)
MIN_TEST = 3    # test photographs per species per source (declared)
SEED = 0
N_BOOT = 2000
INAT_TRAIN_FRAC = 0.6   # of a species' observations, split by observation


def _l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def plantnet_table(variant: str) -> pd.DataFrame:
    """One row per Pl@ntNet-300K photograph: curated name, split, embedding."""
    frames, embs = [], []
    for organ in ORGANS:
        d = load_catalog(organ, variant=variant)
        frames.append(pd.DataFrame({"species_name": d["species_name"], "split": d["split"]}))
        embs.append(d["descriptor"])
    df = pd.concat(frames, ignore_index=True)
    df["emb_row"] = np.arange(len(df))
    df["cn"] = [curated_name(n) for n in df["species_name"]]
    E = _l2(np.vstack(embs).astype(np.float32))
    return df[df["cn"].notna()].reset_index(drop=True), E


def inat_table(variant: str) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per in-catalogue iNaturalist photograph, tagged with its observation.

    The observation id is carried because the train/test split is by observation:
    several photographs share one individual plant, so a photo-level split would
    put the same plant on both sides.
    """
    obs = pd.read_parquet(DATA_PROCESSED / "inat_observations.parquet")
    obs = obs[obs["bucket"] == "in_catalog"]
    rows = [
        {"obs_id": r.obs_id, "species_name": r.species_name, "path": p,
         "cell": None if pd.isna(r.lat) else f"{round(r.lat)}_{round(r.lon)}"}
        for r in obs.itertuples()
        for p in r.local_paths
    ]
    df = pd.DataFrame(rows)
    df["cn"] = [curated_name(n) or n for n in df["species_name"]]

    cache = load_inat(variant)
    pos = {p: i for i, p in enumerate(cache["path"])}
    df = df[df["path"].isin(pos)].reset_index(drop=True)
    df["emb_row"] = [pos[p] for p in df["path"]]
    return df, _l2(cache["descriptor"].astype(np.float32))


def split_inat(df: pd.DataFrame, rng: np.random.Generator, unit: str = "obs_id") -> pd.Series:
    """Per species, assign whole observations to train or test.

    `unit="cell"` splits by 1-degree geographic cell instead, so train and test
    observations of a species come from different places. That is the check on
    whether `inat->inat` is inflated by two observations of the same population
    on the same day by the same photographer -- distinct plants by the
    observation id, but not an independent draw.
    """
    split = pd.Series("test", index=df.index)
    for _, g in df.groupby("cn"):
        g = g[g[unit].notna()] if unit != "obs_id" else g
        if g.empty:
            continue
        ids = np.asarray(g[unit].unique(), dtype=object)
        rng.shuffle(ids)
        n_tr = max(1, min(len(ids) - 1, round(len(ids) * INAT_TRAIN_FRAC)))
        split.loc[g.index[g[unit].isin(ids[:n_tr])]] = "train"
    return split


def subsample(df: pd.DataFrame, species, rng: np.random.Generator, B: int = B) -> np.ndarray:
    """Exactly B rows per species, so the two heads are fitted at equal capacity."""
    keep = []
    for s in species:
        idx = df.index[df["cn"] == s].to_numpy()
        keep.append(rng.choice(idx, B, replace=False))
    return np.concatenate(keep)


def fit(E, rows, labels):
    return LogisticRegression(max_iter=4000, C=10.0).fit(E[rows], labels)


def per_species(clf, E, df, species) -> pd.Series:
    """Top-1 accuracy within each species -- the unit the bootstrap resamples."""
    pred = clf.predict(E[df["emb_row"].to_numpy()])
    hit = pd.Series((pred == df["cn"].to_numpy()).astype(float), index=df.index)
    return hit.groupby(df["cn"]).mean().reindex(species)


def paired_bootstrap(cells: dict[str, pd.Series], species, rng) -> pd.DataFrame:
    """Resample species, not rows. Every cell is recomputed on the same draw, so
    differences between cells stay paired."""
    names = list(cells)
    M = np.column_stack([cells[k].to_numpy() for k in names])
    n = len(species)
    draws = np.array([M[rng.integers(0, n, n)].mean(axis=0) for _ in range(N_BOOT)])
    out = pd.DataFrame(draws, columns=names)
    out["pn_shift"] = out["pn->inat"] - out["pn->pn"]
    out["inat_shift"] = out["inat->pn"] - out["inat->inat"]
    return out


def run(variant: str, B: int = B, unit: str = "obs_id", geo_only: bool = False) -> dict:
    """`geo_only` keeps only observations carrying coordinates while still
    splitting by observation, so the `unit="cell"` arm can be compared against a
    control on the *same* species rather than against the full-corpus arm."""
    rng = np.random.default_rng(SEED)
    pn, Epn = plantnet_table(variant)
    ina, Eina = inat_table(variant)
    if unit == "cell" or geo_only:
        ina = ina[ina["cell"].notna()].reset_index(drop=True)
    ina["split"] = split_inat(ina, rng, unit)

    counts = pd.DataFrame({
        "pn_tr": pn[pn.split == "train"].groupby("cn").size(),
        "pn_te": pn[pn.split == "test"].groupby("cn").size(),
        "in_tr": ina[ina.split == "train"].groupby("cn").size(),
        "in_te": ina[ina.split == "test"].groupby("cn").size(),
    }).fillna(0)
    ok = ((counts.pn_tr >= B) & (counts.in_tr >= B)
          & (counts.pn_te >= MIN_TEST) & (counts.in_te >= MIN_TEST))  # declared rule
    species = sorted(counts.index[ok])

    pn_tr = pn[(pn.split == "train") & pn.cn.isin(species)]
    in_tr = ina[(ina.split == "train") & ina.cn.isin(species)]
    pn_te = pn[(pn.split == "test") & pn.cn.isin(species)].reset_index(drop=True)
    in_te = ina[(ina.split == "test") & ina.cn.isin(species)].reset_index(drop=True)

    pn_pick = subsample(pn_tr, species, rng, B)
    in_pick = subsample(in_tr, species, rng, B)
    h_pn = fit(Epn, pn_tr.loc[pn_pick, "emb_row"].to_numpy(), pn_tr.loc[pn_pick, "cn"].to_numpy())
    h_in = fit(Eina, in_tr.loc[in_pick, "emb_row"].to_numpy(), in_tr.loc[in_pick, "cn"].to_numpy())

    cells = {
        "pn->pn": per_species(h_pn, Epn, pn_te, species),
        "pn->inat": per_species(h_pn, Eina, in_te, species),
        "inat->inat": per_species(h_in, Eina, in_te, species),
        "inat->pn": per_species(h_in, Epn, pn_te, species),
    }
    boot = paired_bootstrap(cells, species, rng)
    return {"variant": variant, "B": B, "unit": unit + ("_geo" if geo_only else ""), "n_species": len(species),
            "cells": cells, "boot": boot, "n_pn_te": len(pn_te), "n_in_te": len(in_te)}


def report(res: dict) -> pd.DataFrame:
    b = res["boot"]
    rows = []
    for k in list(res["cells"]) + ["pn_shift", "inat_shift"]:
        point = res["cells"][k].mean() if k in res["cells"] else b[k].mean()
        lo, hi = np.percentile(b[k], [2.5, 97.5])
        rows.append({"variant": res["variant"], "B": res["B"], "split": res["unit"], "cell": k, "macro_top1": round(point, 4),
                     "lo": round(lo, 4), "hi": round(hi, 4)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["bioclip2"])
    ap.add_argument("--budget", type=int, default=B, help="training photos per species per source")
    ap.add_argument("--inat-split", choices=["obs_id", "cell"], default="obs_id")
    ap.add_argument("--geo-only", action="store_true",
                    help="restrict to observations with coordinates, still split by observation")
    ap.add_argument("--out", default=str(DATA_PROCESSED / "domain_shift.csv"))
    a = ap.parse_args()

    tables = []
    for v in a.variants:
        res = run(v, B=a.budget, unit=a.inat_split, geo_only=a.geo_only)
        print(f"\n== {v}: {res['n_species']} species, "
              f"{res['n_pn_te']} Pl@ntNet / {res['n_in_te']} iNat test photos", flush=True)
        t = report(res)
        print(t.to_string(index=False), flush=True)
        tables.append(t)
    pd.concat(tables).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
