"""Deployable coverage for a narrow, user-chosen catalogue.

`subset_frontier.py` ranks encoders by AUROC, which is threshold-free and so says
nothing about what a deployment actually answers. This fits the three-way
cascade from `eval/rejection.py` -- species / genus / decline -- on a calibration
split and reports coverage and precision on a held-out one, at a *stated*
out-of-catalogue prevalence.

Everything decision-shaped is imported rather than reimplemented: `UTILITY` was
declared before any of this was fitted, and `deployment_weights` is what stops
the eval set's incidental composition from choosing the operating point.

Buckets, mapped onto the ones `rejection.py` already names:

  in_catalog   test rows of the K chosen species
  near_ood     test rows of catalogue species *outside* the chosen set that
               share a genus with it -- the congeners the user did not pick,
               which is the failure mode narrow catalogues actually have
  distant_ood  held-out background pool rows

Genus answers can be correct on near_ood: that is the point of the cascade, and
the reason declining is not the only safe response to an unchosen congener.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.subset_coverage bioclip2 mobileclip2_s2 20
"""

import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from plantid.eval.rejection import (
    DECLINE,
    GENUS,
    SPECIES,
    decide,
    deployment_weights,
    fit_thresholds,
    genus_matrix,
    make_splits,
)
from plantid.features.embed_background import catalog_species, load_background

DP = "data/processed"
ORGANS = ["leaf", "flower"]
OTHER = "__OTHER__"
N_DRAWS = 12
BG_TRAIN_FRAC = 0.6
P_OOD_GRID = (0.5, 0.2, 0.1)
# near/distant shares among out-of-catalogue queries, as in rejection.OOD_MIX_*
OOD_MIX = {"near_ood": 0.32, "distant_ood": 0.68}


def _l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def load(variant):
    cs = catalog_species()
    cat, bg = {}, {}
    for organ in ORGANS:
        d = np.load(f"{DP}/catalog_{organ}_{variant}.npz", allow_pickle=True)
        cat[organ] = (_l2(d["descriptor"]), d["species_name"].astype(str), d["split"].astype(str))
        bg[organ] = _l2(load_background(organ, exclude_species=cs, variant=variant)["descriptor"])
    return cat, bg


def one_draw(cat, bg, species, rng):
    """-> DataFrame of per-observation scores and outcomes, ready for the cascade."""
    keep = set(species)
    keep_genera = {s.split()[0] for s in keep}

    Xtr, ytr = [], []
    rows = []  # (X, true_species, bucket)
    for organ in ORGANS:
        E, names, split = cat[organ]
        m = np.array([n in keep for n in names])
        tr = m & (split == "train")
        Xtr.append(E[tr]); ytr.append(names[tr])

        te = m & (split == "test")
        rows.append((E[te], names[te], "in_catalog"))

        out = (~m) & (split == "test")
        og = np.array([n.split()[0] in keep_genera for n in names])
        rows.append((E[out & og], names[out & og], "near_ood"))

        B = bg[organ]
        cut = rng.permutation(len(B))
        n = int(BG_TRAIN_FRAC * len(B))
        Xtr.append(B[cut[:n]]); ytr.append(np.full(n, OTHER))
        far = B[cut[n:]]
        rows.append((far, np.full(len(far), OTHER), "distant_ood"))

    clf = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(
        np.vstack(Xtr), np.concatenate(ytr)
    )
    classes = np.array(clf.classes_)
    oi = int(np.flatnonzero(classes == OTHER)[0])
    mask = classes != OTHER
    gmat, ug = genus_matrix(classes, mask)

    X = np.vstack([r[0] for r in rows])
    truth = np.concatenate([r[1] for r in rows])
    bucket = np.concatenate([[r[2]] * len(r[0]) for r in rows])

    P = clf.predict_proba(X)
    cataP = P[:, mask]
    species_conf = cataP.max(1)
    gscore = cataP @ gmat.T
    genus_conf = gscore.max(1)

    sp_pred = classes[mask][cataP.argmax(1)]
    gn_pred = ug[gscore.argmax(1)]
    true_genus = np.array([t.split()[0] if t != OTHER else OTHER for t in truth])

    return pd.DataFrame({
        "species_conf": species_conf,
        "genus_conf": genus_conf,
        "species_ok": sp_pred == truth,
        "genus_ok": gn_pred == true_genus,
        "in_catalog": bucket == "in_catalog",
        "bucket": bucket,
        "species": truth,
        "genus": true_genus,
    })


def evaluate(df, p_ood):
    """Fit the cascade on calib at `p_ood`, score on test.

    -> (coverage, precision, species_rate)

    `species_rate` is the share of *in-catalogue* observations answered at
    species level, and it is not optional. On a congener-dense catalogue a genus
    answer is close to vacuous -- if all 20 chosen species are Sedum, "it is a
    Sedum" carries no information -- yet `UTILITY` still scores it at 0.5 and the
    fit will happily buy coverage with it. Coverage and precision alone therefore
    flatter exactly the arm where the answers are least useful.
    """
    fold = make_splits(df, seed=0)
    cal, te = df[fold == "calib"], df[fold == "test"]
    if cal.empty or te.empty:
        return np.nan, np.nan, np.nan

    w_cal = deployment_weights(cal["bucket"].to_numpy(), p_ood=p_ood, ood_mix=OOD_MIX)
    (tg, ts), _ = fit_thresholds(
        cal["species_conf"].to_numpy(), cal["genus_conf"].to_numpy(),
        cal["species_ok"].to_numpy(), cal["genus_ok"].to_numpy(),
        cal["in_catalog"].to_numpy(), sample_weight=w_cal,
    )

    lv = decide(te["species_conf"].to_numpy(), te["genus_conf"].to_numpy(), tg, ts)
    w = deployment_weights(te["bucket"].to_numpy(), p_ood=p_ood, ood_mix=OOD_MIX)
    answered = lv != DECLINE
    correct = ((lv == SPECIES) & te["species_ok"].to_numpy()) | \
              ((lv == GENUS) & te["genus_ok"].to_numpy())
    cov = w[answered].sum() / w.sum()
    prec = w[answered & correct].sum() / w[answered].sum() if answered.any() else np.nan
    inc = te["in_catalog"].to_numpy()
    sp_rate = (lv[inc] == SPECIES).mean() if inc.any() else np.nan
    return cov, prec, sp_rate


def make_draws(all_species, K, rng, hard):
    by_genus = defaultdict(list)
    for s in all_species:
        by_genus[s.split()[0]].append(s)
    multi = [g for g, v in by_genus.items() if len(v) >= 2]
    out = []
    for _ in range(N_DRAWS):
        if hard:
            picked = []
            for g in rng.permutation(multi):
                picked += by_genus[g]
                if len(picked) >= K:
                    break
            out.append(picked[:K])
        else:
            out.append(list(rng.choice(all_species, K, replace=False)))
    return out


def main(variants, Ks):
    ref, _ = load("bioclip2")
    all_species = np.array(sorted(set(ref["leaf"][1]) | set(ref["flower"][1])))
    loaded = {v: load(v) for v in variants}

    for K in Ks:
        for hard in (False, True):
            arm = "HARD (congeners)" if hard else "EASY (random)"
            ds = make_draws(all_species, K, np.random.default_rng(K + hard), hard)
            print(f"\n=== K={K}  {arm}  {N_DRAWS} draws ===", flush=True)
            print(f"{'encoder':18s} " + " ".join(f"  p_ood={p:<4} cov/prec/sp" for p in P_OOD_GRID), flush=True)
            for v in variants:
                cat, bg = loaded[v]
                frames = [one_draw(cat, bg, s, np.random.default_rng(i)) for i, s in enumerate(ds)]
                cells = []
                for p in P_OOD_GRID:
                    r = np.array([evaluate(f, p) for f in frames])
                    cells.append(f"{np.nanmean(r[:,0]):.3f}/{np.nanmean(r[:,1]):.3f}/{np.nanmean(r[:,2]):.3f}")
                print(f"{v:18s} " + "   ".join(cells), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    vs = [a for a in args if not a.isdigit()] or ["bioclip2", "mobileclip2_s2", "mobileclip2_s0"]
    ks = [int(a) for a in args if a.isdigit()] or [20]
    main(vs, ks)
