"""Three-way answer decision: species, genus, or decline.

Replaces the single accept/reject threshold on `P(__OTHER__)`, which accepted
60% of same-genus and 39% of unrelated out-of-catalogue plants.

The rule reports at the most specific level it can defend:

    decline            if genus confidence   < t_genus
    report genus only  if species confidence < t_species
    report species     otherwise

Two facts make this the right shape. First, the scores are *nested* — exactly,
by construction:

    max_c P(c)  <=  max_g sum_{c in g} P(c)  <=  sum_{c != OTHER} P(c) = 1 - P(OTHER)

so the cascade is well-ordered and the state "species confident, genus not" is
unreachable. Second, of out-of-catalogue observations the old rule accepted,
81.9% of the same-genus ones had the *correct genus* while 0% of the unrelated
ones did — so genus-level answering converts most near-OOD errors into correct
answers, and genus confidence is the sharpest detector of the distant ones.

Thresholds are fitted by maximising expected utility on a calibration split,
never on test. Utilities are declared in `UTILITY` rather than implied by a
metric: an "is the answer useful" objective is degenerate here, because genus
accuracy (0.95-0.98) beats species accuracy (0.76-0.83) and "always answer
genus" would win it while deleting the product's main feature.

Usage:
    PYTHONPATH=. python -m plantid.eval.rejection --emb data/processed/inat_bioclip2.npz
"""

import argparse

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED
from plantid.eval.combiners import trimmed
from plantid.eval.inat_fusion import OTHER, _l2, build_heads, build_router, photo_posteriors

# Declared before fitting, deliberately.
#
# lam=0.5: a genus answer is worth half a species answer. mu=2: a wrong answer
# costs twice a right one, encoding the precision-first operating point.
#
# lam is the sensitive parameter and there is a sharp transition just below 0.5:
# at lam=0.25 the rule answers species on ~53% of in-catalogue observations, at
# lam=0.5 on ~12%, at lam=0.75 never. That looks alarming until precision is
# measured at a realistic out-of-catalogue rate rather than this eval set's
# 59.5%: at a 20% OOD rate lam=0.5 gives 99.0% precision at 58% coverage,
# whereas lam=0.25 gives 95.0% at 50%. Genus-heavy answering is what buys the
# precision, and species accuracy (~0.88) caps what species-level answering can
# ever reach. See REJECTION_FINDINGS.md for the full surface.
UTILITY = {"species_correct": 1.0, "genus_correct": 0.5, "wrong": -2.0,
           "decline_ood": 1.0, "decline_in_catalog": 0.0}

# Share of real queries that are out-of-catalogue. Unknown, and the single
# biggest lever on reported precision, so results are always reported as a
# curve over it rather than at this eval set's incidental 59.5%.
OOD_PREVALENCE_GRID = (0.6, 0.4, 0.2, 0.1)
NEAR_OOD_SHARE = 0.32  # of out-of-catalogue queries, the fraction in a catalogue genus

SPECIES, GENUS, DECLINE = "species", "genus", "decline"
IN_CATALOG = "in_catalog"

# Cluster to split on, per bucket. Splitting near-OOD by species would still put
# the same genus in both halves, and the decision being fitted is genus-level.
SPLIT_CLUSTER = {"in_catalog": "species", "near_ood": "genus",
                 "distant_ood": "species", "regional_ood": "species"}

# Which out-of-catalogue buckets stand in for real traffic, and their shares.
# `distant_ood` is drawn from global Plantae and is dominated by mosses, ferns
# and tropical flora a Europe/NA app would never be shown; `regional_ood` is the
# same rule restricted to Europe/N.America and is the deployment-realistic one.
OOD_MIX_GLOBAL = {"near_ood": 0.32, "distant_ood": 0.68}
OOD_MIX_REGIONAL = {"near_ood": 0.32, "regional_ood": 0.68}


def genus_matrix(classes, mask):
    """(n_genera, n_species) indicator, and the genus label per column."""
    genera = np.array([c.split()[0] for c in classes[mask]])
    ug = np.unique(genera)
    return np.stack([(genera == g).astype(float) for g in ug]), ug


def scores(P, mask, oi, gmat):
    """The nested chain, per observation: (species_conf, genus_conf, one_minus_other)."""
    catalog = P[:, mask]
    return catalog.max(1), (catalog @ gmat.T).max(1), 1.0 - P[:, oi]


def decide(species_conf, genus_conf, t_genus, t_species):
    """Vectorised cascade -> array of SPECIES / GENUS / DECLINE."""
    out = np.full(len(species_conf), SPECIES, dtype=object)
    out[species_conf < t_species] = GENUS
    out[genus_conf < t_genus] = DECLINE
    return out


def utility(levels, species_ok, genus_ok, in_catalog, weights=None):
    """Per-observation utility of the decision taken. Vectorised: threshold
    fitting evaluates this tens of thousands of times."""
    w = {**UTILITY, **(weights or {})}
    levels = np.asarray(levels, dtype=object)
    species_ok = np.asarray(species_ok, bool)
    genus_ok = np.asarray(genus_ok, bool)
    in_catalog = np.asarray(in_catalog, bool)

    is_dec, is_sp = levels == DECLINE, levels == SPECIES
    is_gn = ~is_dec & ~is_sp
    return (
        is_dec * np.where(in_catalog, w["decline_in_catalog"], w["decline_ood"])
        + is_sp * np.where(species_ok, w["species_correct"], w["wrong"])
        + is_gn * np.where(genus_ok, w["genus_correct"], w["wrong"])
    )


def fit_thresholds(species_conf, genus_conf, species_ok, genus_ok, in_catalog,
                   weights=None, n_grid=60):
    """Grid-search (t_genus, t_species) maximising mean utility. Calibration only."""
    g_grid = np.quantile(genus_conf, np.linspace(0, 1, n_grid))
    s_grid = np.quantile(species_conf, np.linspace(0, 1, n_grid))
    best, best_u = (0.0, 0.0), -np.inf
    for tg in g_grid:
        for ts in s_grid:
            u = utility(decide(species_conf, genus_conf, tg, ts),
                        species_ok, genus_ok, in_catalog, weights).mean()
            if u > best_u:
                best, best_u = (float(tg), float(ts)), float(u)
    return best, best_u


def make_splits(df, seed=0):
    """Assign 'calib'/'test' per row, splitting on the cluster for each bucket.

    Clustered because ~6 observations share a species and species difficulty is
    the dominant variance component; an observation-level split would put the
    same difficulty on both sides.
    """
    rng = np.random.RandomState(seed)
    fold = pd.Series("test", index=df.index, dtype=object)
    for bucket, group in df.groupby("bucket"):
        key = SPLIT_CLUSTER.get(bucket, "species")
        clusters = np.array(sorted(group[key].unique()))
        rng.shuffle(clusters)
        calib = set(clusters[: len(clusters) // 2])
        fold[group.index[group[key].isin(calib)]] = "calib"
    return fold


def cluster_bootstrap(values, clusters, n=2000, seed=0):
    """Resample *clusters*, not rows. Unclustered CIs have twice given this
    project effects that failed to replicate."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(clusters)))
    index = {c: np.flatnonzero(np.asarray(clusters) == c) for c in uniq}
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([index[c] for c in pick])
        out.append(np.mean(values[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def build_observations(emb_path, cache_dir=DATA_PROCESSED, min_photos=2, combiner=trimmed,
                       temperatures=None):
    """One row per observation: truth, bucket, and the fused posterior scores."""
    heads, proj, classes = build_heads(cache_dir=cache_dir)
    router, router_acc = build_router(cache_dir=cache_dir)
    oi = int(np.flatnonzero(classes == OTHER)[0])
    mask = np.ones(len(classes), bool)
    mask[oi] = False
    gmat, ug = genus_matrix(classes, mask)

    z = np.load(emb_path)
    E = _l2(z["descriptor"])
    path_index = {p: i for i, p in enumerate(z["path"])}
    df = pd.read_parquet(cache_dir / "inat_observations.parquet")

    rows = []
    for r in df.itertuples():
        idx = [path_index[p] for p in r.local_paths if p in path_index]
        if len(idx) < min_photos:
            continue
        P, organs, _ = photo_posteriors(E[idx], heads, proj, router, len(classes),
                                        temperatures=temperatures)
        fused = combiner(P, list(organs), oi)[None, :]
        sc, gc, omo = scores(fused, mask, oi, gmat)
        pred = classes[mask][fused[:, mask].argmax(1)][0]
        gpred = ug[(fused[:, mask] @ gmat.T).argmax(1)][0]
        binom = " ".join(str(r.species_name).split()[:2])
        rows.append({
            "bucket": r.bucket, "species": binom, "genus": r.genus,
            "pred_species": pred, "pred_genus": gpred,
            "species_conf": float(sc[0]), "genus_conf": float(gc[0]),
            "one_minus_other": float(omo[0]),
            "species_ok": pred == binom, "genus_ok": gpred == r.genus,
            "in_catalog": r.bucket == IN_CATALOG, "n_photos": len(idx),
        })
    out = pd.DataFrame(rows)
    out["fold"] = make_splits(out)
    return out, router_acc


def precision_coverage(levels, species_ok, genus_ok, buckets, p_ood=None, ood_mix=None):
    """Precision among answers given, and coverage, at an assumed OOD rate.

    `p_ood=None` uses the eval set as-is, which is ~60% out-of-catalogue — far
    higher than deployment is likely to be, and therefore pessimistic. Passing a
    prevalence re-weights the buckets to that assumption. `ood_mix` chooses
    which OOD buckets stand in for real traffic and in what proportion.
    """
    buckets = np.asarray(buckets)
    answered = levels != DECLINE
    correct = ((levels == SPECIES) & species_ok) | ((levels == GENUS) & genus_ok)
    if p_ood is None:
        w = np.ones(len(levels), float)
    else:
        mix = ood_mix or OOD_MIX_GLOBAL
        w = np.zeros(len(levels), float)
        n_in = max((buckets == IN_CATALOG).sum(), 1)
        w[buckets == IN_CATALOG] = (1 - p_ood) / n_in
        total = sum(mix.values())
        for bucket, share in mix.items():
            m = buckets == bucket
            if m.any():
                w[m] = p_ood * (share / total) / m.sum()
    answered_mass = (w * answered).sum()
    return (float((w * answered * correct).sum() / max(answered_mass, 1e-12)),
            float(answered_mass / w.sum()))


def prevalence_table(levels, species_ok, genus_ok, buckets, grid=OOD_PREVALENCE_GRID, ood_mix=None):
    rows = []
    for p in grid:
        prec, cov = precision_coverage(levels, species_ok, genus_ok, buckets,
                                       p_ood=p, ood_mix=ood_mix)
        rows.append({"ood_rate": p, "precision": prec, "coverage": cov})
    return pd.DataFrame(rows).set_index("ood_rate")


def summarise(df, t_genus, t_species, weights=None):
    """Per-bucket outcome breakdown plus mean utility."""
    levels = decide(df["species_conf"].values, df["genus_conf"].values, t_genus, t_species)
    u = utility(levels, df["species_ok"].values, df["genus_ok"].values,
                df["in_catalog"].values, weights)
    out = df.assign(level=levels, utility=u)
    rows = []
    for bucket, g in out.groupby("bucket"):
        rows.append({
            "bucket": bucket, "n": len(g),
            "species": float((g.level == SPECIES).mean()),
            "genus": float((g.level == GENUS).mean()),
            "decline": float((g.level == DECLINE).mean()),
            "answered_wrong": float((((g.level == SPECIES) & ~g.species_ok)
                                     | ((g.level == GENUS) & ~g.genus_ok)).mean()),
            "utility": float(g.utility.mean()),
        })
    return pd.DataFrame(rows).set_index("bucket"), out


def baseline_levels(df, p_other_threshold=None, coverage=0.95):
    """The rule being replaced: one threshold on P(__OTHER__); accept -> answer
    at species level, else decline. No genus level at all.

    With `p_other_threshold=None` the threshold is refitted to accept
    `coverage` of in-catalogue observations, which is how the current 0.032 was
    derived — refitting keeps the comparison fair on whatever split it is given.
    """
    p_other = 1.0 - df["one_minus_other"].values
    if p_other_threshold is None:
        in_cat = df["in_catalog"].values
        p_other_threshold = float(np.quantile(p_other[in_cat], coverage)) if in_cat.any() else 1.0
    return np.where(p_other <= p_other_threshold, SPECIES, DECLINE), p_other_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(DATA_PROCESSED / "inat_bioclip2.npz"))
    ap.add_argument("--min-photos", type=int, default=2)
    args = ap.parse_args()

    df, router_acc = build_observations(args.emb, min_photos=args.min_photos)
    calib, test = df[df.fold == "calib"], df[df.fold == "test"]
    print(f"organ router accuracy {router_acc:.3f}")
    print(f"observations: {len(df)}  (calib {len(calib)} / test {len(test)})")
    print(df.groupby(["bucket", "fold"]).size().unstack(fill_value=0).to_string(), "\n")

    (tg, ts), u_cal = fit_thresholds(
        calib["species_conf"].values, calib["genus_conf"].values,
        calib["species_ok"].values, calib["genus_ok"].values, calib["in_catalog"].values)
    print(f"fitted on calibration: t_genus={tg:.4f}  t_species={ts:.4f}  (calib utility {u_cal:+.3f})\n")

    table, scored = summarise(test, tg, ts)
    print("TEST SPLIT — outcome rates per bucket")
    print(table.round(3).to_string(), "\n")

    clusters = np.array([r.species if r.bucket != "near_ood" else r.genus for r in test.itertuples()])
    lo, hi = cluster_bootstrap(scored["utility"].values, clusters)
    print(f"mean utility on test: {scored.utility.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # the rule being replaced, threshold refitted on calibration for fairness
    _, thr = baseline_levels(calib)
    base_levels, _ = baseline_levels(test, p_other_threshold=thr)
    base_u = utility(base_levels, test["species_ok"].values, test["genus_ok"].values,
                     test["in_catalog"].values)
    blo, bhi = cluster_bootstrap(base_u, clusters)
    print(f"baseline (single threshold P(OTHER)<={thr:.4f}): {base_u.mean():+.3f}  95% CI [{blo:+.3f}, {bhi:+.3f}]")
    delta = scored["utility"].values - base_u
    dlo, dhi = cluster_bootstrap(delta, clusters)
    star = "" if (dlo <= 0 <= dhi) else "  *"
    print(f"paired gain: {delta.mean():+.3f}  95% CI [{dlo:+.3f}, {dhi:+.3f}]{star}")
    levels = decide(test["species_conf"].values, test["genus_conf"].values, tg, ts)
    for label, mix in (("global OOD (mosses, ferns, tropical)", OOD_MIX_GLOBAL),
                       ("regional OOD (Europe/N.America)", OOD_MIX_REGIONAL)):
        if not set(mix) & set(test["bucket"].unique()):
            continue
        print(f"\nPRECISION / COVERAGE vs assumed out-of-catalogue rate — {label}")
        print(prevalence_table(levels, test["species_ok"].values, test["genus_ok"].values,
                               test["bucket"].values, ood_mix=mix).round(3).to_string())

    print("\nBASELINE outcome rates per bucket")
    bt = pd.DataFrame([{ "bucket": b, "n": len(g),
        "species": float((base_levels[test.index.get_indexer(g.index)] == SPECIES).mean()),
        "decline": float((base_levels[test.index.get_indexer(g.index)] == DECLINE).mean()),
    } for b, g in test.groupby("bucket")]).set_index("bucket")
    print(bt.round(3).to_string())


if __name__ == "__main__":
    main()
