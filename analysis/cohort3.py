"""Three cohorts, separating rarity from nomenclature.

  broad      250 species that surfaced under untargeted queries
  targeted   161 that appeared only when asked for by name
  recovered   54 that needed a taxon_id because the catalogue's name is stale

The distinction matters. `targeted` is a rarity cohort and should be harder.
`recovered` is not: *Anemone nemorosa* has 84,623 research-grade observations
and was invisible only because the catalogue calls it by a superseded name. If
`recovered` scores like `broad`, the difficulty gradient really is about how
often a plant is photographed, and the naming gap was costing us easy species.
"""

import sys

from pathlib import Path as _P
OUT = _P(__file__).parent / 'out'
OUT.mkdir(exist_ok=True)

import numpy as np
import pandas as pd


from plantid.eval.rejection import (  # noqa: E402
    DECLINE, GENUS, SPECIES, OOD_MIX_REGIONAL, build_observations, cluster_bootstrap,
    decide, deployment_weights, fit_thresholds, precision_coverage, summarise,
)

SC = str(OUT)
ASSUMED_OOD = 0.2
COHORTS = ("broad", "targeted", "recovered")


def gap(a, b, col, n=4000, seed=0):
    """Two-sample cluster bootstrap on a - b, resampling species within each."""
    rng = np.random.RandomState(seed)
    parts = []
    for g in (a, b):
        u = np.array(sorted(g.species.unique()))
        parts.append((g[col].values.astype(float), u,
                      {c: np.flatnonzero(g.species.values == c) for c in u}))
    out = []
    for _ in range(n):
        means = []
        for vals, u, index in parts:
            pick = rng.choice(u, len(u), replace=True)
            means.append(vals[np.concatenate([index[c] for c in pick])].mean())
        out.append(means[0] - means[1])
    out = np.array(out)
    return out.mean(), np.percentile(out, 2.5), np.percentile(out, 97.5)


def main():
    members = {c: {ln.strip() for ln in open(f"{SC}/cohort_{c}.txt") if ln.strip()}
               for c in COHORTS}
    df, router_acc = build_observations("data/processed/inat_bioclip2.npz")
    coh = pd.Series("ood", index=df.index, dtype=object)
    for c in COHORTS:
        coh[df.in_catalog.values & df.species.isin(members[c]).values] = c
    df["cohort"] = coh
    print(f"router {router_acc:.3f}   observations {len(df)}")
    print(df.groupby("bucket").size().to_string())

    inc = df[df.in_catalog]
    print("\n=== IN-CATALOGUE ACCURACY BY COHORT (threshold-free) ===")
    rows = []
    for c in COHORTS + ("combined",):
        g = inc if c == "combined" else inc[inc.cohort == c]
        lo, hi = cluster_bootstrap(g.species_ok.values.astype(float), g.species.values)
        glo, ghi = cluster_bootstrap(g.genus_ok.values.astype(float), g.species.values)
        rows.append({"cohort": c, "obs": len(g), "species": g.species.nunique(),
                     "species_acc": g.species_ok.mean(), "sp_ci": f"[{lo:.3f}, {hi:.3f}]",
                     "genus_acc": g.genus_ok.mean(), "gn_ci": f"[{glo:.3f}, {ghi:.3f}]"})
    print(pd.DataFrame(rows).set_index("cohort").round(3).to_string())

    print("\n=== PAIRWISE GAPS (two-sample cluster bootstrap) ===")
    for a, b in (("broad", "targeted"), ("broad", "recovered"), ("recovered", "targeted")):
        for col in ("species_ok", "genus_ok"):
            m, lo, hi = gap(inc[inc.cohort == a], inc[inc.cohort == b], col)
            star = "" if lo <= 0 <= hi else "  *"
            print(f"  {a:9s} - {b:9s}  {col:11s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]{star}")

    calib, test = df[df.fold == "calib"], df[df.fold == "test"]
    sw = deployment_weights(calib["bucket"].values, p_ood=ASSUMED_OOD, ood_mix=OOD_MIX_REGIONAL)
    (tg, ts), _ = fit_thresholds(calib.species_conf.values, calib.genus_conf.values,
                                 calib.species_ok.values, calib.genus_ok.values,
                                 calib.in_catalog.values, sample_weight=sw)
    print(f"\nfitted: t_genus={tg:.4f}  t_species={ts:.4f}")
    table, scored = summarise(test, tg, ts)
    print(table.round(3).to_string())
    lv = decide(test.species_conf.values, test.genus_conf.values, tg, ts)
    prec, cov = precision_coverage(lv, test.species_ok.values, test.genus_ok.values,
                                   test.bucket.values, p_ood=ASSUMED_OOD,
                                   ood_mix=OOD_MIX_REGIONAL)
    clusters = np.array([r.species if r.bucket != "near_ood" else r.genus
                         for r in test.itertuples()])
    lo, hi = cluster_bootstrap(scored["utility"].values, clusters)
    print(f"utility {scored.utility.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]   "
          f"@20% OOD regional: precision {prec:.3f} coverage {cov:.3f}")

    print("\n=== TEST-SPLIT DECISIONS, IN-CATALOGUE, BY COHORT ===")
    s = test.assign(level=lv)
    rows = []
    for c in COHORTS:
        g = s[s.cohort == c]
        rows.append({"cohort": c, "n": len(g), "species": (g.level == SPECIES).mean(),
                     "genus": (g.level == GENUS).mean(), "decline": (g.level == DECLINE).mean()})
    print(pd.DataFrame(rows).set_index("cohort").round(3).to_string())

    # near-OOD is the bucket the fetch grew; its CI is set by genera, not rows
    no = test[test.bucket == "near_ood"]
    u = scored.loc[no.index, "utility"].values
    nlo, nhi = cluster_bootstrap(u, no.genus.values)
    print(f"\nnear-OOD test: n={len(no)} over {no.genus.nunique()} genera   "
          f"utility {u.mean():+.3f} [{nlo:+.3f}, {nhi:+.3f}]")
    df.to_parquet(f"{SC}/observations_cohort3.parquet", index=False)


if __name__ == "__main__":
    main()
