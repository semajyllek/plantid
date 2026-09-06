"""Is the source-shift null an artifact of unequal organ mix?

`DOMAIN_SHIFT_PREREG.md` named organ mix as a confound that could *manufacture*
an effect. It can also mask one, and in the direction that flatters the
headline: the Pl@ntNet test set is a third bark by construction, bark is the
hard channel, and iNaturalist observations of a regional herbaceous catalogue
are mostly not bark. A within-source control dragged down by a photo type the
cross-source arm barely contains deflates the measured shift.

So: break the Pl@ntNet-side cells out by true organ, measure the iNaturalist
organ mix with the production router rather than assuming it, and re-run the
comparison against a Pl@ntNet control reweighted to that mix.

Reuses the fitted heads and test rows from the pre-registered primary arm; it
refits nothing.

Usage:
    PYTHONPATH=. .venv/bin/python -m analysis.domain_shift_organ --variants bioclip2
"""

import argparse

import numpy as np
import pandas as pd

from analysis.domain_shift import N_BOOT, SEED, run
from plantid.config import DATA_PROCESSED, ORGANS
from plantid.eval.inat_fusion import build_router


def per_species_organ(clf, E, df, species, organ) -> pd.Series:
    sub = df[df["organ"] == organ]
    pred = clf.predict(E[sub["emb_row"].to_numpy()])
    hit = pd.Series((pred == sub["cn"].to_numpy()).astype(float), index=sub.index)
    return hit.groupby(sub["cn"]).mean().reindex(species)


def matched(by_organ: dict[str, pd.Series], w: dict[str, float]) -> pd.Series:
    """Per species, the organ-weighted mean over the organs that species has.

    Renormalised within the organs present, so a species missing bark is not
    penalised for it -- the reweighting is about photo composition, not coverage.
    """
    M = np.column_stack([by_organ[o].to_numpy() for o in ORGANS])
    W = np.where(np.isnan(M), 0.0, np.array([w[o] for o in ORGANS])[None, :])
    num = np.nansum(np.nan_to_num(M) * W, axis=1)
    den = W.sum(axis=1)
    return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan),
                     index=by_organ[ORGANS[0]].index)


def run_organ(variant: str) -> tuple[pd.DataFrame, dict]:
    res = run(variant)
    species = res["species"]
    pn_te, Epn = res["test"]["pn"]
    in_te, Eina = res["test"]["inat"]
    h_pn, h_in = res["heads"]["pn"], res["heads"]["inat"]

    router, _ = build_router(variant=variant)
    routed = router.predict(Eina[in_te["emb_row"].to_numpy()])
    mix = pd.Series(routed).value_counts(normalize=True).reindex(ORGANS).fillna(0.0).to_dict()
    pn_mix = pn_te["organ"].value_counts(normalize=True).reindex(ORGANS).fillna(0.0).to_dict()

    pn_by = {o: per_species_organ(h_pn, Epn, pn_te, species, o) for o in ORGANS}
    in_by = {o: per_species_organ(h_in, Epn, pn_te, species, o) for o in ORGANS}

    cells = {
        "pn->pn (pooled)": res["cells"]["pn->pn"],
        "pn->pn (organ-matched)": matched(pn_by, mix),
        "pn->inat": res["cells"]["pn->inat"],
        "inat->inat": res["cells"]["inat->inat"],
        "inat->pn (pooled)": res["cells"]["inat->pn"],
        "inat->pn (organ-matched)": matched(in_by, mix),
    }
    for o in ORGANS:
        cells[f"pn->pn [{o}]"] = pn_by[o]
        cells[f"inat->pn [{o}]"] = in_by[o]

    rng = np.random.default_rng(SEED)
    n = len(species)
    M = np.column_stack([cells[k].to_numpy() for k in cells])
    draws = np.array([np.nanmean(M[rng.integers(0, n, n)], axis=0) for _ in range(N_BOOT)])
    boot = pd.DataFrame(draws, columns=list(cells))
    boot["shift vs matched"] = boot["pn->inat"] - boot["pn->pn (organ-matched)"]
    boot["reverse vs matched"] = boot["inat->pn (organ-matched)"] - boot["inat->inat"]

    rows = []
    for k in boot.columns:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        rows.append({"variant": variant, "cell": k, "value": round(boot[k].mean(), 4),
                     "lo": round(lo, 4), "hi": round(hi, 4)})
    return pd.DataFrame(rows), {"routed_inat_mix": mix, "plantnet_mix": pn_mix,
                                "n_species": len(species)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["bioclip2"])
    ap.add_argument("--out", default=str(DATA_PROCESSED / "domain_shift_organ.csv"))
    a = ap.parse_args()
    out = []
    for v in a.variants:
        t, meta = run_organ(v)
        print(f"\n== {v}  {meta}")
        print(t.to_string(index=False), flush=True)
        out.append(t)
    pd.concat(out).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
