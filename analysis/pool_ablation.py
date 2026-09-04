"""Does the size of the `__OTHER__` pool bind?

The reject class trains on 357 PlantNet species -- and that is not a build
mistake to be repaired: rebuilding against the current catalogue returns the
identical set. PlantNet holds 1,081 species, the catalogue now claims 530 of
them, and few of the rest have enough images. Growing the catalogue necessarily
shrinks the pool, so the only way to a bigger reject class is a different
corpus.

Before sourcing one, find out whether it would buy anything. Subsample the pool
by *species* and watch rejection. If the curve is flat at 357 the pool is not
the constraint and the effort belongs elsewhere; if it is still climbing, the
reject class is starved and iNaturalist background is worth fetching.
"""

import sys

from pathlib import Path as _P
OUT = _P(__file__).parent / 'out'
OUT.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


import plantid.eval.inat_fusion as fusion  # noqa: E402
from plantid.eval.rejection import (  # noqa: E402
    OOD_MIX_REGIONAL, build_observations, decide, deployment_weights,
    fit_thresholds, precision_coverage,
)
from plantid.features.embed_background import binomial  # noqa: E402

SC = str(OUT)
EMB = "data/processed/inat_bioclip2.npz"
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1)
_real_load_background = fusion.load_background


def subsampled(frac, seed):
    """Wrap load_background to keep only `frac` of its species."""
    def wrapper(organ, **kw):
        d = _real_load_background(organ, **kw)
        names = np.array([binomial(n) for n in d["species_name"]])
        uniq = np.array(sorted(set(names)))
        if frac >= 1.0:
            return d
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(uniq, max(1, int(round(len(uniq) * frac))), replace=False))
        m = np.array([n in keep for n in names])
        return {k: v[m] for k, v in d.items()}
    return wrapper


def auroc(df, bucket):
    """Genus confidence, in-catalogue vs one OOD bucket. The decline score."""
    sub = df[df.in_catalog | (df.bucket == bucket)]
    return roc_auc_score(sub.in_catalog.values.astype(int), sub.genus_conf.values)


def main():
    rows = []
    for frac in FRACTIONS:
        for seed in (SEEDS if frac < 1.0 else (0,)):
            fusion.load_background = subsampled(frac, seed)
            try:
                df, _ = build_observations(EMB)
            finally:
                fusion.load_background = _real_load_background
            n_bg = len({binomial(n) for n in
                        _real_load_background("flower")["species_name"]})
            calib, test = df[df.fold == "calib"], df[df.fold == "test"]
            sw = deployment_weights(calib.bucket.values, p_ood=0.2, ood_mix=OOD_MIX_REGIONAL)
            (tg, ts), _ = fit_thresholds(calib.species_conf.values, calib.genus_conf.values,
                                         calib.species_ok.values, calib.genus_ok.values,
                                         calib.in_catalog.values, sample_weight=sw)
            lv = decide(test.species_conf.values, test.genus_conf.values, tg, ts)
            prec, cov = precision_coverage(lv, test.species_ok.values, test.genus_ok.values,
                                           test.bucket.values, p_ood=0.2,
                                           ood_mix=OOD_MIX_REGIONAL)
            inc = df[df.in_catalog]
            rows.append({
                "frac": frac, "seed": seed,
                "flower_bg_species": int(round(n_bg * frac)),
                "species_acc": inc.species_ok.mean(), "genus_acc": inc.genus_ok.mean(),
                "auroc_global": auroc(df, "distant_ood"),
                "auroc_regional": auroc(df, "regional_ood"),
                "auroc_near": auroc(df, "near_ood"),
                "precision@20": prec, "coverage": cov,
            })
            print(pd.DataFrame(rows[-1:]).round(4).to_string(index=False), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{SC}/pool_ablation.csv", index=False)
    print("\n=== MEAN OVER SEEDS ===")
    print(out.groupby("frac").mean(numeric_only=True).drop(columns="seed").round(4).to_string())


if __name__ == "__main__":
    main()
