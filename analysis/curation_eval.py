"""Does label curation help, and which half of it?

Three label functions, same data, same buckets:

  baseline   two-token truncation, as shipped
  canonical  hybrid-aware naming -- a bug fix, no product judgement
  curated    canonical + MERGE + DROP -- the product decision

Merging trivially raises accuracy on merged species, so the number that matters
is not the headline. It is what happens to the species left *behind* in a
de-crowded genus: if collapsing seven Ophrys microspecies helps the ten real
Ophrys species that remain, curation bought something real. Those species are
the treatment group; species in genera curation never touched are the control.
"""

import sys

from pathlib import Path as _P
OUT = _P(__file__).parent / 'out'
OUT.mkdir(exist_ok=True)

import numpy as np
import pandas as pd


from plantid.data.curation import MERGE, canonical_name, curated_name  # noqa: E402
from plantid.eval.rejection import (  # noqa: E402
    OOD_MIX_REGIONAL, build_observations, cluster_bootstrap, decide,
    deployment_weights, fit_thresholds, precision_coverage,
)

SC = str(OUT)
EMB = "data/processed/inat_bioclip2.npz"
ASSUMED_OOD = 0.2

CONFIGS = {
    "baseline": lambda n: " ".join(str(n).split()[:2]),
    "canonical": lambda n: curated_name(n, merge=False),
    "curated": curated_name,
}

# genera curation touched: their surviving species are the crowding-relief test
TOUCHED_GENERA = {m.split()[0] for m in MERGE} | {t.split()[0] for t in MERGE.values()}
MERGED_AWAY = set(MERGE)


def main():
    cat = pd.read_parquet("data/processed/catalog_index.parquet")["species_name"].unique()
    frames = {}
    rows = []
    for name, fn in CONFIGS.items():
        classes = {c for c in map(fn, cat) if c}
        df, _ = build_observations(EMB, name_fn=fn)
        frames[name] = df
        calib, test = df[df.fold == "calib"], df[df.fold == "test"]
        sw = deployment_weights(calib.bucket.values, p_ood=ASSUMED_OOD, ood_mix=OOD_MIX_REGIONAL)
        (tg, ts), _ = fit_thresholds(calib.species_conf.values, calib.genus_conf.values,
                                     calib.species_ok.values, calib.genus_ok.values,
                                     calib.in_catalog.values, sample_weight=sw)
        lv = decide(test.species_conf.values, test.genus_conf.values, tg, ts)
        prec, cov = precision_coverage(lv, test.species_ok.values, test.genus_ok.values,
                                       test.bucket.values, p_ood=ASSUMED_OOD,
                                       ood_mix=OOD_MIX_REGIONAL)
        inc = df[df.in_catalog]
        rows.append({"config": name, "classes": len(classes),
                     "species_acc": inc.species_ok.mean(), "genus_acc": inc.genus_ok.mean(),
                     "t_genus": tg, "t_species": ts, "precision@20": prec, "coverage": cov})
    print("=== HEADLINE, ALL THREE CONFIGS ===")
    print(pd.DataFrame(rows).set_index("config").round(4).to_string())

    # --- the honest test: species curation did NOT merge -------------------
    print("\n=== CROWDING RELIEF: species curation never merged ===")
    print(f"touched genera: {sorted(TOUCHED_GENERA)}")
    base = frames["baseline"]
    # partition on the BASELINE naming so the groups are identical across configs
    survivor = base.in_catalog.values & base.species.map(
        lambda s: s.split()[0] in TOUCHED_GENERA and s not in MERGED_AWAY).values
    control = base.in_catalog.values & base.species.map(
        lambda s: s.split()[0] not in TOUCHED_GENERA).values
    print(f"treatment (survivors in touched genera): {survivor.sum()} obs, "
          f"{base.species[survivor].nunique()} species")
    print(f"control   (untouched genera):            {control.sum()} obs, "
          f"{base.species[control].nunique()} species")

    out = []
    for name, df in frames.items():
        assert (df.index == base.index).all(), "row alignment broke"
        out.append({"config": name,
                    "treatment_species_acc": df.species_ok.values[survivor].mean(),
                    "treatment_genus_acc": df.genus_ok.values[survivor].mean(),
                    "control_species_acc": df.species_ok.values[control].mean(),
                    "control_genus_acc": df.genus_ok.values[control].mean()})
    print(pd.DataFrame(out).set_index("config").round(4).to_string())

    print("\npaired change vs baseline, bootstrapped over species clusters:")
    for name in ("canonical", "curated"):
        for label, mask in (("treatment", survivor), ("control", control)):
            d = (frames[name].species_ok.values[mask].astype(float)
                 - base.species_ok.values[mask].astype(float))
            lo, hi = cluster_bootstrap(d, base.species.values[mask])
            star = "" if lo <= 0 <= hi else "  *"
            print(f"  {name:10s} {label:10s} species_ok {d.mean():+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]{star}")

    frames["curated"].to_parquet(f"{SC}/observations_curated.parquet", index=False)


if __name__ == "__main__":
    main()
