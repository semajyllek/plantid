"""Two controls for the top-up run.

A. Pre-top-up replay. The table in REJECTION_FINDINGS.md is mu=2 era, so the
   current thresholds cannot be compared against it -- most of the movement
   happened when mu was raised to 4 and thresholds were anchored to a stated
   prevalence. Replaying the *previous* manifest through *today's* code isolates
   what the new observations actually changed.

B. Synonym relabel. The catalogue carries pre-split names (`Anemone nemorosa`,
   `Sedum kamtschaticum`), so an OOD observation filed under iNat's current name
   (`Anemonoides nemorosa`, `Phedimus kamtschaticus`) escapes the binomial match
   that assigns buckets. 6 rows are catalogue species scored as rejects and 39
   sit in a catalogue genus while labelled distant/regional. Both push the
   reported numbers pessimistic; this measures by how much.
"""

import json
import shutil
import sys
from pathlib import Path

from pathlib import Path as _P
OUT = _P(__file__).parent / 'out'
OUT.mkdir(exist_ok=True)

import numpy as np
import pandas as pd


from plantid.eval.rejection import (  # noqa: E402
    DECLINE, GENUS, SPECIES, OOD_MIX_REGIONAL, build_observations, cluster_bootstrap,
    decide, deployment_weights, fit_thresholds, precision_coverage, summarise, utility,
)

SC = Pathstr(OUT)
EMB = "data/processed/inat_bioclip2.npz"
ASSUMED_OOD = 0.2


def run(cache_dir, label):
    df, router_acc = build_observations(EMB, cache_dir=cache_dir)
    calib, test = df[df.fold == "calib"], df[df.fold == "test"]
    sw = deployment_weights(calib["bucket"].values, p_ood=ASSUMED_OOD, ood_mix=OOD_MIX_REGIONAL)
    (tg, ts), _ = fit_thresholds(calib.species_conf.values, calib.genus_conf.values,
                                 calib.species_ok.values, calib.genus_ok.values,
                                 calib.in_catalog.values, sample_weight=sw)
    table, scored = summarise(test, tg, ts)
    lv = decide(test.species_conf.values, test.genus_conf.values, tg, ts)
    prec, cov = precision_coverage(lv, test.species_ok.values, test.genus_ok.values,
                                   test.bucket.values, p_ood=ASSUMED_OOD,
                                   ood_mix=OOD_MIX_REGIONAL)
    clusters = np.array([r.species if r.bucket != "near_ood" else r.genus
                         for r in test.itertuples()])
    lo, hi = cluster_bootstrap(scored["utility"].values, clusters)
    inc = df[df.in_catalog]
    print(f"\n########## {label} ##########")
    print(f"observations {len(df)} (calib {len(calib)} / test {len(test)}), "
          f"in-catalogue species {inc.species.nunique()}, router {router_acc:.3f}")
    print(f"t_genus={tg:.4f}  t_species={ts:.4f}")
    print(f"in-catalogue species acc {inc.species_ok.mean():.3f}  genus acc {inc.genus_ok.mean():.3f}")
    print(f"utility {scored.utility.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]   "
          f"@20% OOD regional: precision {prec:.3f} coverage {cov:.3f}")
    print(table.round(3).to_string())
    return dict(label=label, tg=tg, ts=ts, prec=prec, cov=cov,
                util=scored.utility.mean(), n=len(df))


def build_relabelled(src_cache: Path, dst: Path):
    """Copy the live cache, then rebucket rows whose name is a catalogue synonym."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src_cache.glob("*.npz"):
        (dst / f.name).unlink(missing_ok=True)
        (dst / f.name).symlink_to(f.resolve())
    for f in ("catalog_index.parquet", "plantnet_background.parquet"):
        (dst / f).unlink(missing_ok=True)
        (dst / f).symlink_to((src_cache / f).resolve())

    res = pd.DataFrame(json.load(open(SC / "resolved_missing.json")))
    res = res[res.resolved.notna()]
    cat = pd.read_parquet(src_cache / "catalog_index.parquet")
    binom = lambda s: " ".join(str(s).split()[:2])  # noqa: E731
    catsp = {binom(n) for n in cat.species_name.unique()}
    catgen = {s.split()[0] for s in catsp}
    # the catalogue's true reach once its own names are modernised
    catsp |= set(res.resolved)
    catgen |= {n.split()[0] for n in res.resolved}

    obs = pd.read_parquet(src_cache / "inat_observations.parquet").copy()
    b = obs.species_name.map(binom)
    in_sp, in_gen = b.isin(catsp), b.map(lambda n: n.split()[0] in catgen)
    was = obs.bucket.copy()
    new = obs.bucket.copy()
    new[~in_sp & in_gen & was.isin(["distant_ood", "regional_ood"])] = "near_ood"
    new[in_sp & (was != "in_catalog")] = "in_catalog"
    obs["bucket"] = new
    print("synonym relabel, rows moved:")
    print(pd.crosstab(was, new).to_string())
    obs.to_parquet(dst / "inat_observations.parquet", index=False)


if __name__ == "__main__":
    live = Path("/Users/jameskelly/Documents/plantid/data/processed")
    out = [run(live, "CURRENT — 497-species catalogue, post top-up")]
    out.append(run(SC / "cache_pre247", "CONTROL A — pre-top-up manifest, today's code"))
    build_relabelled(live, SC / "cache_syn")
    out.append(run(SC / "cache_syn", "CONTROL B — synonym-corrected buckets"))
    print("\n########## SUMMARY ##########")
    print(pd.DataFrame(out).set_index("label").round(4).to_string())
