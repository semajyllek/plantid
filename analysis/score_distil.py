"""Score the distilled student: does it recover BioCLIP-2's advantage?

The chain it has to beat:
  BioCLIP v1 (what ships today)  genus 0.931  species 0.760  vs Pl@ntNet -8.4pp
  BioCLIP-2  (cannot ship)       genus 0.975  species 0.846  vs Pl@ntNet +4.5pp

Distillation is only worth anything if the student lands materially above v1.
"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from plantid.config import DATA_PROCESSED as D
from plantid.eval.combiners import trimmed
from plantid.eval.headtohead import alias_sets, boot, correct
from plantid.eval.rejection import (OOD_MIX_REGIONAL, build_observations, cluster_bootstrap,
                                    decide, deployment_weights, fit_thresholds,
                                    precision_coverage)

VARIANTS = [("bioclip2", "BioCLIP-2 (cannot ship)"),
            ("bioclip1_distil", "distilled student (ships)"),
            ("bioclip1", "BioCLIP v1 (ships today)"),
            ("bioclip1_cml4", "BioCLIP v1 int4 (ships today)")]
A = alias_sets()
frames, rows = {}, []
for v, label in VARIANTS:
    df, _ = build_observations(str(D / f"inat_{v}.npz"), variant=v)
    frames[v] = df
    inc = df[df.in_catalog]
    glo, ghi = cluster_bootstrap(inc.genus_ok.values.astype(float), inc.species.values)
    slo, shi = cluster_bootstrap(inc.species_ok.values.astype(float), inc.species.values)
    calib, test = df[df.fold == "calib"], df[df.fold == "test"]
    sw = deployment_weights(calib.bucket.values, p_ood=0.2, ood_mix=OOD_MIX_REGIONAL)
    (tg, ts), _ = fit_thresholds(calib.species_conf.values, calib.genus_conf.values,
                                 calib.species_ok.values, calib.genus_ok.values,
                                 calib.in_catalog.values, sample_weight=sw)
    lv = decide(test.species_conf.values, test.genus_conf.values, tg, ts)
    prec, cov = precision_coverage(lv, test.species_ok.values, test.genus_ok.values,
                                   test.bucket.values, p_ood=0.2, ood_mix=OOD_MIX_REGIONAL)
    s = df[df.in_catalog | (df.bucket == "regional_ood")]
    rows.append({"encoder": label, "genus": inc.genus_ok.mean(), "g_lo": glo, "g_hi": ghi,
                 "species": inc.species_ok.mean(), "s_lo": slo, "s_hi": shi,
                 "AUROC_reg": roc_auc_score(s.in_catalog.values.astype(int), s.genus_conf.values),
                 "prec@20": prec, "cov@20": cov})
    print(f"  {label}: genus {inc.genus_ok.mean():.4f} species {inc.species_ok.mean():.4f}", flush=True)

print("\n=== FULL EVALUATION, 5,534 real observations ===")
print(pd.DataFrame(rows).set_index("encoder").round(4).to_string())

print("\n=== PAIRED vs BioCLIP v1 (what ships today) ===")
base = frames["bioclip1"]; m = base.in_catalog.values; sp = base.species.values[m]
for v, label in VARIANTS:
    if v == "bioclip1":
        continue
    for col in ("genus_ok", "species_ok"):
        d = frames[v][col].values[m].astype(float) - base[col].values[m].astype(float)
        lo, hi = cluster_bootstrap(d, sp)
        star = "" if lo <= 0 <= hi else "  *"
        print(f"  {label:30s} {col:11s} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")

print("\n=== HEAD-TO-HEAD vs Pl@ntNet, same 465 observations, single photo ===")
h = pd.read_parquet(D / "headtohead.parquet"); h = h[h.plantnet_top1.notna()]
pn = np.array([correct(r.plantnet_top1, r.truth, A) for r in h.itertuples()], float)
lo, hi = boot(pn)
print(f"  {'Pl@ntNet (~50k species)':34s} {pn.mean():.4f} [{lo:.4f}, {hi:.4f}]")
def first(P, organs, oi): return P[0]
for v, label in VARIANTS:
    df, _ = build_observations(str(D / f"inat_{v}.npz"), variant=v, combiner=first)
    mm = df.set_index("obs_id").reindex(h.obs_id)
    ok = np.array([correct(p, t, A) for p, t in zip(mm.pred_species, h.truth)], float)
    d = ok - pn; dlo, dhi = boot(d); olo, ohi = boot(ok)
    star = "" if dlo <= 0 <= dhi else "  *"
    print(f"  {label:34s} {ok.mean():.4f} [{olo:.4f}, {ohi:.4f}]   "
          f"vs Pl@ntNet {d.mean():+.4f} [{dlo:+.4f}, {dhi:+.4f}]{star}")
