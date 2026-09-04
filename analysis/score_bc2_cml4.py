"""What int4 costs BioCLIP-2, measured through the shipped Core ML artifact.

Cosine says 0.982 and therefore almost nothing. Cosine has been wrong here twice
in opposite directions -- it under-predicted the cost of quantization (0.932
cost 1.3pp) and wildly over-predicted the value of distillation (0.956 bought
nothing). So this measures the head.
"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from plantid.config import DATA_PROCESSED as D
from plantid.eval.headtohead import alias_sets, boot, correct
from plantid.eval.rejection import (OOD_MIX_REGIONAL, build_observations, cluster_bootstrap,
                                    decide, deployment_weights, fit_thresholds,
                                    precision_coverage)

VARIANTS = [("bioclip2", "BioCLIP-2 fp32 (cannot ship)"),
            ("bioclip2_cml4", "BioCLIP-2 Core ML int4 (160 MB)"),
            ("bioclip1_cml4", "BioCLIP v1 Core ML int4 (46 MB)")]
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

print("\n=== WHAT int4 COSTS BioCLIP-2 (paired, fp32 - int4) ===")
base = frames["bioclip2"]; m = base.in_catalog.values; sp = base.species.values[m]
for col in ("genus_ok", "species_ok"):
    d = base[col].values[m].astype(float) - frames["bioclip2_cml4"][col].values[m].astype(float)
    lo, hi = cluster_bootstrap(d, sp)
    star = "" if lo <= 0 <= hi else "  *"
    print(f"  {col:11s} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")

print("\n=== HEAD-TO-HEAD, same 465, single photo, vs the offline field ===")
h = pd.read_parquet(D / "headtohead.parquet"); h = h[h.plantnet_top1.notna() & h.inat_top1.notna()]
def first(P, o, oi): return P[0]
inat = np.array([correct(p, t, A) for p, t in zip(h.inat_top1, h.truth)], float)
lo, hi = boot(inat)
print(f"  {'iNaturalist server (needs network)':40s} {inat.mean():.4f} [{lo:.4f}, {hi:.4f}]")
for v, label in VARIANTS:
    df, _ = build_observations(str(D / f"inat_{v}.npz"), variant=v, combiner=first)
    mm = df.set_index("obs_id").reindex(h.obs_id)
    ok = np.array([correct(p, t, A) for p, t in zip(mm.pred_species, h.truth)], float)
    olo, ohi = boot(ok); d = ok - inat; dlo, dhi = boot(d)
    star = "" if dlo <= 0 <= dhi else "  *"
    print(f"  {label:40s} {ok.mean():.4f} [{olo:.4f}, {ohi:.4f}]  vs iNat {d.mean():+.4f} [{dlo:+.4f}, {dhi:+.4f}]{star}")
