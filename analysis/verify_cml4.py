"""Verify the Core ML caches, then score both arms against the declared criterion.

Pre-registered before the run: PASS = in-catalogue genus accuracy on real
observations, matched configuration, with the *lower* bound of the 95% cluster
bootstrap above 0.90.
"""

import sys

from pathlib import Path as _P
OUT = _P(__file__).parent / 'out'
OUT.mkdir(exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


from plantid.config import DATA_PROCESSED as D  # noqa: E402
from plantid.eval.rejection import (  # noqa: E402
    OOD_MIX_REGIONAL, build_observations, cluster_bootstrap, decide,
    deployment_weights, fit_thresholds, precision_coverage,
)
from plantid.features.embed_background import cache_path as bg_path  # noqa: E402
from plantid.features.embed_catalog import cache_path as cat_path  # noqa: E402

V = "bioclip1_cml4"
REF = "bioclip1"
ORGANS = ("leaf", "flower", "bark")

print("=" * 72)
print("CACHE INTEGRITY")
print("=" * 72)
cat = pd.read_parquet(D / "catalog_index.parquet")
cat = cat[cat.local_path.notna()]
bg = pd.read_parquet(D / "plantnet_background.parquet")
bg = bg[bg.local_path.notna()]
ok = True
for organ in ORGANS:
    for tag, path_fn, manifest in (("cat", cat_path, cat), ("bg", bg_path, bg)):
        z = np.load(path_fn(organ, V, D))
        n_exp = int((manifest.organ == organ).sum())
        n_got, dim = z["descriptor"].shape
        nan = int(np.isnan(z["descriptor"]).sum())
        good = n_got == n_exp and dim == 512 and nan == 0
        ok &= good
        print(f"  {tag}[{organ}]: {n_got:6d}/{n_exp:6d} rows, dim {dim}, "
              f"{nan} NaN  {'OK' if good else 'FAIL'}")

obs = pd.read_parquet(D / "inat_observations.parquet")
paths = {p for ps in obs.local_paths for p in ps}
zi = np.load(D / f"inat_{V}.npz")
covered = paths <= set(zi["path"].tolist())
ok &= covered and not np.isnan(zi["descriptor"]).any()
print(f"  inat: {zi['descriptor'].shape}, covers every manifest photo: {covered}")

print("\n" + "=" * 72)
print("CROSS-CHECK vs THE 64-IMAGE VALIDATION (expected ~0.935)")
print("=" * 72)
for organ in ORGANS:
    a, b = np.load(cat_path(organ, V, D)), np.load(cat_path(organ, REF, D))
    ida = {k: i for i, k in enumerate(a["image_id"])}
    shared = [k for k in b["image_id"] if k in ida]
    ia = np.array([ida[k] for k in shared])
    ib = np.array([i for i, k in enumerate(b["image_id"]) if k in ida])
    X, Y = a["descriptor"][ia], b["descriptor"][ib]
    cos = (X * Y).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1))
    print(f"  {organ}: n={len(shared):6d}  cosine mean {cos.mean():.4f}  "
          f"p05 {np.percentile(cos, 5):.4f}")

print("\n" + "=" * 72)
print("BOTH ARMS")
print("=" * 72)
emb_cml = str(D / f"inat_{V}.npz")
arms = [("A · matched (deployment): int4 head, int4 eval", V, emb_cml),
        ("B · mismatched control: fp32 head, int4 eval", REF, emb_cml),
        ("baseline: fp32 head, fp32 eval", REF, str(D / f"inat_{REF}.npz"))]

rows = []
for label, variant, emb in arms:
    df, _ = build_observations(emb, variant=variant)
    inc = df[df.in_catalog]
    lo, hi = cluster_bootstrap(inc.genus_ok.values.astype(float), inc.species.values)
    slo, shi = cluster_bootstrap(inc.species_ok.values.astype(float), inc.species.values)
    calib, test = df[df.fold == "calib"], df[df.fold == "test"]
    sw = deployment_weights(calib.bucket.values, p_ood=0.2, ood_mix=OOD_MIX_REGIONAL)
    (tg, ts), _ = fit_thresholds(calib.species_conf.values, calib.genus_conf.values,
                                 calib.species_ok.values, calib.genus_ok.values,
                                 calib.in_catalog.values, sample_weight=sw)
    lv = decide(test.species_conf.values, test.genus_conf.values, tg, ts)
    prec, cov = precision_coverage(lv, test.species_ok.values, test.genus_ok.values,
                                   test.bucket.values, p_ood=0.2, ood_mix=OOD_MIX_REGIONAL)

    def au(bucket):
        s = df[df.in_catalog | (df.bucket == bucket)]
        return roc_auc_score(s.in_catalog.values.astype(int), s.genus_conf.values)

    rows.append({"arm": label, "genus": inc.genus_ok.mean(), "g_lo": lo, "g_hi": hi,
                 "species": inc.species_ok.mean(), "s_lo": slo, "s_hi": shi,
                 "auroc_reg": au("regional_ood"), "auroc_near": au("near_ood"),
                 "prec@20": prec, "cov@20": cov})
    print(f"\n{label}")
    print(f"  genus   {inc.genus_ok.mean():.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"  species {inc.species_ok.mean():.4f}  [{slo:.4f}, {shi:.4f}]")
    print(f"  AUROC regional {au('regional_ood'):.4f}  near {au('near_ood'):.4f}")
    print(f"  precision@20% {prec:.4f}   coverage {cov:.4f}", flush=True)

t = pd.DataFrame(rows).set_index("arm")
print("\n" + "=" * 72)
print(t.round(4).to_string())
a = rows[0]
print("\n" + "=" * 72)
verdict = "PASS" if a["g_lo"] > 0.90 else "FAIL"
print(f"PRE-REGISTERED CRITERION (arm A genus CI lower bound > 0.90): {verdict}")
print(f"  genus {a['genus']:.4f}, 95% CI [{a['g_lo']:.4f}, {a['g_hi']:.4f}]")
print(f"cache integrity: {'OK' if ok else 'FAIL'}")
t.to_csv(str(OUT) + "/cml4_arms.csv")
