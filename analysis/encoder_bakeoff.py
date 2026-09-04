"""Compare candidate encoders on both corpora, because one of them is compromised.

`bioclip_inat` is trained on iNaturalist only, and the evaluation set *is*
iNaturalist. Every encoder here carries some iNat contamination — BioCLIP-2 via
TreeOfLife-200M, BioCLIP v1 via TreeOfLife-10M — but an iNat-only model has the
tightest possible match to the evaluation distribution, so a win on iNat alone
cannot be read as a better encoder.

The PlantNet catalogue test split is the control: a different corpus, held out
from head fitting, and not what `bioclip_inat` was trained on. An encoder that
wins on *both* is genuinely better; one that wins only on iNat is showing
contamination.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


from plantid.config import DATA_PROCESSED as D, ORGANS  # noqa: E402
from plantid.eval.inat_fusion import OTHER, _l2, build_heads  # noqa: E402
from plantid.eval.rejection import (  # noqa: E402
    OOD_MIX_REGIONAL, build_observations, cluster_bootstrap, decide,
    deployment_weights, fit_thresholds, precision_coverage,
)
from plantid.features.embed_catalog import load_catalog  # noqa: E402

CANDIDATES = ["bioclip2", "bioclip1", "bioclip_inat"]


def plantnet_control(variant):
    """Species / genus accuracy on the catalogue's own held-out test split.

    Same heads the iNat evaluation uses, scored on PlantNet images the head never
    saw. Weighted across organs by test-set size, matching how
    ONDEVICE_FINDINGS reports it.
    """
    heads, proj, classes = build_heads(variant=variant)
    mask = classes != OTHER
    genus_of = {c: c.split()[0] for c in classes[mask]}
    n_tot = sp_ok = gn_ok = 0
    for organ in ORGANS:
        d = load_catalog(organ, variant=variant)
        te = d["split"] == "test"
        if not te.any():
            continue
        E = _l2(d["descriptor"])[te]
        from plantid.data.curation import curated_name
        truth = np.array([curated_name(n) or "" for n in d["species_name"]])[te]
        keep = truth != ""
        E, truth = E[keep], truth[keep]
        pred = heads[organ].predict(E)
        sp_ok += int((pred == truth).sum())
        gn_ok += int(sum(genus_of.get(p, p).split()[0] == t.split()[0]
                         for p, t in zip(pred, truth)))
        n_tot += len(truth)
    return {"n": n_tot, "species": sp_ok / n_tot, "genus": gn_ok / n_tot}


def main():
    rows, frames = [], {}
    for v in CANDIDATES:
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

        def au(bucket, df=df):
            s = df[df.in_catalog | (df.bucket == bucket)]
            return roc_auc_score(s.in_catalog.values.astype(int), s.genus_conf.values)

        ctl = plantnet_control(v)
        rows.append({"encoder": v,
                     "iNat genus": inc.genus_ok.mean(), "g_lo": glo, "g_hi": ghi,
                     "iNat species": inc.species_ok.mean(), "s_lo": slo, "s_hi": shi,
                     "AUROC reg": au("regional_ood"), "AUROC near": au("near_ood"),
                     "prec@20": prec, "cov@20": cov,
                     "PlantNet genus": ctl["genus"], "PlantNet species": ctl["species"]})
        print(f"{v}: iNat genus {inc.genus_ok.mean():.4f} [{glo:.4f}, {ghi:.4f}] | "
              f"PlantNet genus {ctl['genus']:.4f} (n={ctl['n']})", flush=True)

    t = pd.DataFrame(rows).set_index("encoder")
    print("\n=== ON iNATURALIST (contaminated for all three, most for bioclip_inat) ===")
    print(t[["iNat genus", "g_lo", "g_hi", "iNat species", "AUROC reg", "AUROC near",
             "prec@20", "cov@20"]].round(4).to_string())
    print("\n=== ON PLANTNET TEST SPLIT (the control) ===")
    print(t[["PlantNet genus", "PlantNet species"]].round(4).to_string())

    print("\n=== PAIRED vs bioclip1, on iNat, over species clusters ===")
    base = frames["bioclip1"]
    m = base.in_catalog.values
    sp = base.species.values[m]
    for v in ("bioclip_inat", "bioclip2"):
        for col in ("genus_ok", "species_ok"):
            d = frames[v][col].values[m].astype(float) - base[col].values[m].astype(float)
            lo, hi = cluster_bootstrap(d, sp)
            star = "" if lo <= 0 <= hi else "  *"
            print(f"  {v:13s} − bioclip1  {col:11s} {d.mean():+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]{star}")

    t.to_csv("/private/tmp/claude-501/-Users-jameskelly-Documents-plantid/"
             "f3b6d3aa-5322-4d68-a854-87044e191fb8/scratchpad/encoder_bakeoff.csv")


if __name__ == "__main__":
    main()
