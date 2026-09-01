"""Evaluate species ID and open-set rejection on real iNaturalist observations.

Answers the question synthetic groups cannot: when the photos are of *one
individual plant* — and therefore correlated — how much does fusing them
actually buy over a single photo?

iNat photos carry no organ label, so per-organ heads are combined by
marginalising over an organ router:

    P(class | x) = sum_o P(organ=o | x) * P(class | x, organ=o)

Each organ head is trained on the catalog's train split plus a class-weighted
`__OTHER__` class built from non-catalog species (`embed_background`). Heads
span different class sets — bark only covers the 77 species that have bark — so
each is projected into a common class space before combining.

Usage:
    PYTHONPATH=. python -m plantid.eval.inat_fusion --min-photos 3 --min-organs 2
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.data.curation import curated_name
from plantid.features.embed_background import catalog_species, eval_species_names, load_background
from plantid.features.embed_catalog import load_catalog

OTHER = "__OTHER__"


def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def build_heads(cache_dir=DATA_PROCESSED, C=10.0, exclude_eval_species=True,
                name_fn=curated_name):
    """Per-organ species heads with a class-weighted reject class, plus the
    projection from each head's classes into a shared class space.

    `exclude_eval_species` drops evaluation-set species from the `__OTHER__`
    pool. Leave it on: without it the reject class is fitted on 47 of the 183
    near-OOD species (32% of those observations), making near-OOD rejection
    partly in-sample. Set False only to reproduce the leaked measurement.
    """
    cs = catalog_species(cache_dir)
    ev = eval_species_names(cache_dir) if exclude_eval_species else None
    cat = pd.read_parquet(cache_dir / "catalog_index.parquet")
    # `name_fn` decides what counts as one nameable plant; it returns None for
    # labels curation drops, whose images leave the training set entirely.
    classes = sorted({c for c in map(name_fn, cat["species_name"].unique()) if c}) + [OTHER]
    index = {c: i for i, c in enumerate(classes)}

    heads, proj = {}, {}
    for organ in ORGANS:
        d = load_catalog(organ, cache_dir=cache_dir)
        E, tr = _l2(d["descriptor"]), d["split"] == "train"
        names = np.array([name_fn(n) or "" for n in d["species_name"]])
        tr = tr & (names != "")
        bg = _l2(
            load_background(organ, exclude_species=cs, exclude_names=ev, cache_dir=cache_dir)["descriptor"]
        )
        heads[organ] = LogisticRegression(max_iter=4000, C=C, class_weight="balanced").fit(
            np.vstack([E[tr], bg]), np.r_[names[tr], np.full(len(bg), OTHER)]
        )
        proj[organ] = np.array([index[c] for c in heads[organ].classes_])
    return heads, proj, np.array(classes)


def build_router(cache_dir=DATA_PROCESSED, C=10.0):
    """3-way leaf/bark/flower classifier, trained on the catalog's organ labels."""
    X, y, S = [], [], []
    for organ in ORGANS:
        d = load_catalog(organ, cache_dir=cache_dir)
        X.append(_l2(d["descriptor"]))
        y += [organ] * len(d["descriptor"])
        S.append(d["split"])
    X, y, S = np.vstack(X), np.array(y), np.concatenate(S)
    tr = S == "train"
    clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(X[tr], y[tr])
    acc = (clf.predict(X[S == "test"]) == y[S == "test"]).mean()
    return clf, acc


def photo_posteriors(E, heads, proj, router, n_classes, temperatures=None):
    """Organ-marginalised posterior per photo, plus routed organ and confidence.

    `temperatures` maps organ -> T, applied to that head's logits *before* the
    router mixes them. Because a mixture of tempered distributions is not a
    monotone transform of the tempered mixture, this can change the ranking and
    therefore the decision — unlike temperature on a single classifier. See
    `eval/calibration.py`.
    """
    from scipy.special import softmax

    W = router.predict_proba(E)
    order = list(router.classes_)
    out = np.zeros((len(E), n_classes))
    for organ in ORGANS:
        if temperatures and organ in temperatures:
            p = softmax(heads[organ].decision_function(E) / temperatures[organ], axis=1)
        else:
            p = heads[organ].predict_proba(E)
        out[:, proj[organ]] += W[:, order.index(organ)][:, None] * p
    return out, np.array([order[i] for i in W.argmax(1)]), W.max(1)


def evaluate(min_photos=2, min_organs=1, router_conf=0.0, emb_path=None, cache_dir=DATA_PROCESSED):
    heads, proj, classes = build_heads(cache_dir)
    router, router_acc = build_router(cache_dir)
    oi = list(classes).index(OTHER)
    mask = np.ones(len(classes), bool)
    mask[oi] = False
    genera = np.array([c.split()[0] for c in classes[mask]])
    ug = np.unique(genera)
    gmat = np.stack([(genera == g).astype(float) for g in ug])

    z = np.load(emb_path)
    E, path_index = _l2(z["descriptor"]), {p: i for i, p in enumerate(z["path"])}
    df = pd.read_parquet(cache_dir / "inat_observations.parquet")

    rows = []
    for r in df.itertuples():
        idx = [path_index[p] for p in r.local_paths if p in path_index]
        if len(idx) < min_photos:
            continue
        P, organs, conf = photo_posteriors(E[idx], heads, proj, router, len(classes))
        distinct = len({o for o, c in zip(organs, conf) if c >= router_conf})
        if distinct < min_organs:
            continue
        rows.append({"bucket": r.bucket, "species": r.species_name, "genus": r.genus,
                     "single": P[0], "fused": P.mean(0), "n_photos": len(idx), "n_organs": distinct})
    R = pd.DataFrame(rows)
    return R, classes, mask, oi, gmat, ug, router_acc


def report(R, classes, mask, oi, gmat, ug, label=""):
    if R.empty:
        print(f"  {label}: no observations pass the filter")
        return
    inc = (R["bucket"] == "in_catalog").values
    print(f"  {label} (n={len(R)}: {inc.sum()} in-catalog, "
          + ", ".join(f"{b} {(R['bucket'] == b).sum()}" for b in ("near_ood", "distant_ood")) + ")")
    for key in ("single", "fused"):
        P = np.stack(R[key].values)
        conf = 1 - P[:, oi]
        pred = classes[mask][P[:, mask].argmax(1)]
        gpred = ug[(P[:, mask] @ gmat.T).argmax(1)]
        parts = [f"species {(pred[inc] == R['species'].values[inc]).mean():.3f}",
                 f"genus {(gpred[inc] == R['genus'].values[inc]).mean():.3f}"]
        for b in ("distant_ood", "near_ood"):
            m = inc | (R["bucket"] == b).values
            n_neg = (R["bucket"] == b).sum()
            parts.append(f"{b.split('_')[0]}-OOD "
                         + (f"{roc_auc_score(inc[m].astype(int), conf[m]):.3f}" if n_neg >= 8 else "n/a"))
        print(f"    {key:7s} " + " | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="npz of iNat photo embeddings (descriptor, path)")
    ap.add_argument("--min-photos", type=int, default=2)
    ap.add_argument("--min-organs", type=int, default=1)
    ap.add_argument("--router-conf", type=float, default=0.0)
    args = ap.parse_args()

    R, classes, mask, oi, gmat, ug, racc = evaluate(
        args.min_photos, args.min_organs, args.router_conf, emb_path=args.emb)
    print(f"organ router test accuracy: {racc:.3f}")
    report(R, classes, mask, oi, gmat, ug,
           label=f">={args.min_photos} photos, >={args.min_organs} organs @conf>={args.router_conf}")


if __name__ == "__main__":
    main()
