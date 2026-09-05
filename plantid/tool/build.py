"""Fit a head for a chosen species list, measure it honestly, write a bundle.

The encoder is frozen and shared; what gets built per user is a logistic head
plus two thresholds. That is the whole personalisation story, and it is why this
costs CPU-seconds rather than GPU-hours: at 20 species and 512 dimensions the
head is ~40 KB against an encoder of 17.9 MB.

Measurement here is not the same as `plan`'s projection. `plan` interpolates a
grid measured on someone else's catalogue; `build` fits on the user's actual data
and evaluates on held-out rows of it, so the card reports the real thing.

Three evaluation buckets, matching `eval/rejection.py`:

  in_catalog   held-out rows of the chosen species
  near_ood     rows of pool species outside the set that share a genus with it
  distant_ood  held-out background rows

near_ood is built from the relatives the user did not choose, which is the
failure mode a narrow catalogue actually has. A build whose pool contains no
such relatives reports that, rather than quietly scoring rejection on easy
negatives alone.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from plantid.config import DATA_PROCESSED
from plantid.eval.rejection import (
    DECLINE,
    GENUS,
    SPECIES,
    UTILITY,
    cluster_bootstrap,
    decide,
    deployment_weights,
    fit_thresholds,
    genus_matrix,
    make_splits,
)
from plantid.features.embed_background import catalog_species, load_background
from plantid.tool.species import canonical, genus

ORGANS = ("leaf", "flower")
OTHER = "__OTHER__"
BG_TRAIN_FRAC = 0.6
OOD_MIX = {"near_ood": 0.32, "distant_ood": 0.68}
BUNDLE_VERSION = 1


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    frame: pd.DataFrame          # per-observation scores are added after fitting
    X_eval: np.ndarray
    truth: np.ndarray
    bucket: np.ndarray
    counts: dict
    cluster: np.ndarray          # real species name per eval row, background included


def _l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def load_local(variant: str, chosen: list[str], cache_dir=DATA_PROCESSED,
               seed: int = 0) -> Dataset:
    """Assemble train/eval matrices for `chosen` from the local catalogue caches."""
    rng = np.random.default_rng(seed)
    keep = set(chosen)
    keep_genera = {genus(s) for s in chosen}
    cs = catalog_species(cache_dir)

    # `cluster` carries the real species name for *every* eval row, including
    # background. `truth` cannot: background rows must be labelled __OTHER__ so
    # that species_ok/genus_ok are False for them. Conflating the two put all
    # background rows in one cluster, and `make_splits` then sent every one of
    # them to test (clusters[:1//2] is empty) -- thresholds were being fitted
    # with no distant negatives at all.
    Xtr, ytr, ev, truth, cluster, bucket = [], [], [], [], [], []
    counts = {"in_catalog": 0, "near_ood": 0, "distant_ood": 0, "train": 0}
    missing_organ = []

    for organ in ORGANS:
        path = Path(cache_dir) / f"catalog_{organ}_{variant}.npz"
        if not path.exists():
            missing_organ.append(organ)
            continue
        d = np.load(path, allow_pickle=True)
        E = _l2(d["descriptor"])
        names = np.array([canonical(n) or n for n in d["species_name"].astype(str)])
        split = d["split"].astype(str)

        m = np.isin(names, list(keep))
        tr = m & (split == "train")
        Xtr.append(E[tr]); ytr.append(names[tr]); counts["train"] += int(tr.sum())

        te = m & (split == "test")
        ev.append(E[te]); truth.append(names[te]); cluster.append(names[te])
        bucket += ["in_catalog"] * int(te.sum()); counts["in_catalog"] += int(te.sum())

        gm = np.array([genus(n) in keep_genera for n in names])
        nr = (~m) & gm & (split == "test")
        ev.append(E[nr]); truth.append(names[nr]); cluster.append(names[nr])
        bucket += ["near_ood"] * int(nr.sum()); counts["near_ood"] += int(nr.sum())

        bg = load_background(organ, exclude_species=cs, variant=variant,
                             cache_dir=cache_dir)
        B = _l2(bg["descriptor"])
        bg_names = np.array([canonical(n) or n for n in bg["species_name"].astype(str)])
        cut = rng.permutation(len(B))
        n = int(BG_TRAIN_FRAC * len(B))
        Xtr.append(B[cut[:n]]); ytr.append(np.full(n, OTHER))
        far = cut[n:]
        ev.append(B[far]); truth.append(np.full(len(far), OTHER))
        cluster.append(bg_names[far])
        bucket += ["distant_ood"] * len(far); counts["distant_ood"] += len(far)

    if missing_organ:
        counts["missing_organs"] = missing_organ
    if not Xtr:
        raise FileNotFoundError(
            f"no embedding caches for variant {variant!r} in {cache_dir}. "
            f"Run plantid.features.embed_catalog / embed_background first."
        )
    return Dataset(np.vstack(Xtr), np.concatenate(ytr), pd.DataFrame(),
                   np.vstack(ev), np.concatenate(truth), np.array(bucket), counts,
                   np.concatenate(cluster))


def load_rows(rows, encoder_variant: str, background=None, seed: int = 0) -> Dataset:
    """Assemble a Dataset from a caller-supplied source (`tool/sources.py`).

    The tool does not fetch. It is handed images, a manifest or precomputed
    vectors, and the domain that produced them keeps ownership of how.

    `background` is an optional second source of negatives. Without it there is
    no reject class: the model is closed-set, cannot decline, and the card says
    so rather than implying a rejection capability that was never fitted.
    """
    def _vecs(r):
        if r.descriptor is not None:
            return _l2(np.asarray(r.descriptor, dtype="float32"))
        from plantid.features.pretrained import embed_images, load_encoder
        model, preprocess, device = load_encoder(encoder_variant)
        return _l2(embed_images(list(r.path), model, preprocess, device))

    X = _vecs(rows)
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(rows.cluster)))
    rng.shuffle(uniq)
    tr = np.isin(rows.cluster, uniq[: len(uniq) // 2])

    Xtr, ytr = [X[tr]], [rows.label[tr]]
    ev, truth, cluster, bucket = [X[~tr]], [rows.label[~tr]], [rows.cluster[~tr]], \
        ["in_catalog"] * int((~tr).sum())
    counts = {"in_catalog": int((~tr).sum()), "near_ood": 0, "distant_ood": 0,
              "train": int(tr.sum())}
    notes = list(rows.notes)

    if background is not None:
        B = _vecs(background)
        cut = rng.permutation(len(B))
        n = int(BG_TRAIN_FRAC * len(B))
        Xtr.append(B[cut[:n]]); ytr.append(np.full(n, OTHER))
        far = cut[n:]
        ev.append(B[far]); truth.append(np.full(len(far), OTHER))
        cluster.append(background.cluster[far])
        bucket += ["distant_ood"] * len(far)
        counts["distant_ood"] = len(far)
        counts["train"] += n
    else:
        notes.append("no background supplied: closed-set only, the model cannot decline")

    counts["notes"] = notes
    counts["has_clusters"] = bool(rows.has_clusters)
    return Dataset(np.vstack(Xtr), np.concatenate(ytr), pd.DataFrame(),
                   np.vstack(ev), np.concatenate(truth), np.array(bucket), counts,
                   np.concatenate(cluster))


def fit_head(ds: Dataset, C: float = 10.0) -> LogisticRegression:
    return LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(
        ds.X_train, ds.y_train
    )


def score_frame(clf, ds: Dataset) -> pd.DataFrame:
    """Per-observation cascade inputs and outcomes."""
    classes = np.array(clf.classes_)
    mask = classes != OTHER
    gmat, ug = genus_matrix(classes, mask)

    cata = clf.predict_proba(ds.X_eval)[:, mask]
    gscore = cata @ gmat.T
    sp_pred = classes[mask][cata.argmax(1)]
    gn_pred = ug[gscore.argmax(1)]
    true_genus = np.array([genus(t) if t != OTHER else OTHER for t in ds.truth])

    return pd.DataFrame({
        "species_conf": cata.max(1),
        "genus_conf": gscore.max(1),
        "species_ok": sp_pred == ds.truth,
        "genus_ok": gn_pred == true_genus,
        "pred_species": sp_pred,
        "pred_genus": gn_pred,
        "truth": ds.truth,
        "in_catalog": ds.bucket == "in_catalog",
        "bucket": ds.bucket,
        # Clustering identity, not the label: `make_splits` and the bootstrap
        # both key on these, and both need background rows to carry their real
        # species rather than collapsing into a single __OTHER__ cluster.
        "species": ds.cluster,
        "genus": np.array([genus(c) for c in ds.cluster]),
    })


def _ci(numer, denom, clusters, n=2000, seed=0):
    """Cluster-bootstrapped 95% interval for the ratio sum(numer)/sum(denom).

    A ratio, not a mean, because coverage and precision are *prevalence-weighted*
    across buckets. Bootstrapping the unweighted mean of the same rows estimates
    a different quantity entirely -- it put precision's interval at 22-77% around
    a point estimate of 96%, since unweighted it is dominated by the OOD rows
    that `deployment_weights` deliberately down-weights.
    """
    numer, denom, clusters = (np.asarray(numer, float), np.asarray(denom, float),
                              np.asarray(clusters))
    if len(numer) == 0 or denom.sum() <= 0 or len(set(clusters)) < 2:
        return None
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(clusters)))
    index = {c: np.flatnonzero(clusters == c) for c in uniq}
    out = []
    for _ in range(n):
        idx = np.concatenate([index[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        d = denom[idx].sum()
        if d > 0:
            out.append(numer[idx].sum() / d)
    if not out:
        return None
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def hazard_metrics(te, lv, hazards, seed=0) -> dict:
    """For each consequential label: how often is it given a *non*-consequential name?

    This is the union over every wrong answer, and it is not optional. Measured on
    Oregon's lethal plants (`OREGON_SAFETY_FINDINGS.md`), no single confusion
    exceeded 2.5% while the union reached **6.7%** -- because the errors scatter
    across many different harmless-looking labels. A per-pair report passes a
    model that a union report fails.

    Being named as *another* consequential label is a wrong answer but not a
    dangerous one: the user still does not eat it. Those are counted separately
    rather than folded in.

    **Genus answers count.** A coarse answer naming a group that contains no
    consequential label is just as actionable as a wrong species name -- "it is a
    Lomatium" for poison hemlock is precisely the error that kills foragers. Only
    declining, or answering with the hazard's own group, is safe.
    """
    if not hazards:
        return {}
    hz = set(hazards)
    hz_genera = {h.split()[0] for h in hz}
    truth = te["truth"].to_numpy()
    pred = te["pred_species"].to_numpy()
    pgen = te["pred_genus"].to_numpy()
    named = lv == SPECIES
    genus_only = (lv != SPECIES) & (lv != DECLINE)
    out = {}
    for label in sorted(hz):
        m = truth == label
        if not m.any():
            continue
        # answered as something the user would treat as harmless
        sp_safe = named[m] & (pred[m] != label) & ~np.isin(pred[m], list(hz))
        gn_safe = genus_only[m] & ~np.isin(pgen[m], list(hz_genera))
        wrong_safe = sp_safe | gn_safe
        wrong_haz = (named[m] & (pred[m] != label) & np.isin(pred[m], list(hz))) | \
                    (genus_only[m] & np.isin(pgen[m], list(hz_genera)))
        ci = _ci(wrong_safe.astype(float), np.ones(m.sum()),
                 te["species"].to_numpy()[m], seed=seed)
        out[label] = {
            "n": int(m.sum()),
            "declined": float((lv[m] == DECLINE).mean()),
            "named_correctly": float((named[m] & (pred[m] == label)).mean()),
            "named_other_hazard": float(wrong_haz.mean()),
            "named_non_hazard": float(wrong_safe.mean()),
            "ci": ci,
            # No interval when the catalogue offers no cluster inside one label:
            # its images are not grouped by individual plant, so a row-level
            # bootstrap would treat several photographs of one plant as
            # independent -- the error CLAUDE.md's first convention exists to
            # prevent. Sources that carry observation ids (iNaturalist) do get one.
            "ci_unavailable_reason": None if ci else "no cluster within a single label",
        }
    return out


def fit_and_measure(df: pd.DataFrame, p_ood: float, seed: int = 0,
                    hazards=None) -> dict:
    """Fit thresholds on a clustered calibration half, report on the other."""
    fold = make_splits(df, seed=seed)
    cal, te = df[fold == "calib"], df[fold == "test"]
    if cal.empty or te.empty:
        raise ValueError("calibration or test split is empty; too few observations")

    w_cal = deployment_weights(cal["bucket"].to_numpy(), p_ood=p_ood, ood_mix=OOD_MIX)
    (tg, ts), _ = fit_thresholds(
        cal["species_conf"].to_numpy(), cal["genus_conf"].to_numpy(),
        cal["species_ok"].to_numpy(), cal["genus_ok"].to_numpy(),
        cal["in_catalog"].to_numpy(), sample_weight=w_cal,
    )

    lv = decide(te["species_conf"].to_numpy(), te["genus_conf"].to_numpy(), tg, ts)
    w = deployment_weights(te["bucket"].to_numpy(), p_ood=p_ood, ood_mix=OOD_MIX)
    answered = lv != DECLINE
    correct = ((lv == SPECIES) & te["species_ok"].to_numpy()) | \
              ((lv == GENUS) & te["genus_ok"].to_numpy())
    inc = te["in_catalog"].to_numpy()

    per_bucket = {}
    for b in ("in_catalog", "near_ood", "distant_ood"):
        bm = te["bucket"].to_numpy() == b
        if bm.any():
            per_bucket[b] = {
                "n": int(bm.sum()),
                "answered": float((lv[bm] != DECLINE).mean()),
                "correct_when_answered": float(
                    correct[bm & answered].mean()) if (bm & answered).any() else None,
            }

    # Cluster bootstrap over species, never over rows -- CLAUDE.md's first
    # convention, and it exists because row-level intervals have twice produced
    # effects here that failed to replicate. A card at 14 species rests on very
    # few clusters, so a wide interval is itself the finding the user needs.
    clusters = te["species"].to_numpy()
    ones = np.ones(len(te))
    ci = {
        "coverage": _ci(w * answered, w, clusters),
        "precision": _ci(w * answered * correct, w * answered, clusters),
        "species_share": _ci((lv == SPECIES) & inc, inc * ones, clusters),
        "closed_set_top1": _ci(te["species_ok"].to_numpy() & inc, inc * ones, clusters),
    }

    return {
        "t_genus": float(tg), "t_species": float(ts), "p_ood": p_ood,
        "coverage": float(w[answered].sum() / w.sum()),
        "precision": float(w[answered & correct].sum() / w[answered].sum())
        if answered.any() else None,
        "species_share": float((lv[inc] == SPECIES).mean()) if inc.any() else None,
        "closed_set_top1": float(te["species_ok"].to_numpy()[inc].mean()) if inc.any() else None,
        "ci": ci,
        "n_species_clusters": int(len(set(clusters[inc]))),
        "per_bucket": per_bucket,
        "hazard": hazard_metrics(te, lv, hazards, seed=seed),
        "n_calib": int(len(cal)), "n_test": int(len(te)),
    }


def save_bundle(out: Path, clf, chosen, encoder, metrics, composition, counts,
                source: str, hazards=None) -> Path:
    """Head weights, thresholds, and everything needed to reproduce the claim."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "head.npz", coef=clf.coef_, intercept=clf.intercept_,
                        classes=np.asarray(clf.classes_, dtype=str))
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "encoder": encoder,
        "species": chosen,
        "source": source,
        "hazards": sorted(hazards or []),
        "counts": counts,
        "composition": {k: v for k, v in composition.items()
                        if k != "outside_congeners" and not k.startswith("_")},
        "outside_congeners": composition.get("outside_congeners", {}),
        "metrics": metrics,
        "utility": UTILITY,
        "ood_mix": OOD_MIX,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def load_bundle(path: Path) -> dict:
    return json.loads((Path(path) / "manifest.json").read_text())
