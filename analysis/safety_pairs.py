"""Fatal-direction rates on the pre-registered Oregon safety pairs.

Every pair is reported on its own row. Nothing is averaged across pairs: an
average over these is exactly what hides the case that matters, which is why
SAFETY_PREREG.md names them individually and in advance.

Primary result is a *curve* -- fatal rate against coverage -- because a single
number at one arbitrary operating point tells you nothing about whether a safer
operating point exists. Secondary result runs the project's actual
species/genus/decline cascade so these numbers connect to the rest of the repo.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from plantid.eval.rejection import (
    DECLINE,
    SPECIES,
    decide,
    deployment_weights,
    fit_thresholds,
    genus_matrix,
)
from plantid.features.embed_background import catalog_species, load_background

S = Path("/private/tmp/claude-501/-Users-jameskelly-Documents-plantid/"
         "8f28d0ec-b6fd-4bd9-9ee3-a9ec10992c6b/scratchpad/safety")
OTHER = "__OTHER__"

# (truth, mistaken_for, why) -- declared in SAFETY_PREREG.md before fitting.
PAIRS = [
    ("Conium maculatum", "Daucus carota", "textbook fatal confusion"),
    ("Conium maculatum", "Lomatium nudicaule", "forager digs hemlock root"),
    ("Conium maculatum", "Lomatium utriculatum", "forager digs hemlock root"),
    ("Conium maculatum", "Anthriscus caucalis", "umbel / fern-leaf"),
    ("Conium maculatum", "Osmorhiza berteroi", "sweet cicely is foraged"),
    ("Cicuta douglasii", "Heracleum maximum", "water hemlock vs cow parsnip"),
    ("Cicuta douglasii", "Lomatium dissectum", "both wetland-adjacent, both dug"),
    ("Sambucus racemosa", "Sambucus cerulea", "red toxic raw, blue not"),
    ("Toxicodendron diversilobum", "Rubus ursinus", "poison oak vs blackberry"),
    ("Rubus armeniacus", "Rubus ursinus", "invasive vs native (management)"),
    ("Veratrum viride", "Veratrum californicum", "within-genus control"),
]
LETHAL = {"Conium maculatum", "Cicuta douglasii"}


def _l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def load(variant):
    """Embeddings with duplicate rows dropped.

    `safety_fetch` paginated with order_by=votes, which is unstable when votes
    tie, so 170 observations came back on more than one page. No leakage -- the
    split is by obs_id -- but those observations would be double-weighted, and
    they concentrate in Sambucus racemosa, which is a reported pair.
    """
    d = np.load(S / f"emb_{variant}.npz", allow_pickle=True)
    m = pd.read_parquet(S / "manifest.parquet").reset_index(drop=True)
    keep = ~m.duplicated("local_path").to_numpy()
    n = len(d["descriptor"])
    if n == len(m):                 # embedded from the raw manifest
        sel = keep
    elif n == int(keep.sum()):      # embedded from an already-deduped manifest
        sel = np.ones(n, bool)
    else:
        raise AssertionError(f"{variant}: {n} embeddings match neither the manifest "
                             f"({len(m)}) nor its deduped length ({int(keep.sum())})")
    X, y, obs = d["descriptor"][sel], d["species_name"].astype(str)[sel], \
        d["obs_id"].astype(str)[sel]
    # Dedupe drops duplicate *rows*, not observations: the 170 doubled
    # observations keep one copy each, so the observation count is unchanged.
    assert len(X) == 6023, f"expected 6023 rows after dedupe, got {len(X)}"
    assert len(set(obs)) == 4057, f"expected 4057 observations, got {len(set(obs))}"
    return _l2(X), y, obs


def split_by_observation(obs, seed=0):
    """Half the observations to train, half to test. Never split within one."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(obs)))
    rng.shuffle(uniq)
    train = set(uniq[: len(uniq) // 2])
    return np.array([o in train for o in obs])


def boot(values, clusters, n=2000, seed=0):
    """Interval over observations, not images."""
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    if len(values) == 0 or len(set(clusters)) < 2:
        return None
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(clusters)))
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    out = []
    for _ in range(n):
        pick = np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        if len(pick):
            out.append(values[pick].mean())
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else None


def fit(X, y, tr):
    return LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced").fit(X[tr], y[tr])


def curve(variant):
    X, y, obs = load(variant)
    tr = split_by_observation(obs)
    clf = fit(X, y, tr)
    classes = np.array(clf.classes_)
    P = clf.predict_proba(X[~tr])
    pred, conf = classes[P.argmax(1)], P.max(1)
    yt, ot = y[~tr], obs[~tr]

    print(f"\n{'='*94}")
    print(f"{variant}   27-way top-1 {np.mean(pred == yt):.4f}   "
          f"train {tr.sum()} imgs / test {(~tr).sum()} imgs")
    print(f"{'='*94}")
    print("fatal-direction rate P(predict B | truth A), by coverage on A\n")
    covs = [1.0, 0.75, 0.50, 0.25]
    print(f"{'truth':26s} {'mistaken for':22s} " + "".join(f"{int(c*100):>6d}%" for c in covs)
          + f"{'sep':>8s}")
    for a, b, _ in PAIRS:
        m = yt == a
        if m.sum() == 0:
            continue
        ca, pa = conf[m], pred[m]
        cells = []
        for c in covs:
            t = np.quantile(ca, 1 - c)
            ans = ca >= t
            cells.append(f"{np.mean((pa == b) & ans):>7.3f}" if ans.any() else "      —")
        mb = yt == b
        sub = np.concatenate([m.nonzero()[0], mb.nonzero()[0]])
        st, sp = yt[sub], pred[sub]
        sep = 0.5 * ((sp[st == a] == a).mean() + (sp[st == b] == b).mean())
        print(f"{a:26s} {b:22s} " + "".join(cells) + f"{sep:8.3f}")

    print("\nat full coverage, with 95% CI over observations:")
    for a, b, why in PAIRS:
        m = yt == a
        if m.sum() == 0:
            continue
        v = ((pred[m] == b)).astype(float)
        ci = boot(v, ot[m])
        s = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "—"
        print(f"  {a:26s} -> {b:22s} {v.mean():6.3f} {s:>16s}   {why}")

    print("\nwhere the lethal species actually go when misread:")
    for a in sorted(LETHAL):
        m = yt == a
        if m.sum() == 0:
            continue
        wrong = pred[m][pred[m] != a]
        vc = pd.Series(wrong).value_counts()
        tot = m.sum()
        print(f"  {a} (n={tot} imgs, {len(set(ot[m]))} obs): "
              f"{len(wrong)/tot:.3f} misread")
        for k, n in vc.head(4).items():
            print(f"      -> {k:28s} {n/tot:.3f}")


def cascade(variant):
    """The declared three-way cascade, with the existing background pool as reject."""
    X, y, obs = load(variant)
    tr = split_by_observation(obs)

    cs = catalog_species()
    bg = np.vstack([_l2(load_background(o, exclude_species=cs, variant=variant)["descriptor"])
                    for o in ("leaf", "flower")])
    rng = np.random.default_rng(0)
    cut = rng.permutation(len(bg))
    n = int(0.6 * len(bg))
    Xtr = np.vstack([X[tr], bg[cut[:n]]])
    ytr = np.concatenate([y[tr], np.full(n, OTHER)])
    clf = LogisticRegression(max_iter=4000, C=10.0, class_weight="balanced").fit(Xtr, ytr)

    Xev = np.vstack([X[~tr], bg[cut[n:]]])
    yev = np.concatenate([y[~tr], np.full(len(bg) - n, OTHER)])
    oev = np.concatenate([obs[~tr], np.array([f"bg{i}" for i in range(len(bg) - n)])])
    inc = yev != OTHER

    classes = np.array(clf.classes_)
    mask = classes != OTHER
    gmat, ug = genus_matrix(classes, mask)
    cata = clf.predict_proba(Xev)[:, mask]
    sconf, gconf = cata.max(1), (cata @ gmat.T).max(1)
    sp_pred = classes[mask][cata.argmax(1)]
    gn_pred = ug[(cata @ gmat.T).argmax(1)]
    tg_true = np.array([t.split()[0] if t != OTHER else OTHER for t in yev])

    df = pd.DataFrame({"species_conf": sconf, "genus_conf": gconf,
                       "species_ok": sp_pred == yev, "genus_ok": gn_pred == tg_true,
                       "in_catalog": inc, "bucket": np.where(inc, "in_catalog", "distant_ood"),
                       "species": yev, "genus": tg_true})
    w = deployment_weights(df["bucket"].to_numpy(), p_ood=0.2,
                           ood_mix={"distant_ood": 1.0})
    (tgen, tsp), _ = fit_thresholds(df.species_conf.to_numpy(), df.genus_conf.to_numpy(),
                                    df.species_ok.to_numpy(), df.genus_ok.to_numpy(),
                                    df.in_catalog.to_numpy(), sample_weight=w)
    lv = decide(df.species_conf.to_numpy(), df.genus_conf.to_numpy(), tgen, tsp)

    print(f"\n  cascade thresholds  t_genus={tgen:.4f}  t_species={tsp:.4f}")
    print(f"  {'species':26s} {'declined':>9s} {'genus-only':>11s} {'named':>7s} {'named WRONG':>12s}")
    for a in dict.fromkeys(a for a, _, _ in PAIRS):   # each species once, not once per pair
        m = (yev == a)
        if m.sum() == 0:
            continue
        dec = (lv[m] == DECLINE).mean()
        nsp = (lv[m] == SPECIES)
        gen = ((lv[m] != DECLINE) & ~nsp).mean()
        wrong = (nsp & ~df.species_ok.to_numpy()[m]).mean()
        flag = "  <-- LETHAL" if a in LETHAL else ""
        print(f"  {a:26s} {dec:9.3f} {gen:11.3f} {nsp.mean():7.3f} {wrong:12.3f}{flag}")


if __name__ == "__main__":
    for v in sys.argv[1:] or ["bioclip2"]:
        if not (S / f"emb_{v}.npz").exists():
            print(f"{v}: not embedded")
            continue
        curve(v)
        cascade(v)
