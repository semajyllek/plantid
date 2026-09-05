"""Stage 1: closed-set accuracy AND open-set rejection, per encoder, per catalogue size.

Stage 0 could only measure discrimination, because no small-encoder background
cache existed. This measures the half that decides the concept: when you narrow
the catalogue to K species, can the model still tell that the 510 species you
*didn't* pick are not in it?

Design, per draw:
  train    subset train rows, labelled by species, plus background rows labelled
           __OTHER__ (class_weight balanced) -- mirrors inat_fusion.build_heads
  in-set   subset test rows
  near-OOD test rows of catalogue species NOT in the subset. This is the case
           that actually bites a narrow catalogue: pick 20 of 530 and the other
           510 are all near-OOD, many of them congeners of what you picked.
  far-OOD  held-out background rows, unseen in training

Metrics:
  top1      closed-set, argmax restricted to the K species (in-set rows only)
  auroc_nr  near-OOD detection, score = 1 - P(__OTHER__)
  auroc_fr  far-OOD detection, same score
Both AUROCs are prevalence-free, so they compare across encoders without
committing to a deployment rate. Utility-anchored coverage comes later, once an
encoder is chosen.
"""

import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from plantid.features.embed_background import catalog_species, load_background

DP = "data/processed"
ORGANS = ["leaf", "flower"]  # bark is 1,311 rows over 530 species -- too thin to subset
OTHER = "__OTHER__"
N_DRAWS = 15
BG_TRAIN_FRAC = 0.6
CS = [0.1, 1.0, 10.0, 100.0]


def _l2(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def load(variant):
    """Catalogue + background, with catalogue species removed from the background.

    Loading `background_*.npz` raw is wrong: the pool predates the current
    catalogue and still contains species we now want to *recognise*, so those
    rows would be labelled __OTHER__ while the same species' catalogue rows carry
    a species label -- contradictory supervision on the exact classes under test.
    `load_background(exclude_species=...)` is the documented fix and is what
    `inat_fusion.build_heads` uses.
    """
    cs = catalog_species()
    cat, bg = {}, {}
    for organ in ORGANS:
        d = np.load(f"{DP}/catalog_{organ}_{variant}.npz", allow_pickle=True)
        cat[organ] = (_l2(d["descriptor"]), d["species_name"].astype(str), d["split"].astype(str))
        b = load_background(organ, exclude_species=cs, variant=variant)
        bg[organ] = _l2(b["descriptor"])
    return cat, bg


BG_PER_K = None  # if set, background train rows = BG_PER_K * K, holding the
                 # species:background ratio constant across K. Without it the
                 # ratio shifts 5x between K=10 and K=50, which would confound
                 # "small encoders lose less on narrow catalogues" with "the
                 # reject class dominates the fit at small K".


def one_draw(cat, bg, species, rng):
    keep = set(species)
    Xtr, ytr, Xin, yin, Xva, yva, Xnear, Xfar = [], [], [], [], [], [], [], []
    for organ in ORGANS:
        E, names, split = cat[organ]
        m = np.array([n in keep for n in names])
        tr = m & (split == "train")
        Xtr.append(E[tr]); ytr.append(names[tr])
        va = m & (split == "val")
        Xva.append(E[va]); yva.append(names[va])
        te = m & (split == "test")
        Xin.append(E[te]); yin.append(names[te])
        nr = (~m) & (split == "test")
        Xnear.append(E[nr])

        B = bg[organ]
        cut = rng.permutation(len(B))
        n = int(BG_TRAIN_FRAC * len(B))
        if BG_PER_K:
            n = min(n, BG_PER_K * len(keep))
        Xtr.append(B[cut[:n]]); ytr.append(np.full(n, OTHER))
        Xfar.append(B[cut[int(BG_TRAIN_FRAC * len(B)):]])

    Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
    Xin, yin = np.vstack(Xin), np.concatenate(yin)
    Xva, yva = np.vstack(Xva), np.concatenate(yva)
    Xnear, Xfar = np.vstack(Xnear), np.vstack(Xfar)

    def fit(C):
        return LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(Xtr, ytr)

    def closed_top1(clf, X, y):
        cls = np.array(clf.classes_)
        sp = np.flatnonzero(cls != OTHER)
        return (cls[sp][clf.predict_proba(X)[:, sp].argmax(1)] == y).mean()

    # C is swept per encoder on val, never fixed: feature dimension and norm
    # scale differ across encoders, and in Stage 0 the sweep alone was worth 5pp
    # to MobileCLIP2-S0 -- enough to be mistaken for an encoder-capacity gap.
    best_c = max(CS, key=lambda C: closed_top1(fit(C), Xva, yva))
    clf = fit(best_c)
    cls = list(clf.classes_)
    oi = cls.index(OTHER)

    top1 = closed_top1(clf, Xin, yin)

    def inset_score(X):
        return 1.0 - clf.predict_proba(X)[:, oi]

    s_in = inset_score(Xin)
    auroc_nr = roc_auc_score(np.r_[np.ones(len(s_in)), np.zeros(len(Xnear))],
                             np.r_[s_in, inset_score(Xnear)])
    auroc_fr = roc_auc_score(np.r_[np.ones(len(s_in)), np.zeros(len(Xfar))],
                             np.r_[s_in, inset_score(Xfar)])
    return top1, auroc_nr, auroc_fr


def make_draws(all_species, K, rng, hard):
    by_genus = defaultdict(list)
    for s in all_species:
        by_genus[s.split()[0]].append(s)
    multi = [g for g, v in by_genus.items() if len(v) >= 2]
    out = []
    for _ in range(N_DRAWS):
        if hard:
            picked = []
            for g in rng.permutation(multi):
                picked += by_genus[g]
                if len(picked) >= K:
                    break
            out.append(picked[:K])
        else:
            out.append(list(rng.choice(all_species, K, replace=False)))
    return out


def main(variants, Ks):
    ref_cat, _ = load("bioclip2")
    all_species = np.array(sorted(set(ref_cat["leaf"][1]) | set(ref_cat["flower"][1])))
    print(f"{len(all_species)} catalogue species\n", flush=True)

    loaded = {v: load(v) for v in variants}
    for K in Ks:
        for hard in (False, True):
            arm = "HARD (congeners)" if hard else "EASY (random) "
            ds = make_draws(all_species, K, np.random.default_rng(K + hard), hard)
            print(f"=== K={K}  {arm}  {N_DRAWS} draws ===", flush=True)
            for v in variants:
                cat, bg = loaded[v]
                r = np.array([one_draw(cat, bg, s, np.random.default_rng(i))
                              for i, s in enumerate(ds)])
                t, n, f = r[:, 0], r[:, 1], r[:, 2]
                print(f"  {v:18s} top1 {t.mean():.4f} [{np.percentile(t,2.5):.4f},{np.percentile(t,97.5):.4f}]"
                      f"   auroc_near {n.mean():.4f}   auroc_far {f.mean():.4f}", flush=True)
            print(flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    for a in list(args):
        if a.startswith("--bg-per-k="):
            BG_PER_K = int(a.split("=")[1])
            args.remove(a)
            print(f"background train rows capped at {BG_PER_K} x K\n", flush=True)
    vs = [a for a in args if not a.isdigit()] or ["bioclip2", "mobileclip2_s0"]
    ks = [int(a) for a in args if a.isdigit()] or [20]
    main(vs, ks)
