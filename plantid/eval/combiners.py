"""Ways to combine per-photo posteriors into one observation-level posterior.

Averaging is the naive choice and is what `inat_fusion` used first. It is also
probably wrong: photos of one plant are *correlated*, so treating them as
independent evidence (geometric mean / log-odds sum) over-sharpens, while a
plain mean lets a bad photo drag a good one down. The useful region is between
mean and max.

Each combiner takes `P` (n_photos x n_classes, rows sum to 1) plus the routed
organ per photo and the index of the reject class, and returns one posterior.
`conf` throughout means 1 - P(__OTHER__): how much the head believes the photo
shows something in the catalog.
"""

import numpy as np

EPS = 1e-12


def _norm(v):
    s = v.sum()
    return v / s if s > 0 else np.full_like(v, 1.0 / len(v))


def mean(P, organs, oi):
    """Arithmetic mean — the baseline."""
    return P.mean(0)


def gmean(P, organs, oi):
    """Geometric mean == treating photos as independent evidence. Sharpens."""
    return _norm(np.exp(np.log(P + EPS).mean(0)))


def max_conf(P, organs, oi):
    """Just use the single most confident photo — ignore the rest."""
    return P[np.argmax(1 - P[:, oi])]

def conf_weighted(P, organs, oi):
    """Mean weighted by each photo's own confidence."""
    w = 1 - P[:, oi]
    return P.mean(0) if w.sum() <= 0 else (w[:, None] * P).sum(0) / w.sum()


def top2_mean(P, organs, oi):
    """Mean of the two most confident photos — drops the weakest evidence."""
    if len(P) <= 2:
        return P.mean(0)
    idx = np.argsort(-(1 - P[:, oi]))[:2]
    return P[idx].mean(0)


def trimmed(P, organs, oi):
    """Drop the single least confident photo, average the rest."""
    if len(P) <= 2:
        return P.mean(0)
    idx = np.argsort(-(1 - P[:, oi]))[:-1]
    return P[idx].mean(0)


def organ_best(P, organs, oi):
    """One vote per organ: best photo of each organ, then average across organs.

    Directly targets the correlation problem — three shots of the same flower
    collapse to one vote instead of three.
    """
    picks = []
    for o in set(organs):
        m = np.flatnonzero(np.asarray(organs) == o)
        picks.append(P[m[np.argmax(1 - P[m, oi])]])
    return np.mean(picks, axis=0)


def power_mean(P, organs, oi, p=2.0):
    """Power mean: p=1 is the arithmetic mean, larger p leans toward max."""
    return _norm((P ** p).mean(0) ** (1.0 / p))


def median(P, organs, oi):
    """Robust to one wild photo."""
    return _norm(np.median(P, axis=0))


COMBINERS = {
    "single (1st photo)": lambda P, o, i: P[0],
    "mean (baseline)": mean,
    "geometric mean": gmean,
    "median": median,
    "max-confidence": max_conf,
    "confidence-weighted": conf_weighted,
    "top-2 mean": top2_mean,
    "trimmed mean": trimmed,
    "organ-best": organ_best,
    "power mean p=2": power_mean,
    "power mean p=4": lambda P, o, i: power_mean(P, o, i, p=4.0),
}
