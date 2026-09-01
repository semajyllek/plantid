"""Calibration for the per-organ heads, and conformal sets scoped to the catalogue.

Two separate jobs, often conflated:

**Temperature scaling** makes the number shown to a user mean something — a
stated 0.8 should be right about 80% of the time. On a *single* classifier a
temperature is monotone and cannot change any threshold decision. Here it can,
and that is worth being precise about: T is applied to each organ head's logits
*before* the router mixes them, and a mixture of tempered distributions is not a
monotone transform of the tempered mixture. It also changes which photo
`combiners.trimmed` drops, since that selection reads confidence. So T is a
parameter of the fusion, not a cosmetic rescale, and its effect on the decision
is measured rather than assumed.

**Conformal prediction** is scoped narrowly and deliberately. Its guarantee is
over the true label, so it says nothing about an out-of-catalogue plant whose
true label is not in the label space — using an empty conformal set as a
"decline" signal just reduces to thresholding max-softmax, the weakest of the
nested scores. What survives is the useful part: *if the prediction set falls
entirely inside one genus, answer at genus level*. That is an adaptive
alternative to the fixed `t_species` in `rejection.py`, and it is what this
module exposes.

Exchangeability is also only approximate — the evaluation set has ~6
correlated observations per species — so `conformal_threshold` takes one
observation per cluster by default.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def fit_temperature(logits, labels, bounds=(0.05, 20.0)):
    """Single scalar T minimising negative log-likelihood on held-out data."""
    classes = np.unique(labels)
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[v] for v in labels])

    def nll(t):
        return -log_softmax(logits / t, axis=1)[np.arange(len(y)), y].mean()

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(res.x), float(res.fun)


def apply_temperature(logits, t):
    return softmax(logits / t, axis=1)


def reliability(confidence, correct, n_bins=10):
    """Binned confidence vs observed accuracy, plus expected calibration error.

    Only meaningful on in-catalogue queries: where the true class is absent from
    the label space, "accuracy" is undefined and ECE is uninterpretable.
    """
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if not m.any():
            continue
        conf, acc = confidence[m].mean(), correct[m].mean()
        ece += m.mean() * abs(conf - acc)
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                     "confidence": float(conf), "accuracy": float(acc),
                     "gap": float(conf - acc)})
    return pd.DataFrame(rows), float(ece)


def conformal_threshold(posteriors, true_index, alpha=0.05, clusters=None, seed=0):
    """Split-conformal qhat on the nonconformity score 1 - P(true class).

    Valid only *conditional on the plant being in the catalogue*. `clusters`
    restores approximate exchangeability by sampling one observation per cluster
    (species), since correlated repeats otherwise break the i.i.d. assumption
    the coverage guarantee rests on.
    """
    posteriors = np.asarray(posteriors, float)
    scores = 1.0 - posteriors[np.arange(len(true_index)), np.asarray(true_index, int)]
    if clusters is not None:
        rng = np.random.RandomState(seed)
        keep = [rng.choice(np.flatnonzero(np.asarray(clusters) == c)) for c in sorted(set(clusters))]
        scores = scores[np.array(keep)]
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher")), n


def conformal_set(posterior, qhat):
    """Indices whose posterior is large enough to stay in the prediction set."""
    return np.flatnonzero(posterior >= 1.0 - qhat)


def genus_containment(indices, genus_of):
    """The shared genus if the whole set sits in one, else None.

    This is the adaptive version of `t_species`: a set spanning several species
    of one genus is exactly the case where the genus answer is the honest one.
    """
    if len(indices) == 0:
        return None
    genera = {genus_of[i] for i in indices}
    return genera.pop() if len(genera) == 1 else None


def catalog_logits(organ, heads, cache_dir=None, split="val"):
    """Held-out logits and labels for one organ head, for temperature fitting.

    Uses the catalogue's own val split — the heads were fitted on train — so
    fitting T does not spend the iNaturalist calibration split.
    """
    from plantid.config import DATA_PROCESSED
    from plantid.data.curation import curated_name
    from plantid.eval.inat_fusion import _l2
    from plantid.features.embed_catalog import load_catalog

    d = load_catalog(organ, cache_dir=cache_dir or DATA_PROCESSED)
    m = d["split"] == split
    E = _l2(d["descriptor"])[m]
    names = np.array([curated_name(n) or "" for n in d["species_name"]])[m]
    return heads[organ].decision_function(E), names
