import numpy as np

from plantid.eval.calibration import (
    apply_temperature,
    conformal_set,
    conformal_threshold,
    fit_temperature,
    genus_containment,
    reliability,
)


def _logits(n=400, n_classes=4, margin=0.7, scale=1.0, seed=0):
    """Logits with a *fixed* class margin, then scaled.

    Scaling preserves the argmax, so accuracy is held constant while confidence
    moves — which is what over/under-confidence means. The margin is small
    enough that accuracy stays well below 1, otherwise high confidence would be
    correct and there would be nothing to calibrate.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    z = rng.normal(0, 1.0, (n, n_classes))
    z[np.arange(n), y] += margin
    return z * scale, y


def test_fixture_accuracy_leaves_room_to_calibrate():
    z, y = _logits()
    acc = (z.argmax(1) == y).mean()
    assert 0.3 < acc < 0.8  # scaling won't change this; only confidence moves


def test_fit_temperature_cools_an_overconfident_model():
    z, y = _logits(scale=6.0)  # same accuracy, confidence near 1 -> wants T > 1
    # confidence far above accuracy is what "over-confident" means here
    assert apply_temperature(z, 1.0).max(1).mean() > (z.argmax(1) == y).mean() + 0.2
    t, _ = fit_temperature(z, y)
    assert t > 1.0
    before = apply_temperature(z, 1.0).max(1).mean()
    after = apply_temperature(z, t).max(1).mean()
    assert after < before


def test_fit_temperature_warms_an_underconfident_model():
    z, y = _logits(scale=0.1)  # same accuracy, confidence near chance
    t, _ = fit_temperature(z, y)
    assert t < 1.0


def test_apply_temperature_rows_are_distributions():
    z, _ = _logits()
    p = apply_temperature(z, 2.5)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_reliability_reports_zero_error_when_perfectly_calibrated():
    # confidence exactly equals accuracy in every bin
    conf = np.repeat([0.25, 0.75], 400)
    rng = np.random.default_rng(0)
    correct = np.concatenate([rng.random(400) < 0.25, rng.random(400) < 0.75])
    table, ece = reliability(conf, correct, n_bins=4)
    assert len(table) == 2
    assert ece < 0.05


def test_reliability_flags_overconfidence():
    conf = np.full(200, 0.95)
    correct = np.zeros(200, bool)
    _, ece = reliability(conf, correct)
    assert ece > 0.9


def test_conformal_threshold_achieves_nominal_coverage():
    rng = np.random.default_rng(0)
    n, k = 500, 5
    P = rng.random((n, k))
    P /= P.sum(1, keepdims=True)
    true_idx = rng.integers(0, k, n)
    qhat, used = conformal_threshold(P, true_idx, alpha=0.1)
    assert used == n
    covered = [true_idx[i] in conformal_set(P[i], qhat) for i in range(n)]
    assert np.mean(covered) >= 0.85  # nominal 0.90, finite-sample slack


def test_conformal_threshold_deduplicates_clusters():
    rng = np.random.default_rng(1)
    P = rng.random((60, 3))
    P /= P.sum(1, keepdims=True)
    clusters = np.repeat([f"sp{i}" for i in range(12)], 5)
    _, used = conformal_threshold(P, rng.integers(0, 3, 60), clusters=clusters)
    assert used == 12  # one observation per species, not 60


def test_genus_containment():
    genus_of = {0: "Sedum", 1: "Sedum", 2: "Acacia"}
    assert genus_containment(np.array([0, 1]), genus_of) == "Sedum"
    assert genus_containment(np.array([0, 2]), genus_of) is None
    assert genus_containment(np.array([]), genus_of) is None
