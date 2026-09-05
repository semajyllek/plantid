"""Projecting what a species list will give you, before spending any compute.

`plan` has to answer "what will I get" without fetching images or fitting a head,
so it interpolates the measured grid in `frontier.json` rather than running
anything. Two axes, both of which the measurements show matter:

  K                   catalogue size -- the encoder gap shrinks as K falls
  congener fraction   share of chosen species sharing a genus with another
                      chosen species -- drives the species-level share

The congener anchors are **measured, not assumed**. The "unrelated" arm is not
at 0.0: random draws from a 530-species catalogue over 172 genera collide often
enough to sit at 0.10 (K=10), 0.23 (K=20) and 0.44 (K=50). Interpolating from a
notional 0.0 would make well-separated sets look better than anything measured.

This is a projection from measurements on *a* catalogue, not a guarantee about
the user's. `build` measures the real thing and the card reports that instead.
"""

import json
from bisect import bisect_left
from pathlib import Path

GRID = json.loads((Path(__file__).parent / "frontier.json").read_text())
METRICS = ("top1", "coverage", "precision", "species_share")


def _interp(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _cells(encoder, K):
    """The easy/hard pair measured at this K, with their congener anchors."""
    rows = [r for r in GRID["cells"] if r["encoder"] == encoder and r["K"] == K]
    by_arm = {r["arm"]: r for r in rows}
    anchors = GRID["congener_anchors"][str(K)]
    return by_arm, anchors


def _at_K(encoder, K, cfrac, p_ood):
    """Interpolate between the easy and hard arms at a measured K."""
    by_arm, anchors = _cells(encoder, K)
    if "easy" not in by_arm or "hard" not in by_arm:
        return None
    out = {}
    for m in METRICS:
        lo = by_arm["easy"][m] if m == "top1" else by_arm["easy"]["p_ood"][p_ood][m]
        hi = by_arm["hard"][m] if m == "top1" else by_arm["hard"]["p_ood"][p_ood][m]
        c = min(max(cfrac, anchors["easy"]), anchors["hard"])
        out[m] = _interp(c, anchors["easy"], anchors["hard"], lo, hi)
    return out


def project(encoder: str, n_species: int, congener_frac: float, p_ood: float = 0.2) -> dict:
    """Projected top-1, coverage, precision and species-level share.

    Clamped to the measured range of K rather than extrapolated: outside 10-50
    species there is no measurement, and a projection that silently continues a
    trend past its evidence is the thing this project's conventions exist to
    prevent. `extrapolated` says which way it was clamped.
    """
    ks = sorted({r["K"] for r in GRID["cells"] if r["encoder"] == encoder})
    if not ks:
        raise ValueError(f"no measurements for encoder {encoder!r}")
    key = f"{p_ood:g}"
    if key not in GRID["p_ood_measured"]:
        raise ValueError(f"p_ood {p_ood} not measured; have {GRID['p_ood_measured']}")

    K = min(max(n_species, ks[0]), ks[-1])
    extrapolated = None if K == n_species else ("below" if n_species < ks[0] else "above")

    i = bisect_left(ks, K)
    if i < len(ks) and ks[i] == K:
        vals = _at_K(encoder, K, congener_frac, key)
    else:
        lo, hi = ks[i - 1], ks[i]
        a, b = _at_K(encoder, lo, congener_frac, key), _at_K(encoder, hi, congener_frac, key)
        vals = {m: _interp(K, lo, hi, a[m], b[m]) for m in METRICS}

    return {**vals, "K_used": K, "extrapolated": extrapolated,
            "encoder": encoder, "p_ood": p_ood, "congener_frac": congener_frac}


def measured_p_ood() -> list[float]:
    return [float(p) for p in GRID["p_ood_measured"]]
