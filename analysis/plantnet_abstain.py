"""Could Pl@ntNet just threshold its own score and get our abstention for free?

The claim to test: our "almost never confidently wrong" property is not a moat,
because Pl@ntNet returns calibrated-looking scores and could decline below a
cutoff in an afternoon.

So build that system out of their own cached responses and measure it:

  A. species-only threshold — decline when top-1 score < t
  B. the full cascade we use — sum their scores by genus, decline on low genus
     confidence, back off to genus when species confidence is low

If either matches us at equal coverage, abstention is a feature they can copy and
not a differentiator. If neither does, the reject class is doing something a
threshold cannot.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


from plantid.config import DATA_PROCESSED as D  # noqa: E402
from plantid.eval.combiners import trimmed  # noqa: E402
from plantid.eval.headtohead import alias_sets, boot, correct, genus_of  # noqa: E402
from plantid.eval.rejection import (  # noqa: E402
    DECLINE, GENUS, SPECIES, build_observations, decide,
)

CACHE = D / "headtohead"
A = alias_sets()


def load():
    h = pd.read_parquet(D / "headtohead.parquet")
    h = h[h["plantnet_top1"].notna()].reset_index(drop=True)
    rows = []
    for r in h.itertuples():
        raw = json.loads((CACHE / f"plantnet_{r.obs_id}.json").read_text())["results"]
        res = [(x["name"], float(x["score"])) for x in raw if len(x["name"].split()) >= 2]
        if not res:
            continue
        # their scores over the returned candidates; genus confidence is the
        # summed mass of a genus, exactly as our genus score is built
        gsum = {}
        for name, s in res:
            gsum[genus_of(name)] = gsum.get(genus_of(name), 0.0) + s
        gbest = max(gsum, key=gsum.get)
        rows.append({"obs_id": r.obs_id, "truth": r.truth,
                     "sp": res[0][0], "sp_conf": res[0][1],
                     "gn": gbest, "gn_conf": gsum[gbest]})
    p = pd.DataFrame(rows)
    p["sp_ok"] = [correct(a, b, A) for a, b in zip(p.sp, p.truth)]
    p["gn_ok"] = [genus_of(a) in {genus_of(x) for x in A.get(b, {b})}
                  for a, b in zip(p.gn, p.truth)]
    return p


def curve(conf, ok, grid=np.linspace(0, 1, 201)):
    """Precision at each achievable coverage, thresholding on `conf`."""
    out = []
    for t in grid:
        m = conf >= t
        if m.sum() >= 10:
            out.append((m.mean(), ok[m].mean()))
    return pd.DataFrame(out, columns=["coverage", "precision"])


def at_coverage(conf, ok, target):
    """Precision when the threshold is set to answer `target` of captures."""
    t = np.quantile(conf, 1 - target)
    m = conf >= t
    return m.mean(), ok[m].mean(), m


def main():
    p = load()
    print(f"{len(p)} observations with Pl@ntNet scores\n")

    # --- ours, on the same observations -------------------------------------
    df, _ = build_observations(str(D / "inat_bioclip1_cml4.npz"),
                               variant="bioclip1_cml4", combiner=trimmed)
    m = df.set_index("obs_id").reindex(p.obs_id)
    lv = decide(m.species_conf.values, m.genus_conf.values, 0.4880, 0.5571)
    sp_ok = np.array([correct(a, b, A) for a, b in zip(m.pred_species, p.truth)])
    gn_ok = np.array([correct(a, b, A, genus=True) for a, b in zip(m.pred_species, p.truth)])
    ours_ans = lv != DECLINE
    ours_right = ((lv == SPECIES) & sp_ok) | ((lv == GENUS) & gn_ok)
    ours_cov, ours_prec = ours_ans.mean(), ours_right[ours_ans].mean()
    lo, hi = boot(ours_right[ours_ans].astype(float))
    print(f"ours (three-way, reject class): coverage {ours_cov:.3f}  "
          f"precision {ours_prec:.3f} [{lo:.3f}, {hi:.3f}]\n")

    # --- A: their species score, thresholded to our coverage ----------------
    cov, prec, mask = at_coverage(p.sp_conf.values, p.sp_ok.values, ours_cov)
    lo, hi = boot(p.sp_ok.values[mask].astype(float))
    print(f"A. Pl@ntNet, species threshold at matched coverage:")
    print(f"   coverage {cov:.3f}  precision {prec:.3f} [{lo:.3f}, {hi:.3f}]")

    # --- B: their scores through our cascade --------------------------------
    best = None
    for tg in np.linspace(0, 1, 101):
        for ts in np.linspace(0, 1, 101):
            lvp = decide(p.sp_conf.values, p.gn_conf.values, tg, ts)
            ans = lvp != DECLINE
            if abs(ans.mean() - ours_cov) > 0.02 or ans.sum() < 10:
                continue
            right = ((lvp == SPECIES) & p.sp_ok.values) | ((lvp == GENUS) & p.gn_ok.values)
            pr = right[ans].mean()
            if best is None or pr > best[0]:
                best = (pr, ans.mean(), right, ans)
    pr, cov, right, ans = best
    lo, hi = boot(right[ans].astype(float))
    print(f"\nB. Pl@ntNet through our exact cascade, best thresholds at matched coverage:")
    print(f"   coverage {cov:.3f}  precision {pr:.3f} [{lo:.3f}, {hi:.3f}]")
    print("   (thresholds tuned on this very data — an upper bound for them)")

    print("\n=== precision at matched coverage, per 100 captures ===")
    print(f"   ours              {100*ours_cov*(1-ours_prec):5.1f} wrong")
    print(f"   Pl@ntNet cascade  {100*cov*(1-pr):5.1f} wrong")


if __name__ == "__main__":
    main()
