"""Score us against Pl@ntNet on the identical observations it was given.

Two comparisons, and the distinction matters:

- **single photo** — Pl@ntNet saw one image, so our single-photo answer is the
  like-for-like number. This is the honest head-to-head.
- **fused** — our answer over all of an observation's photos. Reported because it
  is what the product actually does, but it is an advantage Pl@ntNet was not
  given, so it is not the comparison.

Names are reconciled through the same alias sets, since 9.5% of this catalogue's
names are a taxonomic generation behind and string equality would otherwise
measure nomenclature rather than identification.
"""

import sys

import numpy as np
import pandas as pd


from plantid.config import DATA_PROCESSED as D  # noqa: E402
from plantid.eval.combiners import mean, trimmed  # noqa: E402
from plantid.eval.headtohead import alias_sets, boot, correct, genus_of  # noqa: E402
from plantid.eval.rejection import build_observations  # noqa: E402

VARIANTS = [("bioclip1_cml4", "ours — Core ML int4 (ships)"),
            ("bioclip1", "ours — BioCLIP v1 fp32"),
            ("bioclip2", "ours — BioCLIP-2 (cannot ship)")]


def first_photo(P, organs, oi):
    """Combiner that ignores every photo but the first — matches what Pl@ntNet saw."""
    return P[0]


def main():
    h2h = pd.read_parquet(D / "headtohead.parquet")
    h2h = h2h[h2h["plantnet_top1"].notna()].copy()
    A = alias_sets()
    print(f"{len(h2h)} observations answered by Pl@ntNet, one per species\n")

    rows = []
    pn_sp = np.array([correct(r.plantnet_top1, r.truth, A) for r in h2h.itertuples()], float)
    pn_gn = np.array([correct(r.plantnet_top1, r.truth, A, genus=True)
                      for r in h2h.itertuples()], float)
    pn_t5 = np.array([any(correct(p, r.truth, A) for p in r.plantnet_top5)
                      for r in h2h.itertuples()], float)
    lo, hi = boot(pn_sp)
    rows.append({"system": "Pl@ntNet (~50k species)", "input": "1 photo",
                 "species": pn_sp.mean(), "lo": lo, "hi": hi,
                 "genus": pn_gn.mean(), "species_top5": pn_t5.mean()})

    keep = set(h2h.obs_id)
    ours = {}
    for variant, label in VARIANTS:
        for combiner, tag in ((first_photo, "1 photo"), (trimmed, "fused")):
            df, _ = build_observations(str(D / f"inat_{variant}.npz"),
                                       variant=variant, combiner=combiner)
            df = df[df.obs_id.isin(keep)]
            m = df.set_index("obs_id").reindex(h2h.obs_id)
            sp = np.array([correct(p, t, A) for p, t in zip(m.pred_species, h2h.truth)], float)
            gn = np.array([correct(p, t, A, genus=True)
                           for p, t in zip(m.pred_species, h2h.truth)], float)
            lo, hi = boot(sp)
            rows.append({"system": f"{label} (490 species)", "input": tag,
                         "species": sp.mean(), "lo": lo, "hi": hi,
                         "genus": gn.mean(), "species_top5": np.nan})
            ours[(variant, tag)] = sp

    t = pd.DataFrame(rows).set_index(["system", "input"])
    print(t.round(4).to_string())

    print("\n=== PAIRED vs Pl@ntNet, same observations, single photo each ===")
    for variant, label in VARIANTS:
        d = ours[(variant, "1 photo")] - pn_sp
        lo, hi = boot(d)
        star = "" if lo <= 0 <= hi else "  *"
        print(f"  {label:34s} {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]{star}")

    print("\n=== where each wins, deployed model vs Pl@ntNet (single photo) ===")
    o = ours[("bioclip1_cml4", "1 photo")]
    print(f"  both right      {int(((o == 1) & (pn_sp == 1)).sum()):4d}")
    print(f"  only ours       {int(((o == 1) & (pn_sp == 0)).sum()):4d}")
    print(f"  only Pl@ntNet   {int(((o == 0) & (pn_sp == 1)).sum()):4d}")
    print(f"  both wrong      {int(((o == 0) & (pn_sp == 0)).sum()):4d}")
    t.to_csv("/private/tmp/claude-501/-Users-jameskelly-Documents-plantid/"
             "f3b6d3aa-5322-4d68-a854-87044e191fb8/scratchpad/h2h_table.csv")


if __name__ == "__main__":
    main()
