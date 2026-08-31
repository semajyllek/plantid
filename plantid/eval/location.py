"""Does knowing where the user is standing improve the answer?

Two experiments, and a product decision taken before running them: **location
gates, it never renames.** The prior may make the app decline, or drop from a
species answer to a genus answer, but the species named is decided by the
photograph alone — an app should not tell someone it is a different plant
because of where they are standing.

That makes Test B the deliverable and Test A a bounded diagnostic:

- **Test A** — is there identity information in location at all? Reported as
  headroom (how many species errors *could* location fix) and, as a measurement
  only, what multiplicative re-ranking would buy. Not shipped either way.
- **Test B** — does a location score improve the decline gate, especially for
  near-OOD plants, which are the weakest case and the one place a signal
  orthogonal to the embedding could help?

Everything is fitted on the calibration split and reported on test, with
cluster-bootstrap intervals. The pre-registered rule: ship the gate only if
expected utility improves with a 95% CI excluding zero. Better AUROC alone does
not count — a gate that separates well by refusing more genuine catalogue
plants is not a win.

Usage:
    PYTHONPATH=. python -m plantid.eval.location
"""

import argparse

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED
from plantid.data.species_ranges import eval_obs_ids, load_prior
from plantid.eval.rejection import (
    DECLINE,
    GENUS,
    SPECIES,
    build_observations,
    cluster_bootstrap,
    decide,
    fit_thresholds,
    utility,
)

BANDWIDTH_KM = 250.0


def attach_locations(df, cache_dir=DATA_PROCESSED):
    """Join coordinates onto the scored observation frame, matched by obs_id.

    `build_observations` does not carry obs_id, so the join is by
    (bucket, species, order), which is stable because both frames iterate the
    same manifest in the same order.
    """
    man = pd.read_parquet(cache_dir / "inat_observations.parquet")
    man = man[man["local_paths"].map(len) >= 2].reset_index(drop=True)
    man["binom"] = man["species_name"].map(lambda n: " ".join(str(n).split()[:2]))
    keep = ["obs_id", "lat", "lon", "obscured"]
    return df.reset_index(drop=True).join(man[keep])


def location_scores(df, prior, neutral=True):
    """Per-observation location features. Gating signals only, never re-ranking.

    **`loc_prq` is the admissible score**: where the named species ranks among
    the whole catalogue *at this location*, in [0,1]. It is a within-location
    statistic, and that is essential.

    **`loc_mass` is deliberately included as a control, and must not be used.**
    Absolute prior mass is a pure function of the coordinates — how
    well-recorded the area is — so on this evaluation set it "separates" the
    buckets only because they were sampled from different parts of the world.
    In deployment a catalogue plant and an unknown one arrive from the same user
    at the same coordinates, where a location-only score has *zero*
    discriminative power by construction. It is reported to demonstrate the
    artefact, not to be shipped.

    Species absent from the range data (34 of 248, almost all *Anemone*
    reclassified by iNaturalist into other genera) get a neutral 0.5 rank rather
    than the floor: absence of evidence is not evidence of absence, and
    penalising 14% of the catalogue everywhere is a bias, not a signal.
    """
    prq, mass = [], []
    known = set(prior.species)
    for r in df.itertuples():
        lat = getattr(r, "lat", np.nan)
        if not np.isfinite(lat):
            prq.append(np.nan); mass.append(np.nan)
            continue
        if r.pred_species in known:
            q = prior.rank_quantile(lat, r.lon, r.pred_species)
        else:
            q = 0.5 if neutral else 0.0
        prq.append(q)
        mass.append(float(prior.weights(lat, r.lon).sum()))
    out = df.copy()
    out["loc_prq"], out["loc_mass"] = prq, mass
    return out


# ---------------------------------------------------------------- test A ----

def headroom(df, prior, min_weight=1.0):
    """Upper bound on what location could fix: species errors where the named
    species is locally implausible but the truth is locally plausible."""
    ic = df[(df.bucket == "in_catalog") & df.lat.notna()]
    wrong = ic[~ic.species_ok]
    fixable = 0
    for r in wrong.itertuples():
        near = prior.plausible(r.lat, r.lon, min_weight=min_weight)
        if r.pred_species not in near and r.species in near:
            fixable += 1
    return len(ic), len(wrong), fixable


# ---------------------------------------------------------------- test B ----

def decide_with_location(species_conf, genus_conf, loc, t_genus, t_species, t_loc):
    """The existing cascade plus a location gate. Gating only: the label the
    photograph chose is never replaced, only withheld or generalised."""
    levels = decide(species_conf, genus_conf, t_genus, t_species)
    weak = np.asarray(loc) < t_loc
    # a locally implausible plant is first demoted to genus, then declined
    levels = np.where(weak & (levels == SPECIES), GENUS, levels)
    levels = np.where(weak & (levels == GENUS) & (np.asarray(genus_conf) < t_genus * 1.05),
                      DECLINE, levels)
    return levels.astype(object)


def fit_joint_thresholds(cal, score="loc_prq", n_grid=26):
    """Fit (t_genus, t_species, t_loc) together on calibration.

    Fitting t_loc while holding the other two at their location-free optimum
    understates the gate: those thresholds were chosen for a cascade that had no
    location signal, so they leave it almost nothing to do. Fitting jointly lets
    the vision thresholds relax where location can cover for them, which is the
    only way the extra signal can actually pay.
    """
    sp, gn = cal.species_conf.to_numpy(), cal.genus_conf.to_numpy()
    loc = np.nan_to_num(cal[score].to_numpy(float), nan=np.inf)
    sok, gok, inc = cal.species_ok.values, cal.genus_ok.values, cal.in_catalog.values
    q = np.linspace(0, 1, n_grid)
    g_grid, s_grid = np.quantile(gn, q), np.quantile(sp, q)
    l_grid = np.r_[0.0, np.unique(np.quantile(loc[np.isfinite(loc)], np.linspace(0, 0.9, n_grid)))]

    best, best_u = (0.0, 0.0, 0.0), -np.inf
    for tg in g_grid:
        for ts in s_grid:
            for tl in l_grid:
                u = utility(decide_with_location(sp, gn, loc, tg, ts, tl),
                            sok, gok, inc).mean()
                if u > best_u:
                    best, best_u = (float(tg), float(ts), float(tl)), float(u)
    return best, best_u


def clusters_for(df):
    return np.array([r.species if r.bucket != "near_ood" else r.genus for r in df.itertuples()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(DATA_PROCESSED / "inat_bioclip2.npz"))
    ap.add_argument("--bandwidth", type=float, default=BANDWIDTH_KM)
    ap.add_argument("--score", default="loc_prq", choices=["loc_prq", "loc_mass"])
    args = ap.parse_args()

    df, _ = build_observations(args.emb)
    df = attach_locations(df)
    prior = load_prior(exclude_obs_ids=eval_obs_ids(), bandwidth_km=args.bandwidth)
    df = location_scores(df, prior)
    print(f"{len(df)} observations, {df.lat.notna().sum()} with coordinates, "
          f"bandwidth {args.bandwidth:.0f} km, {len(prior.species)} species in the prior\n")

    cal, test = df[df.fold == "calib"], df[df.fold == "test"]

    # ---- Test A: headroom -------------------------------------------------
    n_ic, n_wrong, fixable = headroom(test, prior)
    print("TEST A — is there identity information in location?")
    print(f"  in-catalogue test observations: {n_ic}, species errors: {n_wrong}")
    print(f"  errors where the named species is locally implausible but the truth is plausible: "
          f"{fixable}  ({fixable / max(n_wrong, 1):.0%} of errors, "
          f"{fixable / max(n_ic, 1):.1%} of all)")
    print("  (this is the ceiling on re-ranking, which we are choosing not to do)\n")

    # ---- Test B: the gate -------------------------------------------------
    (tg0, ts0), _ = fit_thresholds(cal.species_conf.values, cal.genus_conf.values,
                                   cal.species_ok.values, cal.genus_ok.values,
                                   cal.in_catalog.values)
    (tg, ts, t_loc), _ = fit_joint_thresholds(cal, score=args.score)
    print(f"TEST B — does a location gate improve the decision? (score={args.score})")
    print(f"  baseline thresholds (no location): t_genus={tg0:.3f} t_species={ts0:.3f}")
    print(f"  joint fit with location:           t_genus={tg:.3f} t_species={ts:.3f} t_loc={t_loc:.3f}")

    base = decide(test.species_conf.values, test.genus_conf.values, tg0, ts0)
    gated = decide_with_location(test.species_conf.values, test.genus_conf.values,
                                 np.nan_to_num(test[args.score].to_numpy(float), nan=np.inf),
                                 tg, ts, t_loc)
    u_base = utility(base, test.species_ok.values, test.genus_ok.values, test.in_catalog.values)
    u_gate = utility(gated, test.species_ok.values, test.genus_ok.values, test.in_catalog.values)
    cl = clusters_for(test)
    lo, hi = cluster_bootstrap(u_gate - u_base, cl)
    verdict = "SHIP" if lo > 0 else "DO NOT SHIP"
    print(f"  utility  base {u_base.mean():+.4f}   gated {u_gate.mean():+.4f}")
    print(f"  paired gain {np.mean(u_gate - u_base):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}\n")

    print("  what the gate changed, per bucket:")
    ch = test.assign(base=base, gated=gated)
    for b, g in ch.groupby("bucket"):
        moved = (g.base != g.gated).mean()
        dec_b, dec_g = (g.base == DECLINE).mean(), (g.gated == DECLINE).mean()
        print(f"    {b:13s} n={len(g):4d}  changed {moved:5.1%}  decline {dec_b:.3f} -> {dec_g:.3f}")

    # ---- controls ---------------------------------------------------------
    from sklearn.metrics import roc_auc_score
    inc = test.in_catalog.values
    print("\n  CONTROL 1 — is this geography or biology? shuffle coordinates within each")
    print("  bucket, which preserves where each bucket was sampled but destroys the")
    print("  species-location pairing. A real signal must collapse.")
    rng = np.random.RandomState(0)
    for label, col in (("real", args.score), ("shuffled", "__perm__")):
        if col == "__perm__":
            perm = test.copy()
            for b, g in perm.groupby("bucket"):
                idx = g.index.to_numpy().copy(); shuf = idx.copy(); rng.shuffle(shuf)
                perm.loc[idx, ["lat", "lon"]] = perm.loc[shuf, ["lat", "lon"]].values
            scored = location_scores(perm, prior)[args.score]
        else:
            scored = test[args.score]
        m = scored.notna().values
        aur = []
        for b in ("near_ood", "regional_ood", "distant_ood"):
            sel = m & (inc | (test.bucket == b).values)
            aur.append(roc_auc_score(inc[sel].astype(int), scored.values[sel]))
        print(f"    {label:9s} near-OOD {aur[0]:.3f}  regional {aur[1]:.3f}  distant {aur[2]:.3f}")

    print("\n  CONTROL 2 — the inadmissible score, for comparison. Absolute prior mass")
    print("  depends only on the coordinates, so it cannot work in deployment where")
    print("  known and unknown plants arrive from the same place:")
    mm = test.loc_mass.notna().values
    for b in ("near_ood", "regional_ood", "distant_ood"):
        sel = mm & (inc | (test.bucket == b).values)
        print(f"    loc_mass {b:13s} AUROC {roc_auc_score(inc[sel].astype(int), test.loc_mass.values[sel]):.3f}")

    print("\n  does the score add to the vision chain? AUROC per bucket:")
    for b in ("near_ood", "regional_ood", "distant_ood"):
        sel = (inc | (test.bucket == b).values) & test[args.score].notna().values
        y = inc[sel].astype(int)
        g_only = roc_auc_score(y, test.genus_conf.values[sel])
        combo = roc_auc_score(y, test.genus_conf.values[sel] * test[args.score].values[sel])
        print(f"    {b:13s} genus_conf {g_only:.3f} -> with location {combo:.3f}  ({combo - g_only:+.3f})")

    print("\n  orthogonality — correlation of the location score with genus confidence:")
    m = test[args.score].notna()
    print(f"    Pearson r = {np.corrcoef(test.loc[m, args.score], test.loc[m, 'genus_conf'])[0, 1]:+.3f}"
          "   (near 0 = genuinely new information)")

    ob = test[test.obscured.fillna(False)]
    print(f"\n  sensitivity: {len(ob)} observations have iNat-fuzzed coordinates; "
          f"excluding them, gain = "
          f"{np.mean((u_gate - u_base)[~test.obscured.fillna(False).values]):+.4f}")


if __name__ == "__main__":
    main()
