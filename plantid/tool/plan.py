"""What a species list will give you, before any compute is spent.

This is the command that earns the tool its keep, because the two things a user
most needs to know are things they cannot see by looking at their own list:

1. **How often it will answer at species level.** A congener-dense set answers
   with a genus that narrows nothing -- "it is a Sedum" on a list of six Sedums.
   Its coverage and precision look *better* than a well-separated set while being
   worse in every way the user cares about, so the species-level share is
   reported alongside them and never omitted.

2. **Which species it will confidently get wrong.** Relatives left outside the
   set have no correct label available, and near-OOD is the weakest rejection
   bucket. Naming them beats any aggregate.

Everything here is interpolated from a measured grid, not fitted. `build`
measures the user's actual data and the card supersedes these numbers.
"""

from dataclasses import dataclass

from plantid.tool import encoders, projection, species as sp


@dataclass
class Warning_:
    kind: str
    headline: str
    detail: str


def _fmt_pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def make_plan(chosen: list[str], budget_mb: float | None = None,
              p_ood: float = 0.2, pool: list[str] | None = None,
              encoder: str | None = None) -> dict:
    comp = sp.analyse(chosen, pool=pool)
    enc = encoders.BY_VARIANT[encoder] if encoder else encoders.choose(budget_mb)

    warnings = []

    # Availability first, and it can suppress the projection entirely. The local
    # catalogue holds ~499 binomials, so most real lists are partly or wholly
    # absent -- projecting "coverage 79%, species-level 84%" for a list `build`
    # will then refuse to fit is the worst thing this command could do.
    pool_set = set(comp["_pool"])
    missing = [s for s in comp["species"] if s not in pool_set] if pool_set else []
    available = comp["n_species"] - len(missing)
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        warnings.append(Warning_(
            "missing",
            f"{len(missing)} of your {comp['n_species']} species have no images in "
            f"the local catalogue",
            f"`build` cannot fit these and will drop them:\n      {shown}\n"
            f"      Fetching arbitrary species from iNaturalist is not wired up yet, "
            f"so a list is currently limited to what the catalogue covers.",
        ))

    if pool_set and available < max(2, comp["n_species"] / 2):
        warnings.append(Warning_(
            "unprojectable",
            f"only {available} of {comp['n_species']} species are available — "
            f"not projecting",
            "Too little of this list can be built for a projection to mean anything.",
        ))
        return {"composition": comp, "encoder": enc, "projection": None,
                "warnings": warnings, "p_ood": p_ood, "budget_mb": budget_mb,
                "missing": missing, "n_available": available,
                "budget_note": encoders.budget_note(budget_mb)}

    proj = projection.project(enc.variant, max(available, 1),
                              comp["in_set_congener_frac"], p_ood=p_ood)

    crowded = comp["crowded_genera"]
    if crowded:
        worst = ", ".join(f"{g} ({n})" for g, n in list(crowded.items())[:4])
        warnings.append(Warning_(
            "crowded",
            f"{sum(crowded.values())} of your {comp['n_species']} species share a "
            f"genus with another: {worst}",
            f"Projected species-level answers: {_fmt_pct(proj['species_share'])} of "
            f"in-catalogue observations. The rest resolve to genus, which on a "
            f"genus-crowded list may not narrow anything. Coverage and precision "
            f"will look good regardless -- read the species share instead.",
        ))

    outside = comp["outside_congeners"]
    if outside:
        shown = list(outside.items())[:6]
        lines = "\n".join(
            f"      {g}: {len(r)} not in your set"
            f"  ({', '.join(r[:3])}{' ...' if len(r) > 3 else ''})"
            for g, r in shown
        )
        more = f"\n      (+{len(outside) - len(shown)} more genera)" if len(outside) > len(shown) else ""
        warnings.append(Warning_(
            "outside_congeners",
            f"{comp['n_species_exposed']} of your species are in genera with relatives "
            f"NOT in your set",
            f"These have no correct label available and are the weakest rejection "
            f"case; expect confident mislabelling.\n{lines}{more}",
        ))

    if comp["pool_size"] == 0:
        warnings.append(Warning_(
            "no_pool",
            "no reference pool available, so relatives outside your set were not checked",
            "The local catalogue index was not found. Relatives are the main "
            "failure mode for a narrow catalogue, so treat every number below as "
            "unverified until it is built.",
        ))
    elif comp["pool_size"] < 1000:
        warnings.append(Warning_(
            "shallow_pool",
            f"relatives were checked against only {comp['pool_size']} species",
            "That is a floor, not a census: a genus whose relatives the reference "
            "pool never included will look safer than it is.",
        ))

    if proj["extrapolated"]:
        warnings.append(Warning_(
            "extrapolated",
            f"{comp['n_species']} species is outside the measured range (10-50)",
            f"Projections are clamped to K={proj['K_used']} rather than "
            f"extrapolated, so they carry even more error than usual here.",
        ))

    return {"composition": comp, "encoder": enc, "projection": proj,
            "warnings": warnings, "p_ood": p_ood, "budget_mb": budget_mb,
            "missing": missing, "n_available": available,
            "budget_note": encoders.budget_note(budget_mb)}


def render(plan: dict) -> str:
    c, e, p = plan["composition"], plan["encoder"], plan["projection"]
    L = []
    avail = plan.get("n_available", c["n_species"])
    got = "" if avail == c["n_species"] else f", {avail} with images"
    L.append(f"  {c['n_species']} species, {c['n_genera']} genera{got}"
             f"   (reference pool: {c['pool_size']} species)")
    L.append("")

    for w in plan["warnings"]:
        L.append(f"  [!] {w.headline}")
        for line in w.detail.splitlines():
            L.append(f"      {line}" if not line.startswith("      ") else line)
        L.append("")

    L.append(f"  Encoder: {e.label}, {e.size_mb():.1f} MB int4")
    L.append(f"           {plan['budget_note']}")
    L.append("")
    if p is None:
        L.append("  No projection: too little of this list is available to build.")
        return "\n".join(L)
    L.append(f"  Projected, assuming {_fmt_pct(plan['p_ood'])} of what you photograph "
             f"is outside your list:")
    L.append(f"      coverage             {_fmt_pct(p['coverage'])}  of queries answered")
    L.append(f"      precision            {_fmt_pct(p['precision'])}  of answers correct")
    L.append(f"      species-level share  {_fmt_pct(p['species_share'])}  of in-list "
             f"observations named to species")
    L.append(f"      closed-set top-1     {_fmt_pct(p['top1'])}  when the plant IS on "
             f"your list and it answers")
    L.append("")
    L.append("  Projected from measurements on a different catalogue -- indicative, not")
    L.append("  a guarantee. `plantid build` measures your actual data.")
    return "\n".join(L)
