"""The model card -- the artifact that makes a built model trustworthy.

The weights are the commodity here; this is not. A tool that lets anyone train a
narrow classifier and says nothing about its failure modes is a machine for
producing confident wrong answers at scale, and plant identification has
consequences that make that worse than usually.

So the card states, in order: what it was built from, what it actually scores on
held-out data, the species-level share alongside coverage (without which a
congener-dense set reads as the best case rather than the worst), which relatives
it will confuse, and what it cannot do.
"""

import json
from pathlib import Path

from plantid.tool.encoders import BY_VARIANT


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _ci(metrics, key):
    """Interval for `key`, or an honest dash when there was nothing to resample."""
    iv = (metrics.get("ci") or {}).get(key)
    return "—" if not iv else f"{100 * iv[0]:.1f}–{100 * iv[1]:.1f}%"


def render(manifest: dict) -> str:
    m = manifest["metrics"]
    comp = manifest["composition"]
    enc = BY_VARIANT.get(manifest["encoder"])
    sizing = f"{enc.size_mb():.1f} MB int4" if enc else "size unknown"

    L = [
        f"# Model card — {comp['n_species']} species",
        "",
        f"Built {manifest['created']} · encoder `{manifest['encoder']}` ({sizing}) · "
        f"source `{manifest['source']}`",
        "",
        "## What it answers",
        "",
        f"Measured on held-out data at an assumed **{_pct(m['p_ood'])} out-of-list "
        f"rate** — the share of photographs you take that are of something not on "
        f"your list. That assumption is the single biggest lever on these numbers; "
        f"rebuild with `--ood-rate` if it is wrong for you.",
        "",
        "| | | 95% CI |",
        "|---|---|---|",
        f"| Coverage — queries it answers | **{_pct(m['coverage'])}** | "
        f"{_ci(m, 'coverage')} |",
        f"| Precision — answers that are correct | **{_pct(m['precision'])}** | "
        f"{_ci(m, 'precision')} |",
        f"| **Species-level share** — in-list observations named to species | "
        f"**{_pct(m['species_share'])}** | {_ci(m, 'species_share')} |",
        f"| Closed-set top-1 — accuracy when the plant is on your list | "
        f"{_pct(m['closed_set_top1'])} | {_ci(m, 'closed_set_top1')} |",
        "",
        f"Intervals are bootstrapped over **species**, not rows, because "
        f"observations of one species are not independent. This model rests on "
        f"{m.get('n_species_clusters', '?')} species in the test half, so they are "
        f"wide — that width is a fact about your list, not a formatting choice.",
        "",
    ]

    share = m.get("species_share")
    if share is not None and share < 0.6:
        L += [
            f"> **Read the species-level share, not the coverage.** This model names a "
            f"species on only {_pct(share)} of in-list observations; the rest are "
            f"answered at genus. Because your list is genus-crowded, a genus answer "
            f"may narrow nothing — \"it is a {next(iter(comp['crowded_genera']), 'genus')}\" "
            f"when most of your list is that genus. Coverage and precision look "
            f"healthy here *because* of those genus answers, not despite them.",
            "",
        ]

    L += ["## Where it declines and where it errs", "", "| bucket | n | answered | correct when answered |",
          "|---|---|---|---|"]
    labels = {"in_catalog": "on your list", "near_ood": "relatives you did not choose",
              "distant_ood": "unrelated plants"}
    for b, v in m.get("per_bucket", {}).items():
        L.append(f"| {labels.get(b, b)} | {v['n']} | {_pct(v['answered'])} | "
                 f"{_pct(v['correct_when_answered'])} |")
    L.append("")

    oc = manifest.get("outside_congeners") or {}
    if oc:
        L += [
            "## Relatives it will confuse",
            "",
            "These species are close relatives of ones on your list but are **not on "
            "it**, so no correct answer exists for them. This is the weakest "
            "rejection case measured.",
            "",
        ]
        for g, rel in list(oc.items())[:12]:
            noun = "relative" if len(rel) == 1 else "relatives"
            L.append(f"- **{g}** — {len(rel)} {noun} not on your list: "
                     f"{', '.join(rel[:8])}" + (" …" if len(rel) > 8 else ""))
        if len(oc) > 12:
            L.append(f"- _(+{len(oc) - 12} more genera)_")
        L.append("")

    L += [
        "## How the decision is made",
        "",
        "Three-way: name the species, name the genus, or decline. Thresholds were "
        "fitted by maximising expected utility on a calibration split held out from "
        "these numbers, with payoffs declared before fitting:",
        "",
        "```",
        json.dumps(manifest["utility"], indent=2),
        "```",
        "",
        f"Fitted thresholds: `t_genus={m['t_genus']:.4f}`, `t_species={m['t_species']:.4f}`. "
        f"Calibrated on {m['n_calib']} observations, reported on {m['n_test']}.",
        "",
        "## What it cannot do",
        "",
        f"- It knows {comp['n_species']} species. Everything else it can only decline "
        f"or get wrong — and the relatives listed above are the ones it will get "
        f"wrong confidently.",
        "- **Do not eat anything on the basis of this model.** A correct-looking answer "
        "is not verification. Toxic species have edible lookalikes and the "
        "within-genus case is the measured weak point.",
        "- Numbers above are held-out but come from the same image source as training. "
        "Photographs taken differently — your phone, your light, your angles — will "
        "score lower.",
    ]
    counts = manifest.get("counts", {})
    if counts.get("missing_organs"):
        L.append(f"- No embeddings were available for: "
                 f"{', '.join(counts['missing_organs'])}. Built from the rest.")
    L += ["", "---", "",
          f"Training rows {counts.get('train', '?')} · evaluation rows "
          f"{sum(v['n'] for v in m.get('per_bucket', {}).values())} · "
          f"bundle format v{manifest['bundle_version']}"]
    return "\n".join(L)


def write(bundle_dir: Path) -> Path:
    manifest = json.loads((Path(bundle_dir) / "manifest.json").read_text())
    out = Path(bundle_dir) / "CARD.md"
    out.write_text(render(manifest))
    return out
