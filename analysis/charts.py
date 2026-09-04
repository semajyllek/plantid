"""SVG charts for the experimental-record page, computed from figures.json.

Coordinates are derived rather than hand-authored so a number cannot drift
between the chart and the table beside it. Every mark carries a <title> for a
native accessible tooltip, and every chart ships with a table — which is also
the relief the palette validator requires for the aqua series in light mode.
"""

import json

W, H = 720, 300
PAD = {"l": 56, "r": 20, "t": 18, "b": 40}
S1, S2, S3 = "var(--s1)", "var(--s2)", "var(--s3)"


def _x(v, lo, hi):
    return PAD["l"] + (v - lo) / (hi - lo) * (W - PAD["l"] - PAD["r"])


def _y(v, lo, hi):
    return H - PAD["b"] - (v - lo) / (hi - lo) * (H - PAD["t"] - PAD["b"])


def _frame(xlab, ylab, xticks, yticks, xlo, xhi, ylo, yhi, h=H):
    g = [f'<line class="ax" x1="{PAD["l"]}" y1="{h-PAD["b"]}" x2="{W-PAD["r"]}" y2="{h-PAD["b"]}"/>']
    for t in yticks:
        y = h - PAD["b"] - (t - ylo) / (yhi - ylo) * (h - PAD["t"] - PAD["b"])
        g.append(f'<line class="grid" x1="{PAD["l"]}" y1="{y:.1f}" x2="{W-PAD["r"]}" y2="{y:.1f}"/>')
        g.append(f'<text class="tick" x="{PAD["l"]-9}" y="{y+4:.1f}" text-anchor="end">{t:g}</text>')
    for t in xticks:
        x = _x(t, xlo, xhi)
        g.append(f'<text class="tick" x="{x:.1f}" y="{h-PAD["b"]+20}" text-anchor="middle">{t:g}</text>')
    g.append(f'<text class="axlab" x="{(W+PAD["l"])/2:.0f}" y="{h-4}" text-anchor="middle">{xlab}</text>')
    g.append(f'<text class="axlab" transform="rotate(-90 14 {h/2:.0f})" x="14" y="{h/2:.0f}" '
             f'text-anchor="middle">{ylab}</text>')
    return "".join(g)


def _legend(items, x, y, gap=17):
    """Swatch + label rows. Series labels placed inline over a line chart collide
    with its own point labels, so identity lives in dead space instead."""
    out = []
    for i, (col, label) in enumerate(items):
        yy = y + i * gap
        out.append(f'<rect x="{x}" y="{yy-8}" width="10" height="10" rx="2.5" fill="{col}"/>')
        out.append(f'<text class="lgd" x="{x+15}" y="{yy+1}" fill="{col}">{label}</text>')
    return "".join(out)


def precision_coverage(D):
    """Parametric: each point is an assumed out-of-catalogue rate."""
    xlo, xhi, ylo, yhi = 0.25, 0.85, 0.80, 1.0
    out = [_frame("coverage — share of captures answered", "precision",
                  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.8, 0.85, 0.9, 0.95, 1.0], xlo, xhi, ylo, yhi)]
    for key, col, label in (("bioclip2", S1, "BioCLIP-2 · cannot ship"),
                            ("bioclip1", S2, "BioCLIP v1 · ships")):
        pts = D[key]["prevalence"]["regional"]
        xy = [(_x(c, xlo, xhi), _y(p, ylo, yhi)) for _, p, c in pts]
        out.append(f'<polyline class="ln" stroke="{col}" points="'
                   + " ".join(f"{x:.1f},{y:.1f}" for x, y in xy) + '"/>')
        for (x, y), (rate, p, c) in zip(xy, pts):
            out.append(f'<circle class="mk" cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}">'
                       f'<title>{int(rate*100)}% out-of-catalogue — precision {p:.3f}, '
                       f'coverage {c:.3f}</title></circle>')
            out.append(f'<text class="pt" x="{x:.1f}" y="{y-11:.1f}" text-anchor="middle">'
                       f'{int(rate*100)}%</text>')
    out.append(_legend([(S1, "BioCLIP-2 · cannot ship"), (S2, "BioCLIP v1 · ships")],
                       PAD["l"] + 14, PAD["t"] + 12))
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="precision against coverage">' \
           + "".join(out) + "</svg>"


def outcomes(D, key="bioclip2"):
    """What the rule does to each bucket. Stacked, 2px surface gaps."""
    rows = [("in-catalogue", "in_catalog"), ("near-OOD — genus in catalogue", "near_ood"),
            ("global OOD", "distant_ood"), ("regional OOD", "regional_ood")]
    h, bar, gap = 40, 22, 18
    height = PAD["t"] + len(rows) * (bar + gap) + 34
    L, RW = 210, W - 210 - PAD["r"]
    out = []
    for i, (label, k) in enumerate(rows):
        b = D[key]["buckets"][k]
        y = PAD["t"] + i * (bar + gap)
        out.append(f'<text class="rowlab" x="{L-12}" y="{y+15}" text-anchor="end">{label}</text>')
        out.append(f'<text class="rowN" x="{L-12}" y="{y+15}" text-anchor="end" dx="0" dy="14">'
                   f'n={b["n"]}</text>')
        x = L
        for val, col, name in ((b["species"], S1, "species"), (b["genus"], S3, "genus"),
                               (b["decline"], "var(--mute)", "declined")):
            w = val * RW
            if w > 0.5:
                out.append(f'<rect class="seg" x="{x:.1f}" y="{y}" width="{max(w-2,1):.1f}" '
                           f'height="{bar}" rx="3" fill="{col}">'
                           f'<title>{label}: {name} {val*100:.1f}%</title></rect>')
                if w > 46:
                    out.append(f'<text class="inbar" x="{x+w/2-1:.1f}" y="{y+15:.1f}" '
                               f'text-anchor="middle">{val*100:.0f}%</text>')
            x += w
    ly = PAD["t"] + len(rows) * (bar + gap) + 12
    for i, (col, name) in enumerate(((S1, "species"), (S3, "genus only"), ("var(--mute)", "declined"))):
        lx = L + i * 118
        out.append(f'<rect x="{lx}" y="{ly}" width="11" height="11" rx="2.5" fill="{col}"/>')
        out.append(f'<text class="tick" x="{lx+17}" y="{ly+10}">{name}</text>')
    return f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="outcome by bucket">' \
           + "".join(out) + "</svg>"


def cohorts(D):
    """Species accuracy with cluster-bootstrap intervals — uncertainty as the mark."""
    rows = D["cohorts"] + [dict(D["combined"], name="all in-catalogue")]
    xlo, xhi = 0.60, 1.0
    h = PAD["t"] + len(rows) * 46 + 34
    L = 190
    out = [f'<line class="ax" x1="{L}" y1="{h-PAD["b"]}" x2="{W-PAD["r"]}" y2="{h-PAD["b"]}"/>']
    for t in (0.6, 0.7, 0.8, 0.9, 1.0):
        x = L + (t - xlo) / (xhi - xlo) * (W - L - PAD["r"])
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{PAD["t"]}" x2="{x:.1f}" y2="{h-PAD["b"]}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{h-PAD["b"]+20}" text-anchor="middle">{t:g}</text>')
    for i, r in enumerate(rows):
        y = PAD["t"] + i * 46 + 14
        last = i == len(rows) - 1
        col = "var(--ink)" if last else S1
        px = lambda v: L + (v - xlo) / (xhi - xlo) * (W - L - PAD["r"])  # noqa: E731
        out.append(f'<text class="rowlab" x="{L-12}" y="{y+4}" text-anchor="end">{r["name"]}</text>')
        out.append(f'<text class="rowN" x="{L-12}" y="{y+18}" text-anchor="end">'
                   f'{r["species"]} spp · {r["obs"]} obs</text>')
        out.append(f'<line class="civ" x1="{px(r["sp_lo"]):.1f}" y1="{y}" x2="{px(r["sp_hi"]):.1f}" '
                   f'y2="{y}" stroke="{col}"/>')
        for e in ("sp_lo", "sp_hi"):
            out.append(f'<line class="cap" x1="{px(r[e]):.1f}" y1="{y-5}" x2="{px(r[e]):.1f}" '
                       f'y2="{y+5}" stroke="{col}"/>')
        out.append(f'<circle class="mk" cx="{px(r["sp"]):.1f}" cy="{y}" r="5.5" fill="{col}">'
                   f'<title>{r["name"]}: species accuracy {r["sp"]:.3f} '
                   f'[{r["sp_lo"]:.3f}, {r["sp_hi"]:.3f}]</title></circle>')
        out.append(f'<text class="pt" x="{px(r["sp"]):.1f}" y="{y-12:.1f}" text-anchor="middle">'
                   f'{r["sp"]:.3f}</text>')
    out.append(f'<text class="axlab" x="{(W+L)/2:.0f}" y="{h-4}" text-anchor="middle">'
               f'species accuracy, 95% cluster-bootstrap interval</text>')
    return f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="species accuracy by cohort">' \
           + "".join(out) + "</svg>"


def congeners(D):
    """The anti-correlation: species falls as genus rises."""
    rows = D["congeners"]
    xlo, xhi, ylo, yhi = 0, len(rows) - 1, 0.75, 1.0
    out = [_frame("catalogue congeners the species has", "accuracy",
                  [], [0.8, 0.85, 0.9, 0.95, 1.0], xlo, xhi, ylo, yhi)]
    for i, r in enumerate(rows):
        out.append(f'<text class="tick" x="{_x(i, xlo, xhi):.1f}" y="{H-PAD["b"]+20}" '
                   f'text-anchor="middle">{r["bin"]}</text>')
    for field, col, label in (("sp", S1, "species"), ("gn", S3, "genus")):
        xy = [(_x(i, xlo, xhi), _y(r[field], ylo, yhi)) for i, r in enumerate(rows)]
        out.append(f'<polyline class="ln" stroke="{col}" points="'
                   + " ".join(f"{x:.1f},{y:.1f}" for x, y in xy) + '"/>')
        for (x, y), r in zip(xy, rows):
            out.append(f'<circle class="mk" cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}">'
                       f'<title>{r["bin"]} congeners — {label} accuracy {r[field]:.3f} '
                       f'({r["obs"]} obs)</title></circle>')
    out.append(_legend([(S1, "species accuracy"), (S3, "genus accuracy")],
                       W - 190, PAD["t"] + 12))
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="accuracy against congener count">' \
           + "".join(out) + "</svg>"


def ablation(D):
    """The flat line: reject-class size does not bind."""
    rows = {}
    for r in D["pool_ablation"]:
        rows.setdefault(r["flower_bg_species"], []).append(r)
    xs = sorted(rows)
    mean = {k: {f: sum(r[f] for r in v) / len(v) for f in
                ("auroc_global", "auroc_regional", "auroc_near")} for k, v in rows.items()}
    xlo, xhi, ylo, yhi = 0, len(xs) - 1, 0.70, 1.0
    out = [_frame("species in the reject class", "AUROC on genus confidence",
                  [], [0.7, 0.8, 0.9, 1.0], xlo, xhi, ylo, yhi)]
    for i, k in enumerate(xs):
        out.append(f'<text class="tick" x="{_x(i, xlo, xhi):.1f}" y="{H-PAD["b"]+20}" '
                   f'text-anchor="middle">{int(k)}</text>')
    for field, col, label in (("auroc_regional", S1, "regional OOD"),
                              ("auroc_global", S3, "global OOD"),
                              ("auroc_near", S2, "near OOD")):
        xy = [(_x(i, xlo, xhi), _y(mean[k][field], ylo, yhi)) for i, k in enumerate(xs)]
        out.append(f'<polyline class="ln" stroke="{col}" points="'
                   + " ".join(f"{x:.1f},{y:.1f}" for x, y in xy) + '"/>')
        for (x, y), k in zip(xy, xs):
            out.append(f'<circle class="mk" cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{col}">'
                       f'<title>{int(k)} species — {label} AUROC {mean[k][field]:.4f}</title></circle>')
    out.append(_legend([(S1, "regional OOD"), (S3, "global OOD"), (S2, "near OOD")],
                       W - 150, PAD["t"] + 78))
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="rejection AUROC against pool size">' \
           + "".join(out) + "</svg>"


if __name__ == "__main__":
    import sys
    D = json.load(open(sys.argv[1]))
    json.dump({"precision_coverage": precision_coverage(D), "outcomes": outcomes(D),
               "cohorts": cohorts(D), "congeners": congeners(D), "ablation": ablation(D)},
              open(sys.argv[2], "w"))
    print("wrote", sys.argv[2])
