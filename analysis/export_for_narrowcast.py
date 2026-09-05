"""Export catalogue embeddings in the format `narrowcast --embeddings` expects.

`plantid/tool/` used to build models in-repo. That job moved to narrowcast, which
deliberately does not know how to read this project's catalogue caches: choosing
a corpus, reconciling its taxonomy and licensing its images are domain decisions,
and narrowcast takes a dataset rather than going to find one.

This is the seam between them. It writes the vectors this project already has
into narrowcast's source format, so the workflow is:

    PYTHONPATH=. .venv/bin/python -m analysis.export_for_narrowcast \\
        --variant bioclip2 --species my.txt --out /tmp/cat.npz

    narrowcast build --embeddings /tmp/cat.npz \\
        --background-embeddings /tmp/bg.npz --out models/mine

**No `cluster` column is written, and that is deliberate.** The catalogue does not
group its images by individual plant, so there is no honest cluster to declare.
narrowcast will say so on the card and report that its intervals are
anticonservative -- which is true, and better than inventing a grouping.
"""

import argparse
from pathlib import Path

import numpy as np

from plantid.config import DATA_PROCESSED
from plantid.data.curation import canonical_name
from plantid.features.embed_background import catalog_species, load_background

ORGANS = ("leaf", "flower")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="bioclip2")
    ap.add_argument("--species", help="file of binomials; omit for the whole catalogue")
    ap.add_argument("--out", required=True)
    ap.add_argument("--background", action="store_true",
                    help="export the reject pool instead of the catalogue")
    a = ap.parse_args()

    vecs, labels, keep = [], [], None
    if a.background:
        cs = catalog_species()
        for organ in ORGANS:
            d = load_background(organ, exclude_species=cs, variant=a.variant)
            vecs.append(d["descriptor"])
            labels += ["__OTHER__"] * len(d["descriptor"])
    else:
        if a.species:
            keep = {canonical_name(x) for x in Path(a.species).read_text().split("\n")
                    if x.strip() and not x.strip().startswith("#")}
        for organ in ORGANS:
            p = Path(DATA_PROCESSED) / f"catalog_{organ}_{a.variant}.npz"
            if not p.exists():
                print(f"  skipping {organ}: no cache for {a.variant}")
                continue
            d = np.load(p, allow_pickle=True)
            names = np.array([canonical_name(n) for n in d["species_name"].astype(str)])
            m = np.ones(len(names), bool) if keep is None else np.isin(names, list(keep))
            vecs.append(d["descriptor"][m])
            labels += list(names[m])

    if not vecs:
        raise SystemExit(f"nothing exported — no caches for variant {a.variant!r}")
    if keep is not None:
        missing = sorted(keep - set(labels))
        if missing:
            # A silently dropped label is a model that cannot see a class the
            # user asked for. Say so loudly rather than exporting a short file.
            print(f"warning: {len(missing)} requested label(s) absent from the "
                  f"{a.variant} catalogue and NOT exported: {', '.join(missing[:8])}"
                  + (" ..." if len(missing) > 8 else ""))
    X = np.vstack(vecs)
    np.savez_compressed(a.out, descriptor=X, label=np.asarray(labels, dtype=str))
    print(f"{a.out}: {X.shape}, {len(set(labels))} labels")


if __name__ == "__main__":
    main()
