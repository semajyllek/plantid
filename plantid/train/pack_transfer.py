"""Bundle exactly what distillation training needs, and nothing else.

`data/processed` is 9.6 GB, most of which the trainer never opens. Training reads
the 48,564 transfer images, the cached teacher embeddings, and two manifests —
**3.9 GB**. The 3.5 GB of iNaturalist photographs are not merely unnecessary,
they are images the student is forbidden to see (`train/distil.py`), so leaving
them out of the bundle makes the leak guard physical rather than procedural.

Evaluation is a separate job and belongs on the machine that already has the
data: embedding 82k images with a ViT-B takes ~11 minutes locally, so the only
thing that needs to come back from the GPU is the checkpoint.

The `plantid` package itself goes in too (960 KB, nothing next to 3.9 GB). There
is no git remote to clone from, and shipping code and data as one artifact means
they cannot drift apart — which matters here because the guarantee that the
student never sees evaluation images lives in the code, not the data.

`config.REPO_ROOT` is derived from the package's own location, so untarring this
anywhere and running from that directory resolves `data/processed` correctly.

Usage:
    python -m plantid.train.pack_transfer --out distil_bundle.tar
"""

import argparse
import tarfile
from pathlib import Path

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.train.distil import TEACHER, build_transfer_set


def code_members(repo_root=None):
    """Every .py under `plantid/`, so the bundle is self-sufficient."""
    from plantid.config import REPO_ROOT

    root = Path(repo_root or REPO_ROOT)
    return sorted(p.relative_to(root) for p in (root / "plantid").rglob("*.py")
                  if "__pycache__" not in p.parts)


def members(cache_dir=DATA_PROCESSED, teacher=TEACHER):
    """Paths relative to `cache_dir`, in the layout the trainer expects."""
    from plantid.features.embed_background import cache_path as bg_path
    from plantid.features.embed_catalog import cache_path as cat_path

    out = ["catalog_index.parquet", "plantnet_background.parquet"]
    for organ in ORGANS:
        for fn in (cat_path, bg_path):
            p = fn(organ, teacher, cache_dir)
            if p.exists():
                out.append(p.name)
    out += list(build_transfer_set(cache_dir)["local_path"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="distil_bundle.tar")
    ap.add_argument("--cache-dir", default=str(DATA_PROCESSED))
    ap.add_argument("--no-code", action="store_true",
                    help="data only; the target already has a matching checkout")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    rel = members(cache_dir)
    total = sum((cache_dir / r).stat().st_size for r in rel if (cache_dir / r).exists())
    print(f"{len(rel):,} files, {total / 1e9:.2f} GB -> {args.out}", flush=True)

    from plantid.config import REPO_ROOT

    with tarfile.open(args.out, "w") as tar:
        if not args.no_code:
            code = code_members()
            for c in code:
                tar.add(REPO_ROOT / c, arcname=str(c))
            print(f"  + {len(code)} source files", flush=True)
        for i, r in enumerate(rel, 1):
            src = cache_dir / r
            if src.exists():
                tar.add(src, arcname=f"data/processed/{r}")
            if i % 10000 == 0:
                print(f"  {i:,}/{len(rel):,}", flush=True)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e9:.2f} GB)")
    print("Self-contained: untar into an empty directory, cd there, and run "
          "notebooks/distil_colab.ipynb — no checkout needed.")


if __name__ == "__main__":
    main()
