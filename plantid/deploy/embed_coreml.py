"""Embed the corpora through the *shipped* Core ML artifact.

Every accuracy figure in this repo comes from PyTorch embeddings. The model that
would actually run on a phone is a Core ML int4 export, and cosine similarity
between the two says less than it appears to: the 4-bit simulation in
`ONDEVICE_FINDINGS.md` moved embeddings to cosine 0.898 and destroyed 38% of
nearest-neighbour relations while costing barely a point of accuracy, because a
refitted head absorbs a systematic shift. Similar is not the same as separable,
in either direction. The only way to know what the shipped model scores is to run
the head on its output.

This writes the same `.npz` caches the PyTorch path writes, under a distinct
variant string, so `build_heads` / `build_router` / `build_observations` consume
them unchanged.

**Compute units are pinned to CPU_AND_NE, and that is not a performance choice.**
The int4 per-grouped-channel model — the configuration that fits the size budget
— is silently wrong on the Metal GPU backend:

    int4 per-grouped-channel, vs PyTorch fp32:  ANE 0.935 · CPU 0.935 · GPU 0.204
    int4 per-tensor:                            ANE 0.422 · GPU 0.422
    fp16:                                       ANE 0.9998 · GPU 1.0000

Only per-grouped-channel breaks, and only on GPU, and it returns a plausible
unit-norm embedding while doing so. Anything that lets Core ML choose the GPU for
this model gets garbage with no error — which applies to the iOS app as much as
to this script.

Usage:
    PYTHONPATH=. .venv-mps/bin/python -m plantid.deploy.embed_coreml \
        --model data/processed/coreml/bioclip1_int4_per_grouped_channel.mlpackage \
        --variant bioclip1_cml4
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.deploy.coreml import _pil_batch

CHUNK = 64
WORKERS = 8


def load_model(path, compute_units=None):
    """The artifact under test. Defaults to CPU_AND_NE — see module docstring."""
    import coremltools as ct

    return ct.models.MLModel(
        str(path), compute_units=compute_units or ct.ComputeUnit.CPU_AND_NE)


def embed_paths(model, paths, desc="", chunk=CHUNK, workers=WORKERS):
    """(n, dim) float32. Decode is threaded; `predict` is one image at a time.

    The export has a fixed (1,3,224,224) input and stays that way: re-exporting
    at another batch size would no longer be the artifact whose accuracy is being
    measured.
    """
    from tqdm import tqdm

    out = []
    with ThreadPoolExecutor(workers) as pool:
        for i in tqdm(range(0, len(paths), chunk), desc=desc):
            batch = paths[i:i + chunk]
            images = list(pool.map(lambda p: _pil_batch([p])[0], batch))
            out.append(np.stack([model.predict({"image": im})["embedding"].ravel()
                                 for im in images]).astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def pack(sub: pd.DataFrame, emb: np.ndarray, with_split: bool) -> dict:
    """The npz payload, matching `embed_catalog` / `embed_background` exactly.

    Key sets are load-bearing and fail silently if wrong: `build_heads` reads
    `split` to pick the training rows, and `load_background` filters on
    `species_id` and `species_name`. A missing key raises far from here, at head
    build time, with nothing pointing back to this function.
    """
    data = {
        "descriptor": emb,
        "image_id": np.asarray(sub["image_id"], dtype=str),
        "species_id": np.asarray(sub["species_id"], dtype=str),
        "species_name": np.asarray(sub["species_name"], dtype=str),
    }
    if with_split:
        data["split"] = np.asarray(sub["split"], dtype=str)
    return data


def _organ_caches(model, manifest, cache_path_fn, variant, cache_dir, with_split, tag):
    idx = pd.read_parquet(cache_dir / manifest)
    idx = idx[idx["local_path"].notna()].reset_index(drop=True)
    for organ in ORGANS:
        out = cache_path_fn(organ, variant, cache_dir)
        if out.exists():
            print(f"{tag}[{organ}]: cached", flush=True)
            continue
        sub = idx[idx["organ"] == organ].reset_index(drop=True)
        if sub.empty:
            continue
        emb = embed_paths(model, [str(cache_dir / p) for p in sub["local_path"]],
                          desc=f"{tag}[{organ}]")
        np.savez_compressed(out, **pack(sub, emb, with_split))
        print(f"{tag}[{organ}]: {emb.shape} -> {out.name}", flush=True)


def catalog(model, variant, cache_dir=DATA_PROCESSED):
    from plantid.features.embed_catalog import cache_path

    _organ_caches(model, "catalog_index.parquet", cache_path, variant, cache_dir,
                  with_split=True, tag="cat")


def background(model, variant, cache_dir=DATA_PROCESSED):
    from plantid.features.embed_background import cache_path

    _organ_caches(model, "plantnet_background.parquet", cache_path, variant, cache_dir,
                  with_split=False, tag="bg")


def inat(model, variant, cache_dir=DATA_PROCESSED):
    """Keyed by file path rather than row, since an observation has many photos."""
    from plantid.features.embed_inat import MANIFEST, cache_path

    out = cache_path(variant, cache_dir)
    if out.exists():
        print("inat: cached", flush=True)
        return
    df = pd.read_parquet(cache_dir / MANIFEST)
    paths = [p for ps in df["local_paths"] for p in ps]
    emb = embed_paths(model, paths, desc="inat")
    np.savez_compressed(out, descriptor=emb, path=np.asarray(paths, dtype=str))
    print(f"inat: {emb.shape} -> {out.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the .mlpackage under test")
    ap.add_argument("--variant", required=True, help="cache variant string to write")
    ap.add_argument("--targets", default="catalog,background,inat")
    ap.add_argument("--compute-units", default="CPU_AND_NE",
                    help="pinning this away from CPU_AND_NE is almost certainly a mistake")
    args = ap.parse_args()

    import coremltools as ct

    units = getattr(ct.ComputeUnit, args.compute_units)
    print(f"model {Path(args.model).name}  units {args.compute_units}  "
          f"variant {args.variant}", flush=True)
    model = load_model(args.model, units)
    for target in args.targets.split(","):
        {"catalog": catalog, "background": background, "inat": inat}[target](
            model, args.variant)


if __name__ == "__main__":
    main()
