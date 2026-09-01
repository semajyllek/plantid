"""Core ML export of the frozen image encoder, and what it costs.

Every accuracy figure in this repo comes from PyTorch embeddings cached as
`.npz`. The deployment claim — BioCLIP v1 at 43 MB and ~22 ms — is MPS on an
M4 Max, which is neither Core ML nor the Neural Engine. This module closes that
gap, and it exists as a module rather than a script because two of the things it
checks are silent failures:

**Preprocessing drift.** `open_clip` resizes bicubic to 224, centre-crops, and
normalises with CLIP's per-channel mean/std. A Core ML model fed differently
normalised pixels still returns a plausible embedding, and nothing in an offline
evaluation would notice. So normalisation is *baked into the traced graph*
rather than left to the caller: the exported model takes an ordinary image and
`ct.ImageType(scale=1/255)` handles only the 0-255 → 0-1 step, which is the one
part a scalar scale can express exactly (the three channel deviations differ, so
a shared scale cannot).

**Silent ANE fallback.** A single unsupported op can push the whole graph to
GPU or CPU while still producing correct output. The latency budget is the thing
that would be wrong, not the answer, so `compute_plan` reports where the ops
actually landed rather than trusting `compute_units=ALL` to mean what it says.

The exported model also L2-normalises its output, because that is what
`inat_fusion._l2` does before the head sees it — doing it in-graph removes a
step the app would otherwise have to reproduce exactly.

Usage:
    PYTHONPATH=. .venv-mps/bin/python -m plantid.deploy.coreml --variant bioclip1
"""

import argparse
import time
from pathlib import Path

import numpy as np

from plantid.config import DATA_PROCESSED

# open_clip's CLIP normalisation, and the resolution BioCLIP v1 expects.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIDE = 224
OUT_DIR = DATA_PROCESSED / "coreml"


def build_traceable(variant: str = "bioclip1"):
    """The image tower plus normalisation and L2, as one traceable module.

    Returns (module, embed_dim). The module takes float pixels in [0, 1] with
    shape (N, 3, 224, 224) -- the `ct.ImageType(scale=1/255)` output -- so the
    only preprocessing left outside the graph is resize and centre-crop.
    """
    import torch
    from torch import nn

    from plantid.features.pretrained import load_encoder

    tower, _, _ = load_encoder(variant, device="cpu")
    visual = tower.clip.visual if hasattr(tower, "clip") else tower

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = visual
            self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

        def forward(self, x):
            x = (x - self.mean) / self.std
            e = self.visual(x)
            return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    # `nn.MultiheadAttention` takes a fused fast path in eval mode that traces as
    # a single `_native_multi_head_attention` node, which the Core ML converter
    # has no implementation for. Disabling it makes the block trace as its
    # constituent matmuls and softmax -- the same arithmetic, and the form the
    # ANE can actually schedule.
    torch.backends.mha.set_fastpath_enabled(False)

    model = Encoder().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        dim = int(model(torch.zeros(1, 3, SIDE, SIDE)).shape[-1])
    return model, dim


def export(variant: str = "bioclip1", out_dir: Path = OUT_DIR, palettize_bits: int | None = None,
           granularity: str = "per_grouped_channel", target: str = "iOS18"):
    """Convert and write an .mlpackage. Returns (path, embedding_dim).

    `granularity` only bites when palettizing. `per_grouped_channel` needs an
    iOS18 deployment target; `per_tensor` works on iOS17 and is the fallback
    whose cost is worth knowing before committing to an OS floor."""
    import coremltools as ct
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    model, dim = build_traceable(variant)

    # `torch.jit.trace` fails on this graph under torch 2.13: the ViT emits an
    # `aten::Int` on a non-scalar shape that the Core ML frontend rejects.
    # `torch.export` produces a cleaner ATen graph and converts, but only after
    # `run_decompositions` -- the raw export is in the TRAINING dialect, which
    # the converter refuses.
    exported = torch.export.export(model, (torch.rand(1, 3, SIDE, SIDE),))
    exported = exported.run_decompositions({})

    mlmodel = ct.convert(
        exported,
        inputs=[ct.ImageType(name="image", shape=(1, 3, SIDE, SIDE), scale=1 / 255.0,
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=getattr(ct.target, target),
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    suffix = f"_int{palettize_bits}_{granularity}" if palettize_bits else ""
    if palettize_bits:
        from coremltools.optimize.coreml import (
            OpPalettizerConfig, OptimizationConfig, palettize_weights,
        )
        # per-channel scale + a k-means palette: ONDEVICE_FINDINGS records that
        # a per-tensor palette with linear centroid init destroys the embedding
        # (cosine 0.214), so the granularity is not incidental.
        kw = {"granularity": granularity}
        if granularity == "per_grouped_channel":
            kw["group_size"] = 1
        cfg = OptimizationConfig(global_config=OpPalettizerConfig(
            nbits=palettize_bits, mode="kmeans", **kw))
        mlmodel = palettize_weights(mlmodel, cfg)

    path = out_dir / f"{variant}{suffix}.mlpackage"
    mlmodel.save(str(path))
    return path, dim


def _pil_batch(paths):
    from PIL import Image
    import torchvision.transforms as T

    # match open_clip: bicubic resize of the short side, then centre crop
    resize = T.Compose([T.Resize(SIDE, interpolation=T.InterpolationMode.BICUBIC),
                        T.CenterCrop(SIDE)])
    return [resize(Image.open(p).convert("RGB")) for p in paths]


def validate(path: Path, variant: str = "bioclip1", paths=None, n: int = 64):
    """Cosine agreement between Core ML and PyTorch on real catalogue images.

    Random noise would not catch a preprocessing bug -- normalisation errors show
    up as a systematic rotation that only real image statistics reveal -- so this
    deliberately uses photographs from the catalogue.
    """
    import coremltools as ct
    import torch
    import torchvision.transforms.functional as TF

    if paths is None:
        import pandas as pd
        cat = pd.read_parquet(DATA_PROCESSED / "catalog_index.parquet")
        # `local_path` is stored relative to the processed-data directory
        paths = [DATA_PROCESSED / p
                 for p in cat["local_path"].dropna().sample(n, random_state=0)]

    images = _pil_batch(paths)
    model, _ = build_traceable(variant)
    with torch.no_grad():
        ref = model(torch.stack([TF.to_tensor(im) for im in images])).numpy()

    mlmodel = ct.models.MLModel(str(path))
    got = np.stack([mlmodel.predict({"image": im})["embedding"].ravel() for im in images])

    cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
    return {"n": len(images), "cosine_mean": float(cos.mean()), "cosine_min": float(cos.min()),
            "max_abs_err": float(np.abs(ref - got).max())}


def benchmark(path: Path, repeats: int = 50, warmup: int = 5):
    """Latency per compute-unit setting, plus where the ops actually ran."""
    import coremltools as ct
    from PIL import Image

    img = Image.fromarray((np.random.rand(SIDE, SIDE, 3) * 255).astype("uint8"))
    rows = []
    for name, units in (("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
                        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
                        ("ALL", ct.ComputeUnit.ALL)):
        try:
            m = ct.models.MLModel(str(path), compute_units=units)
            for _ in range(warmup):
                m.predict({"image": img})
            t0 = time.perf_counter()
            for _ in range(repeats):
                m.predict({"image": img})
            rows.append({"compute_units": name,
                         "ms_per_image": (time.perf_counter() - t0) / repeats * 1000})
        except Exception as exc:  # a unit may be unavailable on this host
            rows.append({"compute_units": name, "ms_per_image": float("nan"),
                         "error": type(exc).__name__})
    return rows


def dispatch(path: Path):
    """Per-op device assignment. `compute_units=ALL` is a request, not a promise."""
    import coremltools as ct

    # MLComputePlan needs a *compiled* `.mlmodelc`; handing it the `.mlpackage`
    # aborts the process from C++ rather than raising, so compile explicitly.
    # `loaded` must stay in scope: the compiled directory is a temp the MLModel
    # owns, and letting it be collected deletes the path out from under us.
    loaded = ct.models.MLModel(str(path))
    compiled = loaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=compiled, compute_units=ct.ComputeUnit.ALL)

    counts: dict[str, int] = {}
    program = plan.model_structure.program
    for func in program.functions.values():
        for op in func.block.operations:
            info = plan.get_compute_device_usage_for_mlprogram_operation(op)
            # ops with no usage info are consts and other non-compute nodes
            dev = (type(info.preferred_compute_device).__name__
                   .removeprefix("ML").removesuffix("ComputeDevice")) if info else "none"
            counts[dev] = counts.get(dev, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="bioclip1")
    ap.add_argument("--bits", type=int, default=None, help="palettize weights to N bits")
    ap.add_argument("--granularity", default="per_grouped_channel",
                    choices=("per_grouped_channel", "per_tensor"))
    ap.add_argument("--target", default="iOS18", help="minimum deployment target")
    ap.add_argument("--n-validate", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()

    path, dim = export(args.variant, palettize_bits=args.bits,
                       granularity=args.granularity, target=args.target)
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    print(f"exported {path}  ({size:.1f} MB on disk, embedding dim {dim})", flush=True)

    print("\nvalidation vs PyTorch on real catalogue images:")
    print(" ", validate(path, args.variant, n=args.n_validate), flush=True)

    print("\nlatency:")
    for row in benchmark(path, repeats=args.repeats):
        ms = row["ms_per_image"]
        print(f"  {row['compute_units']:12s} {ms:7.2f} ms" if ms == ms
              else f"  {row['compute_units']:12s}   n/a ({row.get('error')})")

    print("\nops by assigned compute device:")
    print(" ", dispatch(path))


if __name__ == "__main__":
    main()
