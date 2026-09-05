"""Depth-prune BioCLIP-2, then distil the truncation back toward the teacher.

`LAYER_FINDINGS.md` already gives the zero-shot cost of dropping blocks, because
probing layer k *is* truncating to k blocks and refitting the head:

    24 blocks 0.9754   22 blocks 0.9532   20 blocks 0.9017   18 blocks 0.8107

So the only open question is whether distillation recovers that. This trains the
last `n_train` surviving blocks of the truncated tower to reproduce the *full*
teacher's final embedding, then refits the head and re-measures.

Targets are cached teacher embeddings, so the teacher never runs during training
-- the same trick `plantid/train/distil.py` used, and the reason this is minutes
rather than hours.

Usage:
    PYTHONPATH=. .venv-mps/bin/python -m analysis.prune_distill --blocks 20 22
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

SET = Path("/private/tmp/claude-501/-Users-jameskelly-Documents-plantid/"
           "8f28d0ec-b6fd-4bd9-9ee3-a9ec10992c6b/scratchpad/common")


# ---- structure ------------------------------------------------------------

def truncate(visual, n_blocks):
    """Keep the first `n_blocks` residual blocks. Structural depth pruning."""
    visual.transformer.resblocks = visual.transformer.resblocks[:n_blocks]
    return visual


def tower_params(visual):
    return sum(p.numel() for p in visual.parameters())


# ---- data -----------------------------------------------------------------

def load_targets():
    """Cached full-teacher embeddings, plus labels and observation clusters."""
    z = np.load(SET / "layers_bioclip2.npz", allow_pickle=True)
    return (z["final"].astype("float32"), z["label"].astype(str),
            z["cluster"].astype(str))


def load_pixels(preprocess, device, batch=32):
    """Preprocessed image batches, in manifest order so rows align with targets."""
    import pandas as pd
    from PIL import Image
    df = pd.read_parquet(SET / "manifest.parquet").drop_duplicates("local_path")
    paths = list(df.reset_index(drop=True).local_path)
    for s in range(0, len(paths), batch):
        ims = []
        for p in paths[s:s + batch]:
            with Image.open(p) as im:
                ims.append(preprocess(im.convert("RGB")))
        yield s, torch.stack(ims).to(device)


# ---- forward --------------------------------------------------------------

def cls_forward(visual, px):
    """CLS token after the surviving blocks — the truncated tower's output."""
    x = visual.conv1(px)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = visual.transformer(x)
    return x[:, 0]


def embed_all(visual, head, preprocess, device, n_rows, dim, batch=32):
    """Truncated tower + adapter over the whole set -> (n_rows, dim)."""
    out = np.empty((n_rows, dim), dtype="float32")
    with torch.no_grad():
        for s, px in load_pixels(preprocess, device, batch):
            v = head(cls_forward(visual, px))
            out[s:s + len(v)] = v.float().cpu().numpy()
    return out


# ---- evaluation -----------------------------------------------------------

def probe(X, y, clusters, seeds=6):
    """Mean top-1 of a logistic head, split by cluster so no subject straddles."""
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    acc = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        u = np.array(sorted(set(clusters)))
        rng.shuffle(u)
        tr = np.isin(clusters, u[: len(u) // 2])
        clf = LogisticRegression(max_iter=3000, C=10.0,
                                 class_weight="balanced").fit(X[tr], y[tr])
        acc.append((clf.predict(X[~tr]) == y[~tr]).mean())
    return float(np.mean(acc)), float(np.std(acc))


# ---- distillation ---------------------------------------------------------

def distil(visual, adapter, targets, preprocess, device, n_train, epochs, lr):
    """Train the last `n_train` blocks plus the adapter to match the teacher.

    Cosine loss against the L2-normalised teacher embedding, because that is the
    quantity the downstream head actually consumes.
    """
    for p in visual.parameters():
        p.requires_grad_(False)
    trainable = list(adapter.parameters())
    for blk in visual.transformer.resblocks[-n_train:]:
        for p in blk.parameters():
            p.requires_grad_(True)
        trainable += list(blk.parameters())

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    T = torch.from_numpy(targets).to(device)
    T = T / T.norm(dim=1, keepdim=True)

    for ep in range(epochs):
        tot = n = 0
        for s, px in load_pixels(preprocess, device):
            v = adapter(cls_forward(visual, px))
            v = v / v.norm(dim=1, keepdim=True)
            loss = (1 - (v * T[s:s + len(v)]).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.detach().item() * len(v); n += len(v)
        print(f"    epoch {ep + 1}/{epochs}  cosine loss {tot / n:.4f}", flush=True)
    return visual, adapter


# ---- driver ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", type=int, nargs="+", default=[20, 22])
    ap.add_argument("--train-blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()

    from plantid.features.pretrained import load_encoder
    targets, y, clusters = load_targets()
    full_acc, full_sd = probe(targets, y, clusters)

    model, preprocess, device = load_encoder("bioclip2")
    full_params = tower_params(model.clip.visual)
    print(f"\nteacher: 24 blocks, {full_params/1e6:.1f}M params, "
          f"top-1 {full_acc:.4f} (sd {full_sd:.4f})\n")
    del model

    for n in a.blocks:
        model, preprocess, device = load_encoder("bioclip2")
        visual = truncate(model.clip.visual, n)
        params = tower_params(visual)
        width = visual.ln_pre.normalized_shape[0]
        adapter = nn.Linear(width, targets.shape[1], bias=False).to(device)
        nn.init.zeros_(adapter.weight)
        with torch.no_grad():                       # start from the teacher's projection
            adapter.weight.copy_(model.clip.visual.proj.T.to(device)
                                 if hasattr(visual, "proj") and visual.proj is not None
                                 else adapter.weight)

        before = embed_all(visual, adapter, preprocess, device, len(y), targets.shape[1])
        b_acc, b_sd = probe(before, y, clusters)
        print(f"{n} blocks  {params/1e6:6.1f}M  ({100*params/full_params:.0f}% of teacher, "
              f"{params*4/8/1e6:.0f} MB int4)")
        print(f"    before distillation  {b_acc:.4f} (sd {b_sd:.4f})", flush=True)

        distil(visual, adapter, targets, preprocess, device,
               a.train_blocks, a.epochs, a.lr)
        after = embed_all(visual, adapter, preprocess, device, len(y), targets.shape[1])
        a_acc, a_sd = probe(after, y, clusters)
        gap = full_acc - b_acc
        print(f"    after  distillation  {a_acc:.4f} (sd {a_sd:.4f})   "
              f"recovered {100*(a_acc-b_acc)/gap if gap > 0 else float('nan'):.0f}% "
              f"of a {100*gap:.1f}pp gap\n", flush=True)
        del model, visual, adapter


if __name__ == "__main__":
    main()
